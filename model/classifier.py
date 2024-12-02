from collections import Counter
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from datasets import DatasetDict, Dataset, load_dataset
from transformers import ViTImageProcessor, AutoModelForImageClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import requests
import os
import random
from sklearn.utils import shuffle
import torch.optim as optim

#load dataset
ds = load_dataset("ashraq/fashion-product-images-small")
print(ds)

#clean the dataset (only image and label)
ds_cleaned = ds['train'].remove_columns(['id', 'gender', 'subCategory', 'productDisplayName', 'baseColour', 'season', 'year', 'usage', 'articleType'])
ds_cleaned


#loading and splitting the clothes dataset
split = ds_cleaned.train_test_split(test_size=0.2) #80/20 split

train_data = split['train']
test_data = split['test']


#label encoder for 'masterCategory' labels
label_encoder = LabelEncoder()

#preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#custom dataset class
class FashionDataset(Dataset):
    def __init__(self, dataset, transform=None, undersample=False, max_class_0_samples=8000):
        self.dataset = dataset
        self.transform = transform
        self.image_column = 'image'
        self.label_column = 'masterCategory'
        self.undersample = undersample
        self.max_class_0_samples = max_class_0_samples

        #undersampling
        if self.undersample:
            self.dataset = self.undersample_data(self.dataset)

        #label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([sample[self.label_column] for sample in self.dataset])  # Fit label encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column]
        label = self.dataset[idx][self.label_column]

        #label encoding
        label = self.label_encoder.transform([label])[0]

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label
    
    def undersample_data(self, dataset):
        #separate apparel class (class 0) samples
        class_0_samples = [sample for sample in dataset if sample[self.label_column] == 'Apparel']
        other_class_samples = [sample for sample in dataset if sample[self.label_column] != 'Apparel']

        #undersample apparel class to 8000 samples + combine + shuffle
        class_0_samples = random.sample(class_0_samples, min(len(class_0_samples), self.max_class_0_samples))
        undersampled_data = class_0_samples + other_class_samples

        undersampled_data = shuffle(undersampled_data, random_state=42)

        return undersampled_data
    

#train/test datasets
train_dataset = FashionDataset(
    dataset=train_data,
    transform=transform,
    undersample=True,
    max_class_0_samples=8000
)

test_dataset = FashionDataset(test_data, transform=transform)


#distribution after undersampling
class_distribution = {}
for sample in train_dataset.dataset:
    label = sample['masterCategory']
    if label not in class_distribution:
        class_distribution[label] = 0
    class_distribution[label] += 1

print("Class distribution after undersampling:", class_distribution)

#train/test dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, pin_memory=True)

#training function
def train_model(model, train_dataloader, optimizer, criterion, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            pixel_values, labels = batch

            pixel_values = pixel_values.to(device)  #move image data to device
            labels = labels.to(device)  #move labels to device

            # forward pass: predict
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # backward pass: evaluate losses and optimize weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        accuracy = 100 * correct_preds / total_preds
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_dataloader)}, Accuracy = {accuracy}%")


class_frequencies = list(class_distribution.values())
class_weights = [1.0 / freq for freq in class_frequencies]
class_weights = torch.tensor(class_weights).float()

print("Class Weights:", class_weights)

#initialize the model
model_name = "google/vit-base-patch16-224-in21k"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

#optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

unique_labels = set(train_data['masterCategory'])
print(f"Unique labels: {unique_labels}")

#train the model
train_model(model, train_dataloader, optimizer, criterion, device, epochs=1)

#evaluate the model
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            pixel_values = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            outputs = model(pixel_values)
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

#evaluate
evaluate_model(model, test_dataloader, device)

#save the model
model_dir = '/Users/clarina-manuel/Documents/image_classification_webapp'
model_path = os.path.join(model_dir, 'fashion_classifier.pth')

#check directory
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model.state_dict(), model_path)
print(f"Model saved at: {model_path}")
