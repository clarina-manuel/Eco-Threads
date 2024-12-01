// Drag & Drop functionality
const dropTarget = document.getElementById('dropTarget');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const resultCard = document.getElementById('resultCard');
const resultLabel = document.getElementById('resultLabel');
const resultConfidence = document.getElementById('resultConfidence');

// Handle file upload or drag-and-drop
fileInput.addEventListener('change', handleFileUpload);
dropTarget.addEventListener('dragover', handleDragOver);
dropTarget.addEventListener('drop', handleFileDrop);

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        previewImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
}

function handleFileDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
        previewImage(file);
    }
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        classifyImage(file);
    };
    reader.readAsDataURL(file);
}

function classifyImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction !== undefined) {
            displayResults(data.prediction);
        } else {
            alert('Error in prediction');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function displayResults(prediction) {
    resultCard.style.display = 'block';
    resultLabel.textContent = 'Category: ' + prediction;

    resultConfidence.textContent = ''; //clear previous tips
    console.log('Predicted class:', prediction);

    let message = ''; //unique tips for each category
        if (prediction === "Apparel") {
            message = `🌟 Braid Your Apparel into Baskets, Bowls, or Rugs 🧺 <br> Repurpose your old apparel into stylish and functional items like woven baskets or cozy rugs—perfect for adding a handmade touch to your home. 
            <br><br> 🌸 Create Trendy Headbands 💁‍♀️ <br> Turn your old apparel into cute and comfortable headbands that add a bit of flair to your everyday look. 
            <br><br> 🌿 Make Unique Plant Hangers 🌱 <br> Give your plants a stylish new home by transforming your apparel into fun, boho-inspired plant hangers. 
            <br><br> 💪 Upcycle Your Apparel into Workout Gear 🏋️‍♀️ <br> Repurpose your old clothes into new workout shirts or tanks that are both functional and eco-friendly. 
            <br><br> 🐾 Create Fun Dog Toys 🐕 <br> Transform your worn-out apparel into simple, durable toys for your furry friend. It's a great way to reuse and keep your pup entertained! 
            <br><br> 🛋️ Sew Your Apparel into Patchwork Pillows 🧵 <br> Gather your old apparel and sew it into a patchwork pillow. It's a great way to add a personalized touch to your home decor!`;
        } else if (prediction === "Accessories") {
            message = `🌟 Turn Old Necklaces into Chic Bracelets 💎 <br> Repurpose old, tangled necklaces or chains into stylish bracelets. Layer them for a boho-inspired look or create a delicate piece with one necklace.  
            <br><br> 💕 Upcycle Old Scarves into Trendy Headbands or Hair Ties 🧣 <br> Transform old scarves into headbands or hair ties! Cut them into strips and tie them around your hair for a personalized, fashionable accessory. 
            <br><br> 👜 Make Vintage Brooches into Unique Bag Charms 💍 <br> Use old brooches or pins to create beautiful bag charms. Attach them to purse straps or zippers for a chic, personalized touch. 
            <br><br> 🎨 Repurpose Earrings into Wall Art or Home Décor 🖼️ <br> Old, mismatched earrings can be turned into unique art pieces. Glue them to a canvas or frame for an eco-friendly, personalized decor item that adds character to your home.`;
        } else if (prediction === "Footwear") {
            message = `👟 Turn Old Sneakers into Trendy Plant Holders 🌱 <br> Repurpose your worn-out sneakers into stylish plant holders! Cut off the tops and use the soles as unique pots for small plants. Paint them or add decorative touches to match your home decor. 
            <br><br> 🧑‍🔬 Upcycle Old Boots into Functional Storage Containers 🎁 <br> Convert your old boots into fun storage solutions! Use them to store small items like keys, gloves, or even as pencil holders for your desk. Add some fabric or decorative details for extra flair! 
            <br><br> 🏖️ Transform Sandals into Colorful Keychains or Bag Charms 🏝️ <br> Take the straps from old sandals and turn them into colorful keychains or bag charms! A perfect way to reuse materials and add a fun, eco-friendly accessory to your keys or bags. 
            <br><br> 👢 Turn Worn-Out High Heels into a Jewelry Display Stand 💍 <br> Repurpose old high heels into a chic jewelry display! Simply attach a wooden board or tray to the heel base, and use it to hang necklaces or display earrings in style.`;
        } else if (prediction === "Personal Care") {
            message = `🛁 Turn Old Shampoo Bottles into Craft Containers 🎨 <br> Repurpose your empty shampoo or conditioner bottles as storage containers for small craft supplies like beads, buttons, or threads. Decorate them with washi tape or fabric for a fun and eco-friendly storage solution. 
            <br><br> 💄 Transform Makeup Palettes into Custom Jewelry Trays 💍 <br> Upcycle your old makeup palettes (especially the ones with broken powders or unused shades) into personalized jewelry trays! Just remove the makeup, add a lining, and use it to store rings, earrings, or other small trinkets. 
            <br><br> 🧴 Repurpose Fragrance Bottles as Elegant Flower Vases 💐 <br> Old fragrance bottles can make for gorgeous, unique vases! Clean them out, add a bit of water, and place a small flower or two inside. Perfect for an eco-friendly centerpiece! 
            <br><br> 🧖‍♀️ Turn Old Toothbrushes into Household Cleaning Tools 🧹 <br> Don't throw away those old toothbrushes! Instead, use them for cleaning hard-to-reach places like grout lines, jewelry, or even as a mini scrub brush for shoes or handbags. A great way to reuse and minimize waste!`;
        } else if (prediction === "Sporting Goods") {
            message = `⚽ Repurpose Old Sports Jerseys into Trendy Tote Bags 👜 <br> Turn your old sports jerseys into stylish tote bags! Simply cut, sew, and add some straps, and you've got a unique, sporty bag to carry your essentials. Show off your team spirit while keeping things functional! 
            <br><br> 🏀 Create Cool Headbands or Wristbands 🏋️‍♀️ <br> Use old athletic wear like leggings or jerseys to craft custom headbands or wristbands. Not only are they super comfy, but they also make for a sporty, stylish addition to your workout gear. 
            <br><br> 🏈 Make a Recycled Sports-Themed Pillow 🛋️ <br> Gather up worn-out jerseys or shorts and sew them into a comfy, sports-themed pillow. Whether it's for your couch or your room, it's a fun way to repurpose old gear while showing off your love for the game! 
            <br><br> 🏅 Transform Old Sneakers into Plant Holders 🌱 <br> If your sneakers are worn out, don't toss them—repurpose them into planters! Just cut holes in the soles for drainage, add soil, and plant some succulents or small flowers.`;
        } else if (prediction === "Home") {
            message = `🛋️ Turn Old Bedding into Trendy Throw Pillows 🧵 <br> Repurpose your old duvet covers, sheets, or pillowcases into stylish throw pillows! Just cut and sew them into fun shapes and sizes to add a cozy, custom touch to your living room or bedroom. 
            <br><br> 🪑 Create Unique Upholstery for Chairs or Stools 🎨 <br> Use old blankets, throws, or fabric scraps to reupholster your worn-out chairs or stools. This is a great way to breathe new life into furniture while creating a personalized design for your home. 
            <br><br> 🧣 Craft Cozy Scarves from Old Towels 🛁 <br> Repurpose thick towels or old bathrobes into soft, cozy scarves! Cut and sew them into fashionable wraps or scarves perfect for colder months. Not only are they comfy, but they're eco-friendly too! 
            <br><br> 🏡 Make a Quilted Wall Hanging or Tapestry 🌿 <br> Use your old sheets, pillowcases, or even leftover fabric to create a beautiful, quilted wall hanging. Mix and match patterns and colors to make an eye-catching tapestry that adds a unique touch to any room.`;
        } else if (prediction === "Free Items") {
            message = `🎨 Turn Free T-Shirts into Custom Tote Bags 👜 <br> Repurpose those free promotional t-shirts you've collected over the years into stylish tote bags! Simply cut and sew them into a simple, functional bag for shopping or carrying your essentials. 
            <br><br> 🧢 Transform Hats into Planters or Storage Bins 🌱 <br> Old baseball caps or any other hats can be turned into quirky plant pots or storage bins. Just add a bit of soil and a small plant to create a fun garden display! 
            <br><br> 📚 Upcycle Free Books into Decorative Wall Art 🎨 <br> Use old or free books to create unique wall art by cutting out pages with beautiful designs, words, or quotes, then framing them for an artsy vibe in your space. 
            <br><br> 🏷️ Repurpose Free Swag into Fashion Accessories 🧷 <br> If you've received free bracelets, pins, or other trinkets, repurpose them into custom fashion accessories like keychains, jewelry, or even embellishments for your clothes!`;
        } else {
            message = `Sorry, this item doesn't fit into one of our categories. But we're sure your creativity can come up with a way to upcycle this!`;
        }
    
        // Update the message text content with line breaks
        resultConfidence.innerHTML = message; // Use innerHTML to allow <br> tags to be rendered properly
    }

