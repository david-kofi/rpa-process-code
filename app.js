// app.js

const express = require('express');
const tf = require('@tensorflow/tfjs'); // TensorFlow.js for Node.js
const fs = require('fs');
const jpeg = require('jpeg-js');
const mobilenet = require('@tensorflow-models/mobilenet');
const { Pinecone } = require('@pinecone-database/pinecone');
const axios = require('axios'); // For downloading images
require('dotenv').config(); // For environment variables

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Initialize Pinecone client
const pc = new Pinecone({
    apiKey:"pcsk_33ucs9_DKggDFT8mT3miMN25r3VUBkaLVnCnCEqBZK2JPjizCg4DqhYtsArvbQXb4vtyjG"
});

// Reference to the Pinecone index
let index;

// Load MobileNet model
let model;

// Function to download image from URL and return buffer
async function downloadImage(url) {
    try {
        const response = await axios.get(url, { responseType: 'arraybuffer' });
        return Buffer.from(response.data, 'binary');
    } catch (err) {
        throw new Error(`Failed to download image from URL: ${err.message}`);
    }
}

// Function to convert image buffer to tensor
function imageToTensor(imageBuffer) {
    const imageData = jpeg.decode(imageBuffer, true);

    const rgbArray = [];
    const data = imageData.data; // RGBA format
    for (let i = 0; i < data.length; i += 4) {
        rgbArray.push(data[i]);     // R
        rgbArray.push(data[i + 1]); // G
        rgbArray.push(data[i + 2]); // B
        // Skip Alpha channel
    }

    // Convert to float32
    return tf.tensor3d(
        rgbArray,
        [imageData.height, imageData.width, 3],
        'float32' // Ensure the tensor is float32
    );
}

// Function to preprocess image for MobileNet
function preprocessImage(image) {
    return tf.tidy(() => {
        const resized = tf.image.resizeBilinear(image, [224, 224]); // Resize to 224x224
        const normalized = resized.div(tf.scalar(255.0)); // Normalize pixel values to [0, 1]
        return normalized.expandDims(0); // Add batch dimension
    });
}

// Function to extract features (embeddings) from an image
async function extractFeatures(imageBuffer) {
    const image = imageToTensor(imageBuffer);
    const processedImage = preprocessImage(image);
    const embeddings = model.infer(processedImage, { pooling: 'avg' }); // Extract embeddings
    return embeddings;
}

// Function to upsert vector into Pinecone
async function upsertImageVector(vector, id, namespace = '') {
    try {
        // Validate vector is array
        if (!Array.isArray(vector)) {
            throw new TypeError("Vector must be an array of numbers.");
        }

        // Validate vector length matches index dimension (1280)
        if (vector.length !== 1280) {
            throw new Error(`Vector length (${vector.length}) does not match index dimension (1280).`);
        }

        // Validate vector elements are numbers
        if (!vector.every(num => typeof num === 'number')) {
            throw new TypeError("All elements in the vector must be numbers.");
        }

        // Prepare upsert payload
        const upsertPayload = {
            vectors: [
                {
                    id: id,
                    values: vector,
                    metadata: { source: "image" }
                }
            ],
            namespace: namespace
        };

        console.log('Upserting vectors:', upsertPayload.vectors);

        const response = await index.upsert(upsertPayload);

        console.log('Upsert response:', response);
        return response;
    } catch (err) {
        console.error('Error during upsert:', err);
        throw err;
    }
}

// POST endpoint to upload image vector
app.post('/upload', async (req, res) => {
    const { imageUrl, id } = req.body;

    // Input validation
    if (!imageUrl || !id) {
        return res.status(400).json({ error: "Missing 'imageUrl' or 'id' in request body." });
    }

    try {
        // Download image
        const imageBuffer = await downloadImage(imageUrl);
        console.log(`Image downloaded from URL: ${imageUrl}`);

        // Extract features
        const embeddings = await extractFeatures(imageBuffer);
        const vectorArray = Array.from(embeddings.dataSync()); // Convert tensor to array

        console.log(`Embedding vector length for ${id}:`, vectorArray.length);
        console.log(`Embedding vector for ${id} (first 10 values):`, vectorArray.slice(0, 10));

        // Upsert vector into Pinecone
        await upsertImageVector(vectorArray, id);
        console.log(`Vector for ${id} upserted successfully!`);

        // Respond with success
        res.status(200).json({ message: `Vector for ${id} upserted successfully!` });
    } catch (err) {
        console.error('Error processing and inserting vector:', err);
        res.status(500).json({ error: err.message });
    }
});

// Function to initialize Pinecone and model, then start server
async function initialize() {
    try {
        
        // Reference the Pinecone index
        index = pc.Index('adv-vector-research'); // Ensure the index exists with correct dimension (1280)

        // Load MobileNet model
        model = await mobilenet.load({ version: 2, alpha: 1.0 });
        console.log("MobileNet model loaded successfully.");

        // Start the Express server
        app.listen(PORT,'0.0.0.0', () => {
            console.log(`Server is running on port ${PORT}`);
        });
    } catch (err) {
        console.error("Error during initialization:", err);
        process.exit(1); // Exit if initialization fails
    }
}

// Initialize Pinecone and model, then start server
initialize();
