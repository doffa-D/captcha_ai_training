const fs = require('fs');
const svgCaptcha = require('svg-captcha');
const sharp = require('sharp');
const crypto = require('crypto');
const path = require('path');

const SECRET_KEY = '1d4c4f32f4b3e6e09e5bcb5d8907b711ad6c8c4b28b58a55d7b3d34216b28f68'; // 64-character hex string (32 bytes)
const ALGORITHM = 'aes-256-cbc'; // Define the encryption algorithm
const IV_LENGTH = 16; // For AES, this is always 16 bytes

// Function to encrypt text
function encrypt(text) {
    const iv = crypto.randomBytes(IV_LENGTH); // Generate a random 16-byte IV
    const cipher = crypto.createCipheriv(ALGORITHM, Buffer.from(SECRET_KEY, 'hex'), iv);

    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    // Concatenate the IV and encrypted text without any separator
    return iv.toString('hex') + encrypted;
}

// Function to decrypt text
function decrypt(encryptedText) {
    // Extract the IV from the first 32 characters (16 bytes)
    const iv = Buffer.from(encryptedText.substring(0, 32), 'hex');
    
    // The rest of the string is the encrypted text
    const encryptedTextBuffer = Buffer.from(encryptedText.substring(32), 'hex');
    
    const decipher = crypto.createDecipheriv(ALGORITHM, Buffer.from(SECRET_KEY, 'hex'), iv);
    let decrypted = decipher.update(encryptedTextBuffer, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
}

// Load the custom font (Quantico-Regular.ttf)
svgCaptcha.loadFont('./Quantico-Regular.ttf');

// Custom set of characters, including symbols
const charPreset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?';

// Generate the CAPTCHA with the custom font and custom characters
for (let i = 0; i < 1; i++) {
    const captcha = svgCaptcha.create({
        size: 6,
        noise: 6,
        color: true,
        width: 250,
        height: 70,
        fontSize: 56,
        charPreset: charPreset
    });

    // console.log('Original CAPTCHA text:', captcha.text);

    // Encrypt the CAPTCHA text
    console.log('Encrypted CAPTCHA text:', captcha.text);
    const encryptedText = encrypt(captcha.text);
    // console.log('Encrypted CAPTCHA text:', encryptedText);
    // const decryptedText = decrypt(encryptedText);
    // console.log('Decrypted CAPTCHA text:', decryptedText);

    // Create a safe filename using a hash of the original text
    // const safeFilename = crypto.createHash('md5').update(encryptedText).digest('hex'); // Hash of the encrypted text

    // Create the file path
    const filePath = path.join('images/', `${encryptedText}.png`);

    // Convert the SVG to PNG using Sharp
    sharp(Buffer.from(captcha.data))
        .png()
        .toFile(filePath, (err, info) => {
            if (err) {
                console.error('Error converting CAPTCHA to PNG:', err);
            }
        });
}

// Example usage of decrypting a CAPTCHA text
// Call decrypt(encryptedText) where encryptedText is the value from your stored data
