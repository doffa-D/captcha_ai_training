import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import binascii

# Define your secret key
SECRET_KEY = '1d4c4f32f4b3e6e09e5bcb5d8907b711ad6c8c4b28b58a55d7b3d34216b28f68'  # 64-character hex string
SECRET_KEY_BYTES = binascii.unhexlify(SECRET_KEY)  # Convert hex string to bytes
IV_LENGTH = 16  # For AES, this is always 16 bytes

def decrypt(encrypted_text):
    try:
        # Extract the IV (first 16 bytes, 32 hex characters) and the encrypted text
        iv_hex = encrypted_text[:32]  # First 32 characters (16 bytes in hex)
        encrypted_data_hex = encrypted_text[32:]  # Remaining part is the encrypted text

        # Convert the IV and encrypted text from hex to bytes
        iv = binascii.unhexlify(iv_hex)
        encrypted_data = binascii.unhexlify(encrypted_data_hex)

        # Create the cipher object using AES-256-CBC with the IV and secret key
        cipher = Cipher(algorithms.AES(SECRET_KEY_BYTES), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt the data
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

        # Remove padding (PKCS7)
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

        # Return the decrypted text as a UTF-8 string
        return decrypted_data.decode('utf-8')

    except ValueError as e:
        print(f"Decryption error: {e}")
        return None

def generate_char_preset():
    upper_case = [chr(i) for i in range(65, 91)]  # A-Z
    lower_case = [chr(i) for i in range(97, 123)] # a-z
    digits = [chr(i) for i in range(48, 58)]      # 0-9
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|', ';', ':', '\'', '"', ',', '<', '.', '>', '/', '?', '`', '~']

    # Combine all character sets
    char_preset = ''.join(upper_case + lower_case + digits + special_chars)
    return char_preset

# Define allowed characters
allowed_characters = generate_char_preset()

# Check the number of available CPU cores
cpu_count = mp.cpu_count()
print("Available CPU cores:", cpu_count)

# Data loading
data_path = "./captcha/images"  # Update to your image directory
all_image_fns = os.listdir(data_path)
print("Total samples:", len(all_image_fns))

# Decrypt all filenames to get the labels and filter out invalid samples
valid_image_fns = []
decrypted_texts = []
for fn in all_image_fns:
    encrypted_name = fn.split('.')[0]
    decrypted = decrypt(encrypted_name)
    if decrypted is not None and all(c in allowed_characters for c in decrypted):
        decrypted_texts.append(decrypted)
        valid_image_fns.append(fn)
    else:
        # Skip files that cannot be decrypted or contain invalid characters
        continue

print("Valid samples:", len(valid_image_fns))

# Create the vocabulary
letters = sorted(list(set(allowed_characters)))
vocabulary = ["-"] + letters  # "-" represents the blank label in CTC
num_chars = len(vocabulary)

# Create index mappings
idx2char = {k: v for k, v in enumerate(vocabulary)}
char2idx = {v: k for k, v in idx2char.items()}

# Create a list of (filename, decrypted_text) tuples
fn_text_tuples = list(zip(valid_image_fns, decrypted_texts))

# Split dataset into training and testing
train_data, test_data = train_test_split(fn_text_tuples, random_state=0)
image_fns_train, decrypted_texts_train = zip(*train_data)
image_fns_test, decrypted_texts_test = zip(*test_data)
print("Training samples:", len(image_fns_train), "Testing samples:", len(image_fns_test))

# Define the dataset class
class CAPTCHADataset(Dataset):
    def __init__(self, data_dir, image_fns, decrypted_texts):
        self.data_dir = data_dir
        self.image_fns = image_fns
        self.decrypted_texts = decrypted_texts

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.data_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)

        # Use the precomputed decrypted text
        text = self.decrypted_texts[index]

        return image, text, image_fn

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize((50, 200)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

# Initialize datasets and loaders
batch_size = 16
trainset = CAPTCHADataset(data_path, image_fns_train, decrypted_texts_train)
testset = CAPTCHADataset(data_path, image_fns_test, decrypted_texts_test)
train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=cpu_count, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, num_workers=cpu_count, shuffle=False)

# Define the CRNN model with bidirectional GRU
class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size

        # CNN Part 1
        resnet_modules = list(resnet18(weights='DEFAULT').children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)

        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(1024, 256)

        # Bidirectional GRU
        self.rnn1 = nn.GRU(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size,
                           bidirectional=True, batch_first=True)
        self.rnn2 = nn.GRU(input_size=rnn_hidden_size * 2, hidden_size=rnn_hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size * 2, num_chars)

    def forward(self, batch):
        batch = self.cnn_p1(batch)
        batch = self.cnn_p2(batch)
        batch = batch.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]

        batch_size = batch.size(0)
        T = batch.size(1)
        batch = batch.view(batch_size, T, -1)

        batch = self.linear1(batch)
        batch, _ = self.rnn1(batch)
        batch, _ = self.rnn2(batch)

        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)  # [T, batch_size, num_classes]

        return batch

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crnn = CRNN(num_chars).to(device)
# Note: We will not apply weights_init here since we are loading a pre-trained model

# Load the pre-trained model
pretrained_dict = torch.load('best_model.pth')
model_dict = crnn.state_dict()

# 1. Filter out the parameters of the final layer (since num_chars has changed)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'linear2' not in k}

# 2. Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)

# 3. Load the new state dict
crnn.load_state_dict(model_dict)
print("Loaded pre-trained model from 'best_model.pth' (excluding the final layer).")

# Optionally, re-initialize the final layer
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

crnn.linear2.apply(weights_init)

# Define loss function
criterion = nn.CTCLoss(blank=0)

# Encode text batch
def encode_text_batch(text_batch):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens).to(device)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets).to(device)

    return text_batch_targets, text_batch_targets_lens

# Compute loss
def compute_loss(text_batch, text_batch_logits):
    text_batch_logps = F.log_softmax(text_batch_logits, 2)  # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device)  # [batch_size]
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss

# Training parameters
num_epochs = 100  # Adjust as needed
lr = 0.001
weight_decay = 1e-3
clip_norm = 5
optimizer = optim.Adam(crnn.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Training loop
epoch_losses = []
iteration_losses = []

best_val_loss = float('inf')  # Initialize the best validation loss

for epoch in tqdm(range(1, num_epochs + 1)):
    crnn.train()
    epoch_loss_list = []
    for image_batch, text_batch, _ in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        image_batch = image_batch.to(device)
        text_batch_logits = crnn(image_batch)
        loss = compute_loss(text_batch, text_batch_logits)
        iteration_loss = loss.item()

        if np.isnan(iteration_loss) or np.isinf(iteration_loss):
            continue

        iteration_losses.append(iteration_loss)
        epoch_loss_list.append(iteration_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
        optimizer.step()

    epoch_loss = np.mean(epoch_loss_list)
    print(f"Epoch: {epoch}    Loss: {epoch_loss:.4f}")
    epoch_losses.append(epoch_loss)
    lr_scheduler.step(epoch_loss)

    # Save the best model based on validation loss
    if epoch % 5 == 0:  # Evaluate every 5 epochs
        crnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image_batch, text_batch, _ in test_loader:
                image_batch = image_batch.to(device)
                text_batch_logits = crnn(image_batch)
                loss = compute_loss(text_batch, text_batch_logits)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(crnn.state_dict(), 'best_model.pth')
            print(f"Best model saved with validation loss {val_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Epoch Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output_plot.png')  # Save the plot to a file

# Decode model output (greedy decoding)
def decode_predictions(logits):
    logits = logits.permute(1, 0, 2)  # [batch_size, T, num_classes]
    logits = torch.argmax(logits, dim=2)  # Take the class with the highest score
    decoded_predictions = []
    for logit_seq in logits:
        decoded_text = []
        prev_idx = None
        for idx in logit_seq:
            idx = int(idx)
            if idx != prev_idx and idx != 0:  # Remove consecutive duplicates and blank chars
                decoded_text.append(idx2char[idx])
            prev_idx = idx
        decoded_predictions.append("".join(decoded_text))
    return decoded_predictions

# Evaluate the model on the test set and save correct predictions
def evaluate_and_save_correct_predictions(model, test_loader, save_file='correct_predictions.txt'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    correct_predictions = []

    with torch.no_grad():
        for image_batch, text_batch, image_fns in tqdm(test_loader, leave=False):
            image_batch = image_batch.to(device)
            text_batch_logits = model(image_batch)
            decoded_predictions = decode_predictions(text_batch_logits)

            # Compare predictions with decrypted labels
            for pred, true_text in zip(decoded_predictions, text_batch):
                if pred == true_text:
                    correct_predictions.append((true_text, pred))
                    correct += 1
                total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save correct predictions to a file
    if correct_predictions:
        with open(save_file, "w") as f:
            for true_text, pred in correct_predictions:
                f.write(f"True Text: {true_text}, Prediction: {pred}\n")
        print(f"Correct predictions saved to {save_file}")
    else:
        print("No correct predictions found.")

    return accuracy

# Evaluate model after training
test_accuracy = evaluate_and_save_correct_predictions(crnn, test_loader)
