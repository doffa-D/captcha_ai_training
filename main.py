import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import binascii


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


# Define allowed characters (same as in your training code)
def generate_char_preset():
    upper_case = [chr(i) for i in range(65, 91)]  # A-Z
    lower_case = [chr(i) for i in range(97, 123)]  # a-z
    digits = [chr(i) for i in range(48, 58)]  # 0-9
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|', ';', ':', '\'', '"', ',', '<', '.', '>', '/', '?', '`', '~']

    # Combine all character sets
    char_preset = ''.join(upper_case + lower_case + digits + special_chars)
    return char_preset


allowed_characters = generate_char_preset()

# Create the vocabulary
letters = sorted(list(set(allowed_characters)))
vocabulary = ["-"] + letters  # "-" represents the blank label in CTC
num_chars = len(vocabulary)

# Create index mappings
idx2char = {k: v for k, v in enumerate(vocabulary)}
char2idx = {v: k for k, v in idx2char.items()}


# Load the pre-trained model
def load_model(model_path, num_chars):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_chars).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Transform the image
def preprocess_image(image_path):
    transform_ops = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform_ops(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Decode model output (greedy decoding)
def decode_predictions(logits, idx2char):
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


# Function to test an image
def predict_text_from_image(image_path, model_path, idx2char, num_chars):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path, num_chars)

    # Preprocess image
    image = preprocess_image(image_path)
    image = image.to(device)

    # Run the image through the model
    with torch.no_grad():
        logits = model(image)

    # Decode the prediction
    decoded_text = decode_predictions(logits, idx2char)
    return decoded_text[0]


# Example usage
if __name__ == "__main__":
    image_path = './captcha/images/1a92a23bcaa4b36ed8ae9fe373b340da4928bf8236ae4e6983fbd67002dfe28f.png'  # Replace with your image path
    model_path = 'best_model.pth'          # Path to your trained model
    decoded_text = predict_text_from_image(image_path, model_path, idx2char, num_chars)
    print(f"Predicted text: {decoded_text}")
