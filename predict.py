import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (same order as LabelEncoder during training)
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion"
]

# Load model
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 7)

model.load_state_dict(
    torch.load("skin_cancer_model.pth", map_location=device)
)

model.to(device)
model.eval()

# Image preprocessing
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Get image path
image_path = input("Enter image path: ")

# Load image
image = Image.open(image_path).convert("RGB")

# Transform image
image = transform(image)

# Add batch dimension
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)

print(probabilities)

top_probs, top_classes = torch.topk(probabilities, k=3)

print("\nTop Predictions:\n")

for prob, cls in zip(top_probs[0], top_classes[0]):
    print(f"{class_names[cls.item()]} : {prob.item()*100:.2f}%")

