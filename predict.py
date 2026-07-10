import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights, DenseNet121_Weights
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


def load_models(effnet_path='best_model.pth', densenet_path='best_model_densenet.pth'):
    model1 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model1.classifier[1] = nn.Linear(model1.classifier[1].in_features, 7)
    model1.load_state_dict(torch.load(effnet_path, map_location=device))
    model1 = model1.to(device).eval()

    model2 = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    model2.classifier = nn.Linear(model2.classifier.in_features, 7)
    model2.load_state_dict(torch.load(densenet_path, map_location=device))
    model2 = model2.to(device).eval()

    return model1, model2


def predict(image: Image.Image, model1, model2):
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = torch.softmax(model1(input_tensor), dim=1)
        out2 = torch.softmax(model2(input_tensor), dim=1)
        avg_out = (out1 + out2) / 2

    probs = avg_out.cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]

    return {
        'pred_class': pred_class,
        'pred_full_name': CLASS_FULL_NAMES[pred_class],
        'confidence': confidence,
        'top3': top3
    }


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap


def generate_gradcam_overlay(image: Image.Image, model, target_layer=None):
    if target_layer is None:
        target_layer = model.features[-1]

    input_tensor = val_transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)

    img_resized = np.array(image.resize((224, 224)))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = heatmap_color * 0.4 + img_bgr
    superimposed = np.uint8(superimposed)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    return superimposed_rgb