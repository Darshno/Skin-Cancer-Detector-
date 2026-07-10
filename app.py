"""Premium Healthcare SaaS Dashboard"""
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torchvision import transforms
import plotly.graph_objects as go

st.set_page_config(page_title="Skin Cancer AI", page_icon="🩺", layout="wide")
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "all_probs" not in st.session_state:
    st.session_state.all_probs = None
if "gradcam_img" not in st.session_state:
    st.session_state.gradcam_img = None

class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma", "Melanocytic Nevus", "Vascular Lesion"]
class_codes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


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


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = efficientnet_b0(weights=None)
    model1.classifier[1] = nn.Linear(1280, 7)
    model2 = densenet121(weights=None)
    model2.classifier = nn.Linear(model2.classifier.in_features, 7)

    try:
        model1.load_state_dict(torch.load("best_model.pth", map_location=device))
        model2.load_state_dict(torch.load("best_model_densenet.pth", map_location=device))
    except Exception as e:
        st.error(f"Could not load model weights: {e}")
        return None, None, device

    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    return model1, model2, device


model1, model2, device = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image):
    if model1 is None or model2 is None:
        return None
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = torch.softmax(model1(img_tensor), dim=1)
        out2 = torch.softmax(model2(img_tensor), dim=1)
        avg_probs = (out1 + out2) / 2
    confidence, pred = torch.max(avg_probs, dim=1)
    return {"class": class_names[pred.item()], "confidence": confidence.item(), "all_probs": avg_probs[0].cpu().numpy()}


def get_gradcam(image):
    if model1 is None:
        return None
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    target_layer = model1.features[-1]
    gradcam = GradCAM(model1, target_layer)
    heatmap = gradcam.generate(input_tensor)

    img_resized = np.array(image.resize((224, 224)))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = np.uint8(heatmap_color * 0.4 + img_bgr)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.title("SCD AI")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Detector", "About"], label_visibility="collapsed")
    st.divider()
    st.caption("Educational use only")

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Skin Cancer Detection AI")
with col_h2:
    st.caption("**Online** • EfficientNet-B0 + DenseNet121 Ensemble")
st.divider()

if page == "Dashboard":
    col_left, col_right = st.columns([0.6, 0.4], gap="medium")

    with col_left:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")

            col_img, col_btn = st.columns([1, 1])
            with col_img:
                st.image(st.session_state.uploaded_image, width=300)
            with col_btn:
                st.write("")
                st.write("")
                if st.button("Analyze Image", use_container_width=True, key="analyze", type="primary"):
                    with st.spinner("Analyzing..."):
                        result = predict(st.session_state.uploaded_image)
                        gradcam_img = get_gradcam(st.session_state.uploaded_image)
                        if result:
                            st.session_state.predictions = result["class"]
                            st.session_state.confidence = result["confidence"]
                            st.session_state.all_probs = result["all_probs"]
                            st.session_state.gradcam_img = gradcam_img
                            st.rerun()
        else:
            st.info("Upload image")

    with col_right:
        st.subheader("Diagnosis")

        if st.session_state.predictions:
            conf_pct = int(st.session_state.confidence * 100)

            with st.container(border=True):
                st.markdown(f"### {st.session_state.predictions}")
                st.progress(st.session_state.confidence)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{conf_pct}%")
                with col2:
                    if conf_pct >= 80:
                        st.metric("Risk", "High")
                    elif conf_pct >= 60:
                        st.metric("Risk", "Medium")
                    else:
                        st.metric("Risk", "Low")
        else:
            st.info("Results appear here")

    if st.session_state.all_probs is not None:
        st.divider()

        col_pred, col_conf, col_risk = st.columns(3)
        conf_pct = int(st.session_state.confidence * 100)

        with col_pred:
            st.metric("Prediction", st.session_state.predictions)
        with col_conf:
            st.metric("Confidence", f"{conf_pct}%")
        with col_risk:
            if conf_pct >= 80:
                st.metric("Risk", "High")
            elif conf_pct >= 60:
                st.metric("Risk", "Medium")
            else:
                st.metric("Risk", "Low")

        st.divider()

        col_bars, col_chart = st.columns([0.55, 0.45], gap="medium")

        with col_bars:
            st.subheader("Top 3")
            sorted_indices = np.argsort(st.session_state.all_probs)[::-1][:3]
            for idx in sorted_indices:
                prob = float(st.session_state.all_probs[idx])
                class_name = class_names[idx]
                prob_pct = int(prob * 100)
                st.write(f"**{class_name}**")
                st.progress(prob, text=f"{prob_pct}%")

        with col_chart:
            st.subheader("Distribution")
            sorted_indices = np.argsort(st.session_state.all_probs)[::-1]
            sorted_probs = st.session_state.all_probs[sorted_indices]
            sorted_names = [class_names[i] for i in sorted_indices]

            fig = go.Figure(data=[go.Pie(
                labels=sorted_names,
                values=sorted_probs,
                hole=0.4,
                marker=dict(colors=['#2563EB', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>%{value:.1%}<extra></extra>'
            )])
            fig.update_layout(
                showlegend=False,
                height=280,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.gradcam_img is not None:
            st.divider()
            st.subheader("Grad-CAM — Model Attention")
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.image(st.session_state.uploaded_image.resize((224, 224)), caption="Original", use_container_width=True)
            with col_cam:
                st.image(st.session_state.gradcam_img, caption="Model Attention (EfficientNet-B0)", use_container_width=True)

elif page == "Detector":
    st.subheader("Detector")

    col_left, col_right = st.columns([0.6, 0.4], gap="medium")

    with col_left:
        uploaded_file = st.file_uploader("Select image", type=["jpg", "jpeg", "png"], key="det_upload")

        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            with st.container(border=True):
                st.image(st.session_state.uploaded_image, use_container_width=True)

            if st.button("Analyze", use_container_width=True, key="analyze_det", type="primary"):
                with st.spinner("Processing..."):
                    result = predict(st.session_state.uploaded_image)
                    gradcam_img = get_gradcam(st.session_state.uploaded_image)
                    if result:
                        st.session_state.predictions = result["class"]
                        st.session_state.confidence = result["confidence"]
                        st.session_state.all_probs = result["all_probs"]
                        st.session_state.gradcam_img = gradcam_img
                        st.rerun()

    with col_right:
        if st.session_state.predictions:
            conf_pct = int(st.session_state.confidence * 100)

            with st.container(border=True):
                st.markdown(f"### {st.session_state.predictions}")
                st.metric("Confidence", f"{conf_pct}%")
                st.progress(st.session_state.confidence)

            if st.session_state.gradcam_img is not None:
                st.image(st.session_state.gradcam_img, caption="Grad-CAM", use_container_width=True)
        else:
            st.info("Upload to analyze")

else:
    st.subheader("About")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.write("**Dataset**")
            st.caption("HAM10000")
            st.caption("10,000+ images")

    with col2:
        with st.container(border=True):
            st.write("**Model**")
            st.caption("EfficientNet-B0 + DenseNet121 Ensemble")
            st.caption("83% accuracy · 0.84 weighted F1")

    with col3:
        with st.container(border=True):
            st.write("**Disclaimer**")
            st.caption("Educational only")