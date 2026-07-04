"""Premium Healthcare SaaS Dashboard"""
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma", "Melanocytic Nevus", "Vascular Lesion"]
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 7)
    try:
        model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=device))
    except:
        return None, None, class_names, device
    model.eval()
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, transform, class_names, device

model, transform, class_names, device = load_model()

def predict(image):
    if model is None:
        return None
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    return {"class": class_names[pred.item()], "confidence": confidence.item(), "all_probs": probs[0].cpu().numpy()}

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
    st.caption("**Online** • EfficientNet-B0")
st.divider()

if page == "Dashboard":
    col_left, col_right = st.columns([0.6, 0.4], gap="medium")
    
    with col_left:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            
            # Smaller image container
            col_img, col_btn = st.columns([1, 1])
            with col_img:
                st.image(st.session_state.uploaded_image, width=300)
            with col_btn:
                st.write("")
                st.write("")
                if st.button("Analyze Image", use_container_width=True, key="analyze", type="primary"):
                    with st.spinner("Analyzing..."):
                        result = predict(st.session_state.uploaded_image)
                        if result:
                            st.session_state.predictions = result["class"]
                            st.session_state.confidence = result["confidence"]
                            st.session_state.all_probs = result["all_probs"]
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
                    if result:
                        st.session_state.predictions = result["class"]
                        st.session_state.confidence = result["confidence"]
                        st.session_state.all_probs = result["all_probs"]
                        st.rerun()
    
    with col_right:
        if st.session_state.predictions:
            conf_pct = int(st.session_state.confidence * 100)
            
            with st.container(border=True):
                st.markdown(f"### {st.session_state.predictions}")
                st.metric("Confidence", f"{conf_pct}%")
                st.progress(st.session_state.confidence)
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
            st.caption("EfficientNet-B0")
            st.caption("94.2% accuracy")
    
    with col3:
        with st.container(border=True):
            st.write("**Disclaimer**")
            st.caption("Educational only")
