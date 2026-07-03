"""
AI Skin Cancer Detection System
Professional medical dashboard using Streamlit & PyTorch
"""

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from components import (
    render_sidebar,
    render_header,
    render_metric_cards,
    render_upload_section,
    render_prediction_card,
    render_top_predictions,
    render_probability_chart,
    render_about_page,
    render_footer
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Skin Cancer Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# LOAD CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "pred_class" not in st.session_state:
    st.session_state.pred_class = None
if "all_probs" not in st.session_state:
    st.session_state.all_probs = None

# ═══════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained EfficientNet-B0 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = [
        "Actinic Keratosis",
        "Basal Cell Carcinoma",
        "Benign Keratosis",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic Nevus",
        "Vascular Lesion"
    ]
    
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 7)
    
    model.load_state_dict(
        torch.load(
            "skin_cancer_model.pth",
            map_location=device
        )
    )
    
    model.eval()
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    
    return model, transform, class_names, device

# Load model
model, transform, class_names, device = load_model()

# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def predict(image):
    """Run inference on uploaded image"""
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    
    confidence, pred = torch.max(probs, dim=1)
    
    return {
        "class": class_names[pred.item()],
        "confidence": confidence.item(),
        "all_probs": probs[0].cpu().numpy()
    }

# ═══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════

page = render_sidebar()

# ═══════════════════════════════════════════════════════════════
# MAIN PAGE CONTENT
# ═══════════════════════════════════════════════════════════════

if page == "Dashboard":
    
    # Header
    render_header()
    
    # Metric Cards
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_metric_cards(st.session_state)
    
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    # Main Layout - Two Column
    col_left, col_right = st.columns([1.2, 1], gap="large")
    
    with col_left:
        st.markdown('<h3 class="section-title">📸 Image Analysis</h3>', unsafe_allow_html=True)
        uploaded_file = render_upload_section(st.session_state)
        
        # Process uploaded image
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            
            # Show image preview
            col_img1, col_img2 = st.columns([1, 1])
            with col_img1:
                st.image(
                    st.session_state.uploaded_image,
                    use_container_width=True,
                    caption="Uploaded Lesion Image"
                )
            
            # Predict button
            with col_img2:
                st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
                
                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True,
                    key="predict_btn",
                    help="Click to run AI diagnosis"
                ):
                    with st.spinner("🔄 Analyzing image with AI..."):
                        result = predict(st.session_state.uploaded_image)
                        st.session_state.predictions = result["class"]
                        st.session_state.confidence = result["confidence"]
                        st.session_state.all_probs = result["all_probs"]
                        st.session_state.pred_class = result["class"]
                
                st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<h3 class="section-title">🎯 Diagnosis Result</h3>', unsafe_allow_html=True)
        
        if st.session_state.predictions is not None:
            render_prediction_card(
                st.session_state.predictions,
                st.session_state.confidence,
                class_names
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-text">
                    Upload an image and click "Analyze Image" to see diagnosis results
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results Section
    if st.session_state.all_probs is not None:
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<h3 class="section-title">📊 All Predictions</h3>', unsafe_allow_html=True)
            render_top_predictions(st.session_state.all_probs, class_names)
        
        with col2:
            st.markdown('<h3 class="section-title">📈 Probability Distribution</h3>', unsafe_allow_html=True)
            render_probability_chart(st.session_state.all_probs, class_names)
    
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_footer()

elif page == "Detector":
    
    render_header()
    
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    # Main Layout - Two Column
    col_left, col_right = st.columns([1.2, 1], gap="large")
    
    with col_left:
        st.markdown('<h3 class="section-title">📸 Image Analysis</h3>', unsafe_allow_html=True)
        uploaded_file = render_upload_section(st.session_state)
        
        # Process uploaded image
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
            
            # Show image preview
            col_img1, col_img2 = st.columns([1, 1])
            with col_img1:
                st.image(
                    st.session_state.uploaded_image,
                    use_container_width=True,
                    caption="Uploaded Lesion Image"
                )
            
            # Predict button
            with col_img2:
                st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
                
                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True,
                    key="predict_btn_detector",
                    help="Click to run AI diagnosis"
                ):
                    with st.spinner("🔄 Analyzing image with AI..."):
                        result = predict(st.session_state.uploaded_image)
                        st.session_state.predictions = result["class"]
                        st.session_state.confidence = result["confidence"]
                        st.session_state.all_probs = result["all_probs"]
                        st.session_state.pred_class = result["class"]
                
                st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<h3 class="section-title">🎯 Diagnosis Result</h3>', unsafe_allow_html=True)
        
        if st.session_state.predictions is not None:
            render_prediction_card(
                st.session_state.predictions,
                st.session_state.confidence,
                class_names
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-text">
                    Upload an image and click "Analyze Image" to see diagnosis results
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results Section
    if st.session_state.all_probs is not None:
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<h3 class="section-title">📊 All Predictions</h3>', unsafe_allow_html=True)
            render_top_predictions(st.session_state.all_probs, class_names)
        
        with col2:
            st.markdown('<h3 class="section-title">📈 Probability Distribution</h3>', unsafe_allow_html=True)
            render_probability_chart(st.session_state.all_probs, class_names)
    
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_footer()

else:  # About page
    st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
    render_about_page()
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_footer()