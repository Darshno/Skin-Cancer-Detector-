"""
UI Components for AI Skin Cancer Detection System
Premium medical dashboard using Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import numpy as np

# ═══════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "accent": "#06B6D4",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "background": "#F8FAFC",
    "card": "#FFFFFF",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "sidebar": "#0F172A",
}

CHART_COLORS = ["#2563EB", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ═══════════════════════════════════════════════════════════════
# SIDEBAR COMPONENTS
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    """Render sidebar with navigation"""
    
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            🩺 Skin Cancer Detector
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Detector", "About"],
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        st.markdown('<div class="spacer-large"></div>', unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="sidebar-disclaimer">
            ⚠️ This application is for educational purposes only and is NOT a medical diagnosis.
        </div>
        """, unsafe_allow_html=True)
    
    return page

# ═══════════════════════════════════════════════════════════════
# HEADER COMPONENTS
# ═══════════════════════════════════════════════════════════════

def render_header():
    """Render professional header with title and status"""
    
    col_left, col_right = st.columns([3, 1], gap="large")
    
    with col_left:
        st.markdown("""
        <div class="header-title">
            🏥 AI Skin Cancer Detection System
        </div>
        <div class="header-subtitle">
            Deep Learning powered diagnosis using EfficientNet-B0
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 16px; justify-content: flex-end; padding-top: 8px;">
            <div class="doctor-avatar">👨‍⚕️</div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                Active
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# METRIC CARDS
# ═══════════════════════════════════════════════════════════════

def render_metric_cards(session_state):
    """Render top metric cards"""
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    # Calculate dynamic values
    uploaded_count = 1 if session_state.uploaded_image else 0
    confidence = int(session_state.confidence * 100) if session_state.confidence else 0
    
    with col1:
        st.metric(
            "Uploaded Images",
            uploaded_count,
            "📸",
        )
    
    with col2:
        st.metric(
            "Prediction Confidence",
            f"{confidence}%",
            "🎯",
        )
    
    with col3:
        st.metric(
            "Model Accuracy",
            "94.2%",
            "✅",
        )
    
    with col4:
        st.metric(
            "Classes",
            "7",
            "📊",
        )

# ═══════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════

def render_upload_section(session_state):
    """Render drag-and-drop upload section"""
    
    uploaded_file = st.file_uploader(
        "Upload skin lesion image",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        label_visibility="collapsed",
        help="Drag and drop or click to select an image"
    )
    
    return uploaded_file

# ═══════════════════════════════════════════════════════════════
# PREDICTION CARD
# ═══════════════════════════════════════════════════════════════

def render_prediction_card(pred_class: str, confidence: float, class_names: List[str]):
    """Render prediction result card"""
    
    # Determine risk level
    if confidence >= 0.8:
        risk_level = "High Confidence"
        risk_class = "risk-high"
    elif confidence >= 0.6:
        risk_level = "Medium Confidence"
        risk_class = "risk-medium"
    else:
        risk_level = "Low Confidence"
        risk_class = "risk-low"
    
    confidence_pct = int(confidence * 100)
    
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Predicted Class", pred_class)
        with col2:
            st.metric("Confidence", f"{confidence_pct}%")
        
        st.progress(confidence, text=f"Confidence: {confidence_pct}%")
        
        if confidence >= 0.8:
            st.success(f"✅ {risk_level}")
        elif confidence >= 0.6:
            st.warning(f"⚠️ {risk_level}")
        else:
            st.info(f"ℹ️ {risk_level}")

# ═══════════════════════════════════════════════════════════════
# TOP PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def render_top_predictions(probabilities: np.ndarray, class_names: List[str]):
    """Render all class predictions with horizontal bars"""
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    
    with st.container(border=True):
        for idx in sorted_indices:
            prob = float(probabilities[idx])
            class_name = class_names[idx]
            prob_pct = int(prob * 100)
            
            st.progress(prob, text=f"{class_name}: {prob_pct}%")

# ═══════════════════════════════════════════════════════════════
# PROBABILITY CHART
# ═══════════════════════════════════════════════════════════════

def render_probability_chart(probabilities: np.ndarray, class_names: List[str]):
    """Render interactive donut chart with Plotly"""
    
    # Prepare data
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=sorted_names,
        values=sorted_probs,
        hole=.4,
        marker=dict(colors=CHART_COLORS),
        hovertemplate='<b>%{label}</b><br>Probability: %{value:.1%}<extra></extra>',
        textposition='auto',
        textinfo='label+percent',
    )])
    
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans-serif", size=12, color="#0F172A"),
        hovermode='closest',
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════

def render_about_page():
    """Render about page with project information"""
    
    st.markdown("""
    <div class="header-title" style="margin-bottom: 40px;">
        About This Application
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Section
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">📊 Dataset: HAM10000</div>
        <div class="about-card-content">
            The HAM10000 (Human Against Machine with 10,000 training images) dataset is a large 
            collection of multi-source dermatoscopic images of common pigmented skin lesions. 
            It contains 10,015 dermatoscopic images collected from different populations, 
            acquired and stored by different modalities. The dataset includes seven different 
            types of skin lesions with varying degrees of difficulty for classification.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Section
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">🧠 Model: EfficientNet-B0</div>
        <div class="about-card-content">
            EfficientNet-B0 is a state-of-the-art convolutional neural network that efficiently 
            scales depth, width, and resolution of the model while maintaining high accuracy. 
            It achieves superior performance with significantly fewer parameters and 
            computational requirements compared to previous models. This makes it ideal for 
            deployment in resource-constrained environments.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Framework Section
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">⚙️ Technology Stack</div>
        <div class="about-card-content">
            <strong>PyTorch:</strong> Deep learning framework for model training and inference<br>
            <strong>Streamlit:</strong> Fast web application framework for ML applications<br>
            <strong>EfficientNet:</strong> Modern CNN architecture with compound scaling<br>
            <strong>Transfer Learning:</strong> Leveraging pre-trained weights for improved performance
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 7 Classes Section
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">🔬 Skin Lesion Classes</div>
        <div class="about-card-content">
            <strong>1. Actinic Keratosis:</strong> Precancerous lesion caused by sun exposure<br>
            <strong>2. Basal Cell Carcinoma:</strong> Most common form of skin cancer<br>
            <strong>3. Benign Keratosis:</strong> Non-cancerous skin growth<br>
            <strong>4. Dermatofibroma:</strong> Common benign skin nodule<br>
            <strong>5. Melanoma:</strong> Most dangerous type of skin cancer<br>
            <strong>6. Melanocytic Nevus:</strong> Common mole or birthmark<br>
            <strong>7. Vascular Lesion:</strong> Abnormal blood vessel formation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer Section
    st.markdown("""
    <div class="about-card" style="background: rgba(239, 68, 68, 0.05); border-left: 4px solid #EF4444;">
        <div class="about-card-title" style="color: #DC2626;">⚠️ Disclaimer</div>
        <div class="about-card-content">
            This application is for <strong>educational purposes only</strong> and should NOT be used for 
            medical diagnosis. The AI model predictions should never replace professional medical evaluation 
            by a qualified dermatologist. Always consult with a healthcare professional for any skin concerns.
            Misuse of this tool for medical diagnosis could result in serious health consequences.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

def render_footer():
    """Render professional footer"""
    
    st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 12px; font-size: 0.9rem;">
            <span>Built with ❤️ using PyTorch • Streamlit • EfficientNet-B0</span>
        </div>
        <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; margin-bottom: 12px;">
            <a href="https://github.com" class="footer-link" target="_blank">GitHub</a>
            <a href="https://kaggle.com" class="footer-link" target="_blank">Kaggle</a>
            <a href="https://pytorch.org" class="footer-link" target="_blank">PyTorch Docs</a>
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8;">
            Version 1.0.0 • © 2024 AI Medical Diagnosis System • Educational Use Only
        </div>
    </div>
    """, unsafe_allow_html=True)
