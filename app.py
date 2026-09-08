import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")

# --- CUSTOM CSS (same dark theme as fraud project) ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: #f8fafc;
}
.big-title {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
    font-weight: bold;
    padding-bottom: 10px;
}
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    padding-bottom: 30px;
}
div[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; }
div[data-testid="metric-container"] div { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Pneumonia Detection Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload • Analyze • Diagnose</div>', unsafe_allow_html=True)
st.markdown("---")


@st.cache_resource
def load_model():
    """Load the trained ResNet18 model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        st.warning("No trained model found. Run `python main.py` first.")
        return None, device

    return model.to(device), device


model, device = load_model()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# --- KPI Section ---
st.subheader("System Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Model", "ResNet18")
col2.metric("Transfer Learning", "ImageNet")
col3.metric("Classes", "Normal / Pneumonia")
st.markdown("---")

# --- Upload & Predict ---
st.subheader("Upload Chest X-Ray")
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # Run inference
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence = probs.numpy()
        pred_idx = np.argmax(confidence)
        pred_name = CLASS_NAMES[pred_idx]

    with col_result:
        st.subheader("Prediction")

        if pred_name == "PNEUMONIA":
            st.error(f"**{pred_name}** (Confidence: {confidence[pred_idx]*100:.1f}%)")
        else:
            st.success(f"**{pred_name}** (Confidence: {confidence[pred_idx]*100:.1f}%)")

        st.write("")
        st.write("**Class Probabilities:**")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name}: {confidence[i]*100:.1f}%")
            st.progress(float(confidence[i]))

st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It should not be used for medical diagnosis. Consult a healthcare professional."
)
