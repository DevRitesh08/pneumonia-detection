import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")

# --- CUSTOM CSS ---
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

# --- MODEL DEFINITIONS ---
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_models():
    device = torch.device("cpu") # Force CPU for deployment
    
    # 1. ResNet18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 2))
    try:
        resnet.load_state_dict(torch.load("showcase/resnet18_transfer_best.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        try:
            resnet.load_state_dict(torch.load("pneumonia_model.pth", map_location=device, weights_only=True))
        except FileNotFoundError:
            st.warning("ResNet18 weights not found.")
    resnet.eval()
    
    # 2. Custom CNN
    cnn = PneumoniaCNN(num_classes=2)
    try:
        cnn.load_state_dict(torch.load("showcase/custom_cnn_best.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        st.warning("Custom CNN weights not found.")
    cnn.eval()
    
    return {"ResNet18 (Transfer Learning)": resnet, "Custom CNN (From Scratch)": cnn}, device

models_dict, device = load_models()

# --- GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx):
        self.model.eval()
        output = self.model(x)
        self.model.zero_grad()
        
        target = output[0][class_idx]
        target.backward()
        
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        # Resize using Pillow
        cam_img = Image.fromarray(cam)
        cam_img = cam_img.resize((x.shape[3], x.shape[2]), Image.Resampling.BILINEAR)
        cam = np.array(cam_img)
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    import matplotlib as mpl
    try:
        color_map = mpl.colormaps[colormap_name]
    except AttributeError:
        import matplotlib.cm as cm
        color_map = cm.get_cmap(colormap_name)
    heatmap = color_map(activation)
    heatmap = heatmap[:, :, :3] # discard alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    org_im = np.array(org_im)
    org_im = (org_im - np.min(org_im)) / (np.max(org_im) - np.min(org_im))
    org_im = (org_im * 255).astype(np.uint8)
    
    superimposed_img = heatmap * 0.4 + org_im * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model_name = st.selectbox("Select Model Architecture", list(models_dict.keys()))
    selected_model = models_dict[selected_model_name]

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# --- Upload & Predict ---
st.subheader("Analyze Chest X-Ray")

input_method = st.radio(
    "Choose Input Method",
    ["Upload your own image", "Use Sample NORMAL Image", "Use Sample PNEUMONIA Image"],
    horizontal=True,
    label_visibility="collapsed"
)

image = None

if input_method == "Upload your own image":
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif input_method == "Use Sample NORMAL Image":
    sample_num = st.selectbox("Select Normal Sample", ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"])
    paths = {
        "Sample 1": "showcase/sample_normal.jpeg",
        "Sample 2": "showcase/sample_normal_2.jpeg",
        "Sample 3": "showcase/sample_normal_3.jpeg",
        "Sample 4": "showcase/sample_normal_4.jpeg",
        "Sample 5": "showcase/sample_normal_5.jpeg"
    }
    try:
        image = Image.open(paths[sample_num]).convert("RGB")
    except FileNotFoundError:
        st.error(f"{sample_num} file not found.")
elif input_method == "Use Sample PNEUMONIA Image":
    sample_num = st.selectbox("Select Pneumonia Sample", ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"])
    paths = {
        "Sample 1": "showcase/sample_pneumonia.jpeg",
        "Sample 2": "showcase/sample_pneumonia_2.jpeg",
        "Sample 3": "showcase/sample_pneumonia_3.jpeg",
        "Sample 4": "showcase/sample_pneumonia_4.jpeg",
        "Sample 5": "showcase/sample_pneumonia_5.jpeg"
    }
    try:
        image = Image.open(paths[sample_num]).convert("RGB")
    except FileNotFoundError:
        st.error(f"{sample_num} file not found.")

if image is not None:
    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Original X-Ray", use_container_width=True)
        
    # Run inference
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True # For Grad-CAM
    
    # Get target layer for GradCAM
    if "ResNet18" in selected_model_name:
        target_layer = selected_model.layer4[-1]
    else:
        target_layer = selected_model.features[12] # The last Conv2d layer
        
    grad_cam = GradCAM(selected_model, target_layer)
    
    # Forward pass (done manually without torch.no_grad for CAM)
    selected_model.eval()
    output = selected_model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)[0]
    confidence = probs.detach().numpy()
    pred_idx = np.argmax(confidence)
    pred_name = CLASS_NAMES[pred_idx]
    
    # Generate Heatmap
    cam = grad_cam(input_tensor, pred_idx)
    # Resize original image to match tensor size for heatmap overlay
    heatmap_img = apply_colormap_on_image(image.resize((224, 224)), cam)

    with col_img:
        st.image(heatmap_img, caption="Grad-CAM Heatmap (Explainability)", use_container_width=True)

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
    
    # --- GEMINI API REPORT ---
    st.subheader("🤖 Generative AI Radiology Report")
    st.markdown("Use Google Gemini to generate a professional mock clinical summary based on the model's prediction.")
    
    if st.button("Generate AI Medical Report", type="primary"):
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            gemini_api_key = None
            
        if not gemini_api_key:
            st.warning("⚠️ API key missing! Please add `GEMINI_API_KEY` to your Streamlit Community Cloud Secrets.")
        else:
            with st.spinner("Analyzing image and generating professional report..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    You are an AI radiology assistant. The deep learning model has analyzed a chest X-Ray and provided the following result:
                    Diagnosis: {pred_name}
                    Confidence: {confidence[pred_idx]*100:.1f}%
                    
                    Please generate a short, professional mock medical report based on this finding. Include a disclaimer that this is AI-generated and requires clinical correlation. Keep it concise (3-4 paragraphs).
                    """
                    response = model.generate_content(prompt)
                    st.success("Report Generated Successfully!")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It should not be used for medical diagnosis. Consult a healthcare professional."
)
