import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Real time serverless Image Recognition", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #38bdf8;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.card {
    padding: 15px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}
.result-box {
    padding: 10px;
    border-radius: 10px;
    background: #1e293b;
    color: white;
    margin-bottom: 10px;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Real Time Serverless Image Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by YOLO + ResNet Deep Learning Models</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_model.eval()
    return yolo_model, resnet_model

model_yolo, model_resnet = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚙️ Controls")
option = st.sidebar.radio("Choose Input Type", ["Upload Image", "Use Camera"])
conf_threshold = st.sidebar.slider("Confidence Level", 0.1, 1.0, 0.3)

image = None

# ---------------- INPUT ----------------
st.markdown("### 📤 Upload or Capture Image")

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Camera":
    camera = st.camera_input("Take Photo")
    if camera:
        image = Image.open(camera).convert("RGB")

# ---------------- PROCESS ----------------
if image:
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Input Image")
        st.image(image, use_column_width=True)

    img_array = np.array(image)

    # ---------------- YOLO ----------------
    with st.spinner("🔍 Detecting objects..."):
        results = model_yolo(img_array, conf=conf_threshold)

    # 🔥 IMPORTANT FIX (display correct)
    annotated_frame = results[0].plot()

    with col2:
        st.markdown("### 🎯 Detection Result")
        st.image(annotated_frame, use_column_width=True)

    st.markdown("## 🔍 Detected Objects")

    best_detections = {}

    for r in results:
        for box in r.boxes:
            label = model_yolo.names[int(box.cls)]
            confidence = float(box.conf)

            if label not in best_detections or confidence > best_detections[label]:
                best_detections[label] = confidence

    if best_detections:
        for label, conf in best_detections.items():
            st.markdown(f'<div class="result-box">👉 <b>{label}</b> — {conf*100:.2f}%</div>', unsafe_allow_html=True)
            st.progress(float(conf))  # 🔥 fix (streamlit issue avoid)
    else:
        st.warning("No objects detected")

    # ---------------- RESNET ----------------
    st.markdown("## 🧠 AI Detailed Prediction")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model_resnet(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3 = torch.topk(probabilities, 3)

    classes = models.ResNet18_Weights.DEFAULT.meta["categories"]

    for idx, prob in zip(top3.indices, top3.values):
        st.markdown(f'<div class="card">👉 <b>{classes[int(idx)]}</b><br>Confidence: {prob.item()*100:.2f}%</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.success("✅ Prediction Completed Successfully!")
