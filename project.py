import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np

st.title("Real-Time Image Recognition System")

# ---------------- LOAD MODELS ONCE ----------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # lighter model
    resnet_model.eval()
    return yolo_model, resnet_model

model_yolo, model_resnet = load_models()

# ---------------- INPUT ----------------
option = st.radio("Choose Input Type", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Camera":
    camera = st.camera_input("Take a picture")
    if camera:
        image = Image.open(camera).convert("RGB")

# ---------------- PROCESS ----------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    img_array = np.array(image)

    # ---------------- YOLO ----------------
    st.subheader("🔍 Detected Objects")

    results = model_yolo(img_array, conf=0.3)

    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="Detected Image", use_column_width=True)

    best_detections = {}

    for r in results:
        for box in r.boxes:
            label = model_yolo.names[int(box.cls)]
            confidence = float(box.conf)

            if label not in best_detections or confidence > best_detections[label]:
                best_detections[label] = confidence

    if best_detections:
        for label, conf in best_detections.items():
            st.write(f"👉 {label} ({conf*100:.2f}%)")
            st.progress(float(conf))
    else:
        st.warning("No objects detected")

    # ---------------- RESNET ----------------
    st.subheader("🧠 Detailed Prediction")

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

    # Predefined labels (no internet needed)
    classes = models.ResNet18_Weights.DEFAULT.meta["categories"]

    for idx, prob in zip(top3.indices, top3.values):
        st.write(f"👉 {classes[int(idx)]} ({prob.item()*100:.2f}%)")
        