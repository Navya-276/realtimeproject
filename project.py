import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import urllib.request
import numpy as np

st.title("Real-Time Image Recognition System")

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

    model_yolo = YOLO("yolov8n.pt")

    # 🔥 Lower confidence to detect more animals
    results = model_yolo(img_array, conf=0.3)

    # Show bounding boxes
    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="Detected Image", use_column_width=True)

    best_detections = {}

    for r in results:
        for box in r.boxes:
            label = model_yolo.names[int(box.cls)]
            confidence = float(box.conf)

            # Keep highest confidence per label
            if label not in best_detections or confidence > best_detections[label]:
                best_detections[label] = confidence

    # 🔥 KEEP ALL OBJECTS (no restriction)
    filtered = {}
    for label, conf in best_detections.items():
        if conf > 0.4:   # adjust if needed (0.3–0.5)
            filtered[label] = conf

    # Display results
    if filtered:
        for label, conf in filtered.items():
            st.write(f"👉 {label} ({conf*100:.2f}%)")
            st.progress(float(conf))
    else:
        st.warning("No objects detected clearly")

    # ---------------- RESNET ----------------
    st.subheader("🧠 Detailed Prediction (Top 3)")

    model_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model_resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model_resnet(img_tensor)
        _, indices = torch.topk(outputs, 3)

    # Load labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes = urllib.request.urlopen(url).read().decode('utf-8').split("\n")

    for idx in indices[0]:
        st.write(f"👉 {classes[int(idx)]}")