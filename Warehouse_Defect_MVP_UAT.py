import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import gdown

# ----------------------------
# Model setup
# ----------------------------
MODEL_PATH = "best.pt"
#MODEL_ID = "你的文件ID"  # 替换为你的 Google Drive 文件ID
MODEL_URL = f"https://github.com/Gongheifatchoi/Warehouse_Defect_MVP_UAT/blob/main/best.pt"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

model_file = download_model()

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)

# ----------------------------
# Commentary model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_commentary_model():
    return pipeline("text-generation", model="google/flan-t5-small")

commentary_model = load_commentary_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)

    # Extract defect information
    defect_info = []
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            defect_info.append(f"{cls_name} ({conf:.1%} confidence)")
        defect_summary = f"{len(defect_info)} defect(s) detected: " + ", ".join(defect_info)
        prompt = f"Analyze the following defects in the concrete image and provide detailed commentary:\n{defect_summary}"
    else:
        prompt = "Analyze the uploaded concrete image and provide commentary. No defects detected."

    # Generate commentary
    st.subheader("AI Commentary")
    comment = commentary_model(prompt, max_length=200)[0]['generated_text']
    st.write(comment)
