import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
#MODEL_ID = "你的文件ID"  # 替换为你的 Google Drive 文件ID
MODEL_URL = f"https://github.com/Gongheifatchoi/Warehouse_Defect_MVP_UAT/blob/main/best.pt"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        # Remove existing file to allow overwrite
        if os.path.exists(path):
            os.remove(path)
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

# Ensure model is downloaded
model_file = download_model()

# Load YOLO model
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(image)

    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)
