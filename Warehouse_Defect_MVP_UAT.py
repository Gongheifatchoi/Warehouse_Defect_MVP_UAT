import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

from transformers import pipeline

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://github.com/Gongheifatchoi/Warehouse_Defect_MVP_UAT/blob/main/best.pt"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        if os.path.exists(path):
            os.remove(path)
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

model_file = download_model()

# Load YOLO model
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)

# ----------------------------
# 2. Free LLM for commentary
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-small")

llm = load_llm()

def generate_commentary_llm(detections):
    if hasattr(detections, "boxes") and len(detections.boxes) > 0:
        summary = []
        for box in detections.boxes:
            cls = detections.names[int(box.cls[0])]
            conf = float(box.conf[0])
            summary.append(f"Detected {cls} with confidence {conf*100:.1f}%")
        prompt = "Provide a short inspection comment for these defect detections:\n" + "\n".join(summary)
    else:
        prompt = "The concrete surface has no visible defects. Provide a short inspection comment."
    
    output = llm(prompt, max_length=150)[0]['generated_text']
    return output

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)

    # LLM Commentary
    st.subheader("AI Commentary")
    comment = generate_commentary_llm(results[0])
    st.write(comment)
