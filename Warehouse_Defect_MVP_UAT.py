import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

# ----------------------------
# Model setup
# ----------------------------
MODEL_PATH = "best.pt"
#MODEL_ID = "你的文件ID"  # 替换为你的 Google Drive 文件ID
MODEL_URL = f"https://github.com/Gongheifatchoi/Warehouse_Defect_MVP_UAT/blob/main/best.pt"

# ----------------------------
# Download model if not exists
# ----------------------------
@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

model_file = download_model()

# ----------------------------
# Load YOLO model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(model_file)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model(image)

    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)

    # Output detected classes and confidences
    if hasattr(results[0], 'probs'):
        probs = results[0].probs.tolist()
        st.write("Class Probabilities:", probs)
    elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        preds = [(results[0].names[int(box.cls[0])], float(box.conf[0])) for box in results[0].boxes]
        st.write("Detected Defects:", preds)
    else:
        st.write("No defects detected.")

# ----------------------------
# Commentary Section (AI)
# ----------------------------
st.subheader("AI Commentary")
st.write(
    "Based on the image and detected defects, the AI will provide commentary here."
)
st.write(
    "✅ You can integrate OpenAI API or other free NLP tools to auto-generate commentary."
)
