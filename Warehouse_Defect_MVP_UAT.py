import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import io

# ----------------------------
# Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_ID = "你的文件ID"  # Replace with your Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

model_file = download_model()

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(model_file)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload one or more images of concrete surfaces to detect defects and get AI commentary.")

uploaded_files = st.file_uploader(
    "Choose image(s)...", type=["jpg","jpeg","png"], accept_multiple_files=True
)

def generate_commentary(results):
    """
    Simple commentary based on detection/classification results.
    """
    if model.task == "classify":
        if hasattr(results[0], 'probs'):
            class_id = int(results[0].probs.argmax())
            class_name = results[0].names[class_id]
            confidence = float(results[0].probs[class_id]) * 100
            comment = f"The image is classified as '{class_name}' with {confidence:.2f}% confidence."
            if class_name.lower() == "defect":
                comment += " Attention: Potential concrete defects detected. Consider inspection."
            else:
                comment += " No major defects detected."
        else:
            comment = "Classification probabilities not available."
    elif model.task == "detect":
        boxes = results[0].boxes
        if hasattr(results[0], 'boxes') and len(boxes) > 0:
            comment = f"{len(boxes)} potential defect(s) detected. Please inspect areas highlighted in the annotated image."
        else:
            comment = "No defects detected. The concrete surface appears normal."
    else:
        comment = "No commentary available for this model task."
    return comment

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Run inference
        results = model(image)

        # Annotated image
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Prediction Result", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button(
            label="Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"annotated_{uploaded_file.name}",
            mime="image/png"
        )

        # AI commentary
        commentary = generate_commentary(results)
        st.subheader("AI Commentary")
        st.write(commentary)
