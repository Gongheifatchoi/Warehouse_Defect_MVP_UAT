import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

# ----------------------------
# 模型路径与 Google Drive 链接
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_ID = "你的文件ID"  # 替换为你的 Google Drive 文件ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# ----------------------------
# 下载模型
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
# 加载模型
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(model_file)

# ----------------------------
# Streamlit 用户界面
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 推理
    results = model(image)

    # 标注图片
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Prediction Result", use_column_width=True)

    # 输出预测信息
    if model.task == "classify":
        # 分类模型
        if hasattr(results[0], 'probs'):
            probs = results[0].probs.tolist()
            class_name = results[0].names[int(results[0].probs.argmax())]
            st.write(f"Predicted Class: {class_name}")
            st.write("Class Probabilities:", probs)
        else:
            st.write("No classification probabilities available.")
    elif model.task == "detect":
        # 检测模型
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            st.write("Detected objects:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"- Class: {results[0].names[cls_id]}, Confidence: {conf:.2f}")
        else:
            st.write("No objects detected.")
