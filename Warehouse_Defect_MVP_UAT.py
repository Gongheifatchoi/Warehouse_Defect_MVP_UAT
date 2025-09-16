import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json
import time
from openai import OpenAI
from datetime import datetime

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FHjz3wjLBWk5c04j7kGBynQcPTyVA19R"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
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
# 2. LLM Analysis Functions (Concise)
# ----------------------------
def get_llm_concise_analysis(defect_type, confidence, area_name):
    """
    Use LLM to generate concise 1-2 line analysis of the defect
    """
    # Get API key from Streamlit secrets
    try:
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            api_key = st.secrets['HUGGINGFACEHUB_API_TOKEN']
        elif 'HUGGINGFACE_API_KEY' in st.secrets:
            api_key = st.secrets['HUGGINGFACE_API_KEY']
        elif 'HF_TOKEN' in st.secrets:
            api_key = st.secrets['HF_TOKEN']
        else:
            return f"{defect_type.replace('_', ' ').title()} detected. Professional assessment recommended."
    except:
        return f"{defect_type.replace('_', ' ').title()} detected. Professional assessment recommended."
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        prompt = f"""
        As a structural engineer, provide a concise 1-2 line analysis of this concrete defect:
        
        DEFECT: {defect_type}
        LOCATION: {area_name}
        CONFIDENCE: {confidence:.0%}
        
        Provide only the most essential information: brief description and recommended action.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a structural engineer providing very concise 1-2 line defect analyses. Be direct and practical."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,
            temperature=0.2,
            top_p=0.8,
            stream=False
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"{defect_type.replace('_', ' ').title()} detected. Requires professional assessment."

# ----------------------------
# 3. Helper Functions
# ----------------------------
def analyze_image(image, area_name):
    """Analyze a single image and return defects"""
    with st.spinner(f"Analyzing {area_name}..."):
        results = model(image)
    
    defects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "area": area_name,
                "analysis": None  # Will be populated by LLM
            })
    
    return defects, results[0].plot()

# ----------------------------
# 4. Streamlit UI - Area-based Organization
# ----------------------------
st.title("🏗️ Warehouse Defect Inspection")
st.write("Upload photos organized by warehouse areas for concise defect analysis.")

# Check API status
try:
    has_api_key = any(key in st.secrets for key in ['HUGGINGFACEHUB_API_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_TOKEN'])
    if not has_api_key:
        st.warning("LLM analysis requires Hugging Face API key for optimal results.")
    else:
        st.success("LLM analysis enabled. Ready for concise defect reporting.")
except:
    st.warning("Secrets configuration not accessible.")

# Initialize session state
if 'area_defects' not in st.session_state:
    st.session_state.area_defects = {}

# Area-based photo upload and analysis
st.subheader("📁 Organize by Warehouse Areas")

# Define warehouse areas
warehouse_areas = [
    "North Wall", "South Wall", "East Wall", "West Wall",
    "Floor - Main Area", "Floor - Loading Dock", "Floor - Storage Section",
    "Columns - Main Hall", "Columns - Perimeter", "Columns - Support Beams",
    "Ceiling - Main", "Ceiling - Office Area", "Ceiling - Storage",
    "Doors & Entrances", "Windows & Ventilation", "Other Areas"
]

# Create tabs for each area
area_tabs = st.tabs([f"📍 {area}" for area in warehouse_areas])

for i, area_tab in enumerate(area_tabs):
    with area_tab:
        area_name = warehouse_areas[i]
        st.subheader(f"{area_name}")
        
        # File upload for this specific area
        uploaded_files = st.file_uploader(
            f"Upload photos for {area_name}",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_{i}"
        )
        
        # Process button for this area
        if uploaded_files and st.button(f"Analyze {area_name}", key=f"btn_{i}"):
            area_defects = []
            
            for j, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing image {j+1}/{len(uploaded_files)}..."):
                    image = Image.open(uploaded_file)
                    defects, annotated_image = analyze_image(image, area_name)
                    
                    # Get LLM analysis for each defect
                    for defect in defects:
                        defect['analysis'] = get_llm_concise_analysis(
                            defect['type'], defect['confidence'], area_name
                        )
                    
                    area_defects.extend(defects)
                    
                    # Show image results
                    with st.expander(f"Image {j+1} Results", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original", use_container_width=True)
                        with col2:
                            st.image(annotated_image, caption="Defects Detected", use_container_width=True)
            
            # Store results for this area
            st.session_state.area_defects[area_name] = area_defects
            st.success(f"Analysis complete for {area_name}! Found {len(area_defects)} defects.")

        # Show existing results for this area
        if area_name in st.session_state.area_defects and st.session_state.area_defects[area_name]:
            st.subheader(f"Defects in {area_name}")
            
            defects = st.session_state.area_defects[area_name]
            for k, defect in enumerate(defects, 1):
                st.write(f"**{k}. {defect['type'].replace('_', ' ').title()}**")
                st.write(f"   {defect['analysis']}")
                st.progress(defect['confidence'], text=f"Confidence: {defect['confidence']:.0%}")
                st.write("---")

# Consolidated view of all defects
if st.session_state.area_defects:
    st.subheader("📋 All Detected Defects")
    
    total_defects = 0
    for area_name, defects in st.session_state.area_defects.items():
        if defects:
            total_defects += len(defects)
            with st.expander(f"📍 {area_name} - {len(defects)} defects", expanded=False):
                for i, defect in enumerate(defects, 1):
                    st.write(f"**{i}. {defect['type'].replace('_', ' ').title()}**")
                    st.write(f"   {defect['analysis']}")
                    st.progress(defect['confidence'], text=f"Confidence: {defect['confidence']:.0%}")
                    st.write("---")
    
    # Summary statistics
    if total_defects > 0:
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        areas_with_defects = sum(1 for defects in st.session_state.area_defects.values() if defects)
        unique_defect_types = set()
        for defects in st.session_state.area_defects.values():
            for defect in defects:
                unique_defect_types.add(defect['type'])
        
        col1.metric("Total Defects", total_defects)
        col2.metric("Areas with Defects", areas_with_defects)
        col3.metric("Unique Defect Types", len(unique_defect_types))
        
        # Export option
        report_data = "WAREHOUSE DEFECT INSPECTION REPORT\n"
        report_data += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report_data += "="*50 + "\n\n"
        
        for area_name, defects in st.session_state.area_defects.items():
            if defects:
                report_data += f"AREA: {area_name}\n"
                report_data += "-"*30 + "\n"
                for defect in defects:
                    report_data += f"• {defect['type'].replace('_', ' ').title()}: {defect['analysis']}\n"
                report_data += "\n"
        
        st.download_button(
            label="📄 Export Report",
            data=report_data,
            file_name=f"warehouse_inspection_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Select a warehouse area tab above to upload photos and analyze defects.")

# Quick guide
with st.expander("ℹ️ How to Use"):
    st.write("""
    **Workflow:**
    1. Select a warehouse area tab (walls, floors, columns, etc.)
    2. Upload photos for that specific area
    3. Click "Analyze [Area Name]" to process the photos
    4. Review concise defect analysis for that area
    5. Repeat for other areas as needed
    
    **Features:**
    - Organized by warehouse areas for easy navigation
    - Concise 1-2 line defect analysis using LLM
    - Area-specific photo management
    - Consolidated view of all defects
    """)

st.markdown("---")
st.caption("Warehouse Defect Inspection • Organized by Area • Concise LLM Analysis")
