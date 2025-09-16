import os
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import gdown
import requests
import json
import time
from openai import OpenAI
from datetime import datetime
import numpy as np

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
def analyze_image(image, area_name, filename):
    """Analyze a single image and return defects with image crops"""
    with st.spinner(f"Analyzing {area_name}..."):
        results = model(image)
    
    defects = []
    annotated_image = results[0].plot()
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Create defect crop with outline
            defect_crop = image.copy()
            draw = ImageDraw.Draw(defect_crop)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "area": area_name,
                "analysis": None,
                "bbox": [x1, y1, x2, y2],
                "defect_crop": defect_crop,
                "original_image": image,
                "filename": filename
            })
    
    return defects, annotated_image

def draw_defect_outline(image, bbox):
    """Draw outline around defect area"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(bbox, outline="red", width=3)
    return img_copy

# ----------------------------
# 4. Streamlit UI - Defects Grouped by Photo
# ----------------------------
st.title("🏗️ Warehouse Defect Inspection")
st.write("Create custom areas and upload photos to see defects grouped by photo.")

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
if 'custom_areas' not in st.session_state:
    st.session_state.custom_areas = {}
if 'area_results' not in st.session_state:
    st.session_state.area_results = {}  # Changed to store results by area and photo

# Custom area creation
st.subheader("📁 Create Custom Warehouse Areas")

col1, col2 = st.columns([2, 1])
with col1:
    new_area_name = st.text_input("New Area Name:", placeholder="e.g., North Wall near Entrance")
with col2:
    if st.button("➕ Add Area") and new_area_name:
        if new_area_name not in st.session_state.custom_areas:
            st.session_state.custom_areas[new_area_name] = []
            st.session_state.area_results[new_area_name] = {}
            st.success(f"Added area: {new_area_name}")
        else:
            st.warning("Area already exists!")

# Show existing custom areas
if st.session_state.custom_areas:
    st.subheader("📋 Your Custom Areas")
    
    # Create tabs for each custom area
    area_tabs = st.tabs([f"📍 {area}" for area in st.session_state.custom_areas.keys()])
    
    for i, (area_name, area_photos) in enumerate(st.session_state.custom_areas.items()):
        with area_tabs[i]:
            st.subheader(f"{area_name}")
            
            # File upload for this specific area
            uploaded_files = st.file_uploader(
                f"Upload photos for {area_name}",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"uploader_{area_name}"
            )
            
            # Store uploaded files for this area
            if uploaded_files:
                st.session_state.custom_areas[area_name] = uploaded_files
            
            # Process button for this area
            if st.session_state.custom_areas[area_name] and st.button(f"Analyze {area_name}", key=f"btn_{area_name}"):
                area_photo_results = {}
                
                for j, uploaded_file in enumerate(st.session_state.custom_areas[area_name]):
                    with st.spinner(f"Processing image {j+1}/{len(st.session_state.custom_areas[area_name])}..."):
                        image = Image.open(uploaded_file)
                        defects, annotated_image = analyze_image(image, area_name, uploaded_file.name)
                        
                        # Get LLM analysis for each defect
                        for defect in defects:
                            defect['analysis'] = get_llm_concise_analysis(
                                defect['type'], defect['confidence'], area_name
                            )
                        
                        # Store results by photo
                        area_photo_results[uploaded_file.name] = {
                            'defects': defects,
                            'annotated_image': annotated_image,
                            'original_image': image,
                            'has_defects': len(defects) > 0
                        }
                
                # Store results for this area
                st.session_state.area_results[area_name] = area_photo_results
                st.success(f"Analysis complete for {area_name}!")

            # Show results grouped by photo
            if area_name in st.session_state.area_results and st.session_state.area_results[area_name]:
                st.subheader(f"Photo Results for {area_name}")
                
                photo_results = st.session_state.area_results[area_name]
                
                for filename, result in photo_results.items():
                    with st.expander(f"📷 {filename}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(
                                result['annotated_image'],
                                caption="Annotated Image with Defects",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.image(
                                result['original_image'],
                                caption="Original Image",
                                use_container_width=True
                            )
                        
                        # Show defects for this photo or "No defects" message
                        if result['has_defects']:
                            st.write(f"**Defects found in this photo:**")
                            for k, defect in enumerate(result['defects'], 1):
                                st.write(f"**{k}. {defect['type'].replace('_', ' ').title()}**")
                                st.write(f"**Analysis:** {defect['analysis']}")
                                st.progress(defect['confidence'], text=f"Confidence: {defect['confidence']:.0%}")
                                
                                # Show defect crop
                                st.image(
                                    defect['defect_crop'],
                                    caption=f"Defect {k} location",
                                    use_container_width=True
                                )
                                st.write("---")
                        else:
                            st.success("✅ No defects detected in this photo")
                            st.info("The concrete surface appears to be in good condition.")

# Consolidated view of all areas with photos
if st.session_state.area_results:
    st.subheader("📋 All Inspection Results")
    
    total_photos = 0
    total_defects = 0
    areas_with_defects = 0
    
    for area_name, photo_results in st.session_state.area_results.items():
        total_photos += len(photo_results)
        area_defect_count = 0
        
        for filename, result in photo_results.items():
            area_defect_count += len(result['defects'])
            total_defects += len(result['defects'])
        
        if area_defect_count > 0:
            areas_with_defects += 1
        
        with st.expander(f"📍 {area_name} - {len(photo_results)} photos", expanded=False):
            for filename, result in photo_results.items():
                st.write(f"**Photo:** {filename}")
                
                if result['has_defects']:
                    st.write(f"**Defects:** {len(result['defects'])} found")
                    for i, defect in enumerate(result['defects'], 1):
                        st.write(f"{i}. {defect['type'].replace('_', ' ').title()} - {defect['analysis']}")
                else:
                    st.success("✅ No defects detected")
                
                st.write("---")
    
    # Summary statistics
    if total_photos > 0:
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Areas", len(st.session_state.area_results))
        col2.metric("Total Photos", total_photos)
        col3.metric("Total Defects", total_defects)
        col4.metric("Areas with Defects", areas_with_defects)
        
        # Export option
        report_data = "WAREHOUSE DEFECT INSPECTION REPORT\n"
        report_data += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report_data += "="*50 + "\n\n"
        
        for area_name, photo_results in st.session_state.area_results.items():
            report_data += f"AREA: {area_name}\n"
            report_data += "-"*30 + "\n"
            
            for filename, result in photo_results.items():
                report_data += f"Photo: {filename}\n"
                if result['has_defects']:
                    for defect in result['defects']:
                        report_data += f"  • {defect['type'].replace('_', ' ').title()}: {defect['analysis']} (Confidence: {defect['confidence']:.0%})\n"
                else:
                    report_data += "  • No defects detected\n"
                report_data += "\n"
        
        st.download_button(
            label="📄 Export Report",
            data=report_data,
            file_name=f"warehouse_inspection_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Create custom areas above to start your warehouse inspection.")

# Area management
if st.session_state.custom_areas:
    st.subheader("⚙️ Area Management")
    
    # Show current areas with option to delete
    for area_name in list(st.session_state.custom_areas.keys()):
        col1, col2 = st.columns([3, 1])
        with col1:
            photo_count = len(st.session_state.custom_areas[area_name])
            st.write(f"• {area_name} ({photo_count} photo{'s' if photo_count != 1 else ''})")
        with col2:
            if st.button(f"Delete", key=f"del_{area_name}"):
                del st.session_state.custom_areas[area_name]
                if area_name in st.session_state.area_results:
                    del st.session_state.area_results[area_name]
                st.rerun()

# Quick guide
with st.expander("ℹ️ How to Use"):
    st.write("""
    **Workflow:**
    1. Create custom area names that match your warehouse layout
    2. Upload photos for each specific area
    3. Click "Analyze [Area Name]" to process the photos
    4. See results grouped by photo with "No defects" messages when appropriate
    5. View annotated and original images side by side
    
    **Features:**
    - Defects grouped by photo
    - "No defects" messages for clean areas
    - Side-by-side annotated and original images
    - Visual defect outlines
    - Concise LLM analysis
    """)

st.markdown("---")
st.caption("Warehouse Defect Inspection • Grouped by Photo • Clear No-Defect Messages")
