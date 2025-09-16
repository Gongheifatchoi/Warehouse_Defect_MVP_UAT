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
import numpy as np
from collections import Counter

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
def get_llm_concise_analysis(defect_types, area_name):
    """
    Use LLM to generate concise analysis of the defects in a photo
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
            # Fallback description without LLM
            defect_summary = ", ".join([f"{count} {defect_type}{'s' if count > 1 else ''}" 
                                      for defect_type, count in defect_types.items()])
            return f"Detected {defect_summary}. Professional assessment recommended."
    except:
        defect_summary = ", ".join([f"{count} {defect_type}{'s' if count > 1 else ''}" 
                                  for defect_type, count in defect_types.items()])
        return f"Detected {defect_summary}. Professional assessment recommended."
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        # Create a natural language description of the defects
        defect_description = ", ".join([f"{count} {defect_type.replace('_', ' ')}{'s' if count > 1 else ''}" 
                                      for defect_type, count in defect_types.items()])
        
        prompt = f"""
        As a structural engineer, provide a concise analysis of these concrete defects found in a warehouse:
        
        DEFECTS: {defect_description}
        LOCATION: {area_name}
        
        Provide a brief summary of the issues found and recommended actions.
        Be specific about the defect types and provide practical advice.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a structural engineer providing concise defect analyses. Be direct, practical, and specific about concrete defects. Focus on the overall condition and recommended actions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,
            temperature=0.2,
            top_p=0.8,
            stream=False
        )
        
        analysis = response.choices[0].message.content
        return analysis
        
    except Exception as e:
        defect_summary = ", ".join([f"{count} {defect_type}{'s' if count > 1 else ''}" 
                                  for defect_type, count in defect_types.items()])
        return f"Detected {defect_summary}. Professional assessment recommended."

# ----------------------------
# 3. Helper Functions
# ----------------------------
def analyze_image(image, area_name, filename):
    """Analyze a single image and return defect counts"""
    with st.spinner(f"Analyzing {area_name}..."):
        results = model(image)
    
    # Count defects by type
    defect_counts = Counter()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            defect_counts[class_name] += 1
    
    return defect_counts, image

# ----------------------------
# 4. Streamlit UI - Defects Summary by Photo
# ----------------------------
st.title("🏗️ Warehouse Defect Inspection")
st.write("Create custom areas and upload photos to get a summary of defects in each photo.")

# Check API status
try:
    has_api_key = any(key in st.secrets for key in ['HUGGINGFACEHUB_API_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_TOKEN'])
    if not has_api_key:
        st.warning("LLM analysis requires Hugging Face API key for optimal results.")
    else:
        st.success("LLM analysis enabled. Ready for defect reporting.")
except:
    st.warning("Secrets configuration not accessible.")

# Initialize session state
if 'custom_areas' not in st.session_state:
    st.session_state.custom_areas = {}
if 'area_results' not in st.session_state:
    st.session_state.area_results = {}

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
                        defect_counts, original_image = analyze_image(image, area_name, uploaded_file.name)
                        
                        # Get LLM analysis for the defects in this photo
                        analysis = get_llm_concise_analysis(defect_counts, area_name)
                        
                        # Store results by photo
                        area_photo_results[uploaded_file.name] = {
                            'defect_counts': defect_counts,
                            'analysis': analysis,
                            'original_image': original_image,
                            'has_defects': sum(defect_counts.values()) > 0
                        }
                
                # Store results for this area
                st.session_state.area_results[area_name] = area_photo_results
                
                # Show summary
                total_defects = sum(sum(result['defect_counts'].values()) for result in area_photo_results.values())
                st.success(f"Analysis complete for {area_name}! Found {total_defects} defects across {len(area_photo_results)} photos.")

            # Show results grouped by photo
            if area_name in st.session_state.area_results and st.session_state.area_results[area_name]:
                st.subheader(f"Photo Results for {area_name}")
                
                photo_results = st.session_state.area_results[area_name]
                
                for filename, result in photo_results.items():
                    with st.expander(f"📷 {filename}", expanded=True):
                        # Show the original photo
                        st.image(
                            result['original_image'],
                            caption="Original Photo",
                            use_container_width=True
                        )
                        
                        # Show defects for this photo or "No defects" message
                        if result['has_defects']:
                            # Create a natural language summary of defects
                            defect_summary = ", ".join([
                                f"{count} {defect_type.replace('_', ' ')}{'s' if count > 1 else ''}" 
                                for defect_type, count in result['defect_counts'].items()
                            ])
                            
                            st.write(f"**Defects found:** {defect_summary}")
                            st.write("**Analysis:**", result['analysis'])
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
            area_defect_count += sum(result['defect_counts'].values())
            total_defects += sum(result['defect_counts'].values())
        
        if area_defect_count > 0:
            areas_with_defects += 1
        
        with st.expander(f"📍 {area_name} - {len(photo_results)} photos", expanded=False):
            for filename, result in photo_results.items():
                st.write(f"**Photo:** {filename}")
                
                if result['has_defects']:
                    defect_summary = ", ".join([
                        f"{count} {defect_type.replace('_', ' ')}{'s' if count > 1 else ''}" 
                        for defect_type, count in result['defect_counts'].items()
                    ])
                    st.write(f"**Defects:** {defect_summary}")
                    st.write(f"**Analysis:** {result['analysis']}")
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

else:
    st.info("👆 Create custom areas above to start your warehouse inspection.")

# Quick guide
with st.expander("ℹ️ How to Use"):
    st.write("""
    **Workflow:**
    1. Create custom area names that match your warehouse layout
    2. Upload photos for each specific area
    3. Click "Analyze [Area Name]" to process the photos
    4. See a summary of defects found in each photo
    5. Get concise analysis of the issues
    
    **Features:**
    - Summary of defects per photo (no individual highlighting)
    - Plural terms used when multiple defects of same type found
    - Concise LLM analysis of overall issues in each photo
    - Clear "No defects" messages when appropriate
    """)

st.markdown("---")
st.caption("Warehouse Defect Inspection • Photo Summary • Professional Analysis")
