import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import gdown
import requests
import json
import time
from openai import OpenAI
from datetime import datetime
import numpy as np
from collections import Counter
import base64
from io import BytesIO

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
# 2. LLM Analysis Functions
# ----------------------------
def map_defect_name(defect_type):
    """Map numeric or unclear defect names to meaningful descriptions"""
    defect_mapping = {
        "0": "crack",
        "zero-clearance joint": "crack",
        "00": "hairline crack", 
        "1": "spalling",
        "2": "corrosion",
        "3": "staining",
        "4": "efflorescence",
        "5": "scaling",
        "6": "popout",
        "7": "discoloration",
        "8": "honeycombing",
        "9": "void",
        "crack": "crack",
        "hairline_crack": "hairline crack",
        "medium_crack": "medium crack",
        "wide_crack": "wide crack",
        "spalling": "spalling",
        "corrosion": "corrosion",
        "staining": "staining"
    }
    return defect_mapping.get(defect_type.lower(), defect_type)

def get_llm_concise_analysis(defect_types, area_name):
    """
    Use LLM to generate direct, concise summary of defects with proper naming
    """
    # First, map all defect names to meaningful descriptions
    mapped_defect_types = {}
    for defect_type, count in defect_types.items():
        mapped_name = map_defect_name(defect_type)
        mapped_defect_types[mapped_name] = mapped_defect_types.get(mapped_name, 0) + count
    
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
            defect_summary = ", ".join([f"{count} {defect_name}{'s' if count > 1 else ''}" 
                                      for defect_name, count in mapped_defect_types.items()])
            return defect_summary
    except:
        defect_summary = ", ".join([f"{count} {defect_name}{'s' if count > 1 else ''}" 
                                  for defect_name, count in mapped_defect_types.items()])
        return defect_summary
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        # Create a natural language description of the defects
        defect_description = ", ".join([f"{count} {defect_name}{'s' if count > 1 else ''}" 
                                      for defect_name, count in mapped_defect_types.items()])
        
        prompt = f"""
        Concrete defects: {defect_description} in {area_name}.
        Provide a very brief summary. Just state the defects concisely without explanations.
        Example: "3 hairline cracks detected in north wall"
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a structural engineer. Provide extremely concise defect summaries. No introductions, no explanations. Just state the facts directly. Maximum 1 sentence. Use simple language like 'X cracks detected in Y location'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=30,
            temperature=0.1,
            top_p=0.8,
            stream=False
        )
        
        analysis = response.choices[0].message.content.strip()
        
        # Clean up any verbose language
        analysis = analysis.replace("Based on the provided information, ", "")
        analysis = analysis.replace("I will describe ", "")
        analysis = analysis.replace("The defects include ", "")
        analysis = analysis.replace("There are ", "")
        analysis = analysis.replace("We have detected ", "")
        analysis = analysis.replace("The analysis shows ", "")
        
        return analysis
        
    except Exception as e:
        defect_summary = ", ".join([f"{count} {defect_name}{'s' if count > 1 else ''}" 
                                  for defect_name, count in mapped_defect_types.items()])
        return defect_summary

# ----------------------------
# 3. Helper Functions
# ----------------------------
def analyze_image(image, area_name, filename):
    """Analyze a single image and return defect counts with annotated image"""
    with st.spinner(f"Analyzing {area_name}..."):
        results = model(image)
    
    # Count defects by type and create annotated image
    defect_counts = Counter()
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Map the defect name to something meaningful
            mapped_name = map_defect_name(class_name)
            defect_counts[mapped_name] += 1
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label with confidence (using mapped name)
            label = f"{mapped_name} ({confidence:.2f})"
            draw.text((x1, y1 - 20), label, fill="red", font=font)
    
    return defect_counts, annotated_image, image

def create_html_report(project_name, area_results, user_comments):
    """Create an HTML report of the inspection results"""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .project-name {{ font-size: 24px; font-weight: bold; }}
            .date {{ font-size: 14px; color: #666; }}
            .area-section {{ margin-bottom: 25px; page-break-inside: avoid; }}
            .area-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            .photo-section {{ margin-bottom: 15px; }}
            .defect-summary {{ margin: 5px 0; }}
            .comments {{ margin: 5px 0; font-style: italic; color: #555; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="project-name">{project_name}</div>
            <div class="date">Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
    """
    
    for area_name, photos in area_results.items():
        area_has_defects = any(photo_result['has_defects'] for photo_result in photos.values())
        
        if area_has_defects:
            html_content += f'<div class="area-section">'
            html_content += f'<div class="area-title">Area: {area_name}</div>'
            
            for filename, result in photos.items():
                if result['has_defects']:
                    defect_summary = ", ".join([
                        f"{count} {defect_type}{'s' if count > 1 else ''}" 
                        for defect_type, count in result['defect_counts'].items()
                    ])
                    
                    html_content += f'''
                    <div class="photo-section">
                        <div class="defect-summary"><strong>Photo:</strong> {filename}</div>
                        <div class="defect-summary"><strong>Defects:</strong> {defect_summary}</div>
                        <div class="defect-summary"><strong>Summary:</strong> {result['analysis']}</div>
                    '''
                    
                    # Add user comments if available
                    user_comment = user_comments.get(area_name, {}).get(filename, "")
                    if user_comment:
                        html_content += f'<div class="comments"><strong>Comments:</strong> {user_comment}</div>'
                    
                    html_content += '</div>'
            
            html_content += '</div>'
    
    html_content += "</body></html>"
    return html_content

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Defect Inspection")
st.write("Manage projects and analyze defects across multiple warehouse areas.")

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}

# Project Management Sidebar
with st.sidebar:
    st.header("📁 Project Management")
    
    # Create new project
    new_project_name = st.text_input("New Project Name:", placeholder="e.g., Warehouse A Inspection")
    if st.button("➕ Create New Project") and new_project_name:
        if new_project_name not in st.session_state.projects:
            st.session_state.projects[new_project_name] = {
                'created_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'areas': {},
                'results': {},
                'comments': {}
            }
            st.session_state.current_project = new_project_name
            st.success(f"Created project: {new_project_name}")
        else:
            st.warning("Project name already exists!")
    
    # Project selection
    if st.session_state.projects:
        project_names = list(st.session_state.projects.keys())
        selected_project = st.selectbox(
            "Select Project:",
            project_names,
            index=project_names.index(st.session_state.current_project) if st.session_state.current_project else 0
        )
        
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            st.rerun()

# Main content based on selected project
if st.session_state.current_project:
    project = st.session_state.projects[st.session_state.current_project]
    
    st.header(f"Project: {st.session_state.current_project}")
    st.caption(f"Created: {project['created_date']}")
    
    # Custom area creation
    st.subheader("📁 Create Warehouse Areas")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        new_area_name = st.text_input("New Area Name:", placeholder="e.g., North Wall near Entrance", key="new_area")
    with col2:
        if st.button("➕ Add Area", key="add_area") and new_area_name:
            if new_area_name not in project['areas']:
                project['areas'][new_area_name] = []
                project['results'][new_area_name] = {}
                project['comments'][new_area_name] = {}
                st.success(f"Added area: {new_area_name}")
            else:
                st.warning("Area already exists!")
    
    # Show existing areas and file uploads
    if project['areas']:
        st.subheader("📋 Warehouse Areas")
        
        # Create tabs for each area
        area_tabs = st.tabs([f"📍 {area}" for area in project['areas'].keys()])
        
        for i, (area_name, area_photos) in enumerate(project['areas'].items()):
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
                    project['areas'][area_name] = uploaded_files
                
                # Show individual photo analysis with "Use AI Summary" button
                if area_name in project['results'] and project['results'][area_name]:
                    st.subheader("📸 Photo Analysis")
                    for filename, result in project['results'][area_name].items():
                        with st.expander(f"Photo: {filename}", expanded=False):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.image(result['annotated_image'], caption="Annotated Image", use_container_width=True)
                            
                            with col2:
                                if result['has_defects']:
                                    defect_summary = ", ".join([
                                        f"{count} {defect_type}{'s' if count > 1 else ''}" 
                                        for defect_type, count in result['defect_counts'].items()
                                    ])
                                    st.write(f"**Defects found:** {defect_summary}")
                                    st.write("**Summary:**", result['analysis'])
                                    
                                    # "Use AI Summary" button
                                    st.subheader("📝 Additional Comments")
                                    
                                    # Create columns for the button and text area
                                    btn_col, _ = st.columns([2, 3])
                                    
                                    with btn_col:
                                        if st.button("📋 Use AI Summary", key=f"ai_btn_{area_name}_{filename}", use_container_width=True):
                                            # Populate text area with AI analysis
                                            project['comments'][area_name][filename] = result['analysis']
                                            st.rerun()
                                    
                                    # User comments text area
                                    comment_key = f"comment_{area_name}_{filename}"
                                    user_comment = st.text_area(
                                        "Add your observations:",
                                        value=project['comments'][area_name].get(filename, ""),
                                        height=100,
                                        key=comment_key
                                    )
                                    
                                    # Store user comment
                                    project['comments'][area_name][filename] = user_comment
                                    
                                else:
                                    st.success("✅ No defects detected in this photo")
    
    # Single "Analyze All Areas" button
    if project['areas'] and any(project['areas'].values()):
        if st.button("🔍 Analyze All Areas", type="primary", use_container_width=True):
            with st.spinner("Analyzing all areas..."):
                for area_name, uploaded_files in project['areas'].items():
                    if uploaded_files:
                        area_photo_results = {}
                        
                        for j, uploaded_file in enumerate(uploaded_files):
                            with st.spinner(f"Processing {area_name} - image {j+1}/{len(uploaded_files)}..."):
                                image = Image.open(uploaded_file)
                                defect_counts, annotated_image, original_image = analyze_image(image, area_name, uploaded_file.name)
                                
                                # Get LLM analysis for the defects in this photo
                                analysis = get_llm_concise_analysis(defect_counts, area_name)
                                
                                # Store results by photo
                                area_photo_results[uploaded_file.name] = {
                                    'defect_counts': defect_counts,
                                    'analysis': analysis,
                                    'annotated_image': annotated_image,
                                    'original_image': original_image,
                                    'has_defects': sum(defect_counts.values()) > 0
                                }
                                
                                # Initialize user comments
                                if area_name not in project['comments']:
                                    project['comments'][area_name] = {}
                                project['comments'][area_name][uploaded_file.name] = ""
                        
                        # Store results for this area
                        project['results'][area_name] = area_photo_results
                
                st.success("Analysis complete for all areas!")
    
    # All Inspection Results Section (Only show photos with defects)
    if any(any(photo_result['has_defects'] for photo_result in area_results.values()) 
           for area_results in project['results'].values() if area_results):
        
        st.header("📋 All Inspection Results (Defects Only)")
        
        total_defects = 0
        total_photos_with_defects = 0
        areas_with_defects = 0
        
        # Calculate totals
        for area_name, area_results in project['results'].items():
            area_has_defects = False
            for result in area_results.values():
                if result['has_defects']:
                    total_defects += sum(result['defect_counts'].values())
                    total_photos_with_defects += 1
                    area_has_defects = True
            if area_has_defects:
                areas_with_defects += 1
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Areas with Defects", areas_with_defects)
        col2.metric("Photos with Defects", total_photos_with_defects)
        col3.metric("Total Defects", total_defects)
        
        # Display results with thumbnails
        for area_name, area_results in project['results'].items():
            area_has_defects = any(result['has_defects'] for result in area_results.values())
            
            if area_has_defects:
                with st.expander(f"📍 {area_name}", expanded=True):
                    photo_count = 0
                    
                    for filename, result in area_results.items():
                        if result['has_defects']:
                            photo_count += 1
                            st.write(f"**Photo {photo_count}:** {filename}")
                            
                            # Create thumbnail and text layout
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                # Create thumbnail
                                thumbnail = result['annotated_image'].copy()
                                thumbnail.thumbnail((200, 200))
                                st.image(thumbnail, use_container_width=True)
                            
                            with col2:
                                defect_summary = ", ".join([
                                    f"{count} {defect_type}{'s' if count > 1 else ''}" 
                                    for defect_type, count in result['defect_counts'].items()
                                ])
                                
                                st.write(f"**Defects:** {defect_summary}")
                                st.write(f"**Summary:** {result['analysis']}")
                                
                                # User comments
                                user_comment = project['comments'][area_name].get(filename, "")
                                if user_comment:
                                    st.write(f"**Comments:** {user_comment}")
                            
                            st.write("---")
        
        # HTML Export (replacing PDF)
        st.subheader("📄 Export Report")
        html_report = create_html_report(
            st.session_state.current_project,
            project['results'],
            project['comments']
        )
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name=f"{st.session_state.current_project}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )

else:
    st.info("👆 Create a new project to get started with warehouse inspection.")

# Quick guide
with st.expander("ℹ️ How to Use"):
    st.write("""
    **Workflow:**
    1. Create a new project with a descriptive name
    2. Add warehouse areas to the project
    3. Upload photos for each area
    4. Click "Analyze All Areas" to process all photos
    5. Review results in the "All Inspection Results" section
    6. Download HTML reports for documentation
    
    **Features:**
    - Project-based organization
    - Single "Analyze All Areas" button
    - "Use AI Summary" button for quick comments
    - Thumbnail view of defective photos
    - HTML report generation
    - Running count of defects and affected areas
    """)

st.markdown("---")
st.caption("Warehouse Defect Inspection • Project Management • Professional Reporting")
