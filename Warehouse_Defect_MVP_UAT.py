import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
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
# 2. Helper Functions
# ----------------------------
def analyze_image(image, area_name):
    """Analyze a single image and return defects with concise descriptions"""
    with st.spinner(f"Analyzing {area_name}..."):
        results = model(image)
    
    defects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Generate concise description based on defect type
            description = generate_defect_description(class_name, confidence)
            
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "description": description,
                "area": area_name
            })
    
    return defects, results[0].plot()

def generate_defect_description(defect_type, confidence):
    """Generate concise 1-2 line descriptions for each defect type"""
    descriptions = {
        "crack": f"Hairline crack detected ({confidence:.0%} confidence). Monitor for width changes.",
        "hairline_crack": f"Fine hairline crack ({confidence:.0%} confidence). Typically cosmetic but monitor progression.",
        "medium_crack": f"Medium-width crack ({confidence:.0%} confidence). Requires inspection for structural implications.",
        "wide_crack": f"Significant crack ({confidence:.0%} confidence). Immediate professional assessment recommended.",
        "spalling": f"Concrete spalling detected ({confidence:.0%} confidence). Surface deterioration exposing aggregate.",
        "corrosion": f"Corrosion staining ({confidence:.0%} confidence). Indicates potential reinforcement deterioration.",
        "staining": f"Surface staining ({confidence:.0%} confidence). May indicate moisture penetration or chemical exposure.",
        "efflorescence": f"Efflorescence present ({confidence:.0%} confidence). Salt deposits indicating moisture migration.",
        "scaling": f"Surface scaling ({confidence:.0%} confidence). Freeze-thaw or chemical damage evident.",
        "popout": f"Popout defect ({confidence:.0%} confidence). Localized concrete surface failure.",
        "discoloration": f"Discoloration detected ({confidence:.0%} confidence). May indicate material inconsistencies.",
        "honeycombing": f"Honeycombing present ({confidence:.0%} confidence). Poor compaction during construction.",
        "void": f"Surface void ({confidence:.0%} confidence). Air pocket or imperfect finishing.",
    }
    
    # Find the best matching description
    for key, desc in descriptions.items():
        if key in defect_type.lower():
            return desc
    
    # Default description for unknown defect types
    return f"{defect_type.replace('_', ' ').title()} detected ({confidence:.0%} confidence). Professional assessment recommended."

# ----------------------------
# 3. Streamlit UI - Simplified List View
# ----------------------------
st.title("🏗️ Warehouse Defect Inspection")
st.write("Upload photos of different warehouse areas to generate a concise defect list.")

# Initialize session state for defects
if 'all_defects' not in st.session_state:
    st.session_state.all_defects = []

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Upload warehouse area photos", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    help="Upload photos of different areas: floors, walls, columns, ceilings, etc."
)

# Area naming for each uploaded file
if uploaded_files:
    st.subheader("📝 Identify Warehouse Areas")
    
    area_names = {}
    for i, uploaded_file in enumerate(uploaded_files):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded_file, width=100)
        with col2:
            area_name = st.text_input(
                f"Area description for image {i+1}:",
                placeholder="e.g., North wall, Column B2, Main floor section",
                key=f"area_{i}"
            )
            area_names[i] = area_name if area_name else f"Area {i+1}"

# Process images button
if uploaded_files and st.button("🔍 Analyze All Images", type="primary"):
    st.session_state.all_defects = []  # Reset defects
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        area_name = area_names[i]
        status_text.text(f"Analyzing {area_name}... ({i+1}/{len(uploaded_files)})")
        
        # Process image
        image = Image.open(uploaded_file)
        defects, annotated_image = analyze_image(image, area_name)
        
        # Store results
        st.session_state.all_defects.extend(defects)
        
        # Show quick preview
        with st.expander(f"📸 {area_name} - {len(defects)} defects found", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="Defects Detected", use_container_width=True)
            
            # Show defects for this area
            if defects:
                st.write("**Defects in this area:**")
                for defect in defects:
                    st.write(f"• {defect['description']}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Analysis complete!")
    progress_bar.empty()

# Display consolidated defect list
if st.session_state.all_defects:
    st.subheader("📋 Consolidated Defect List")
    
    # Group defects by area
    defects_by_area = {}
    for defect in st.session_state.all_defects:
        if defect['area'] not in defects_by_area:
            defects_by_area[defect['area']] = []
        defects_by_area[defect['area']].append(defect)
    
    # Display defects in a clean list view
    for area, defects in defects_by_area.items():
        with st.expander(f"📍 {area} - {len(defects)} defects", expanded=True):
            for i, defect in enumerate(defects, 1):
                st.write(f"**{i}. {defect['type'].replace('_', ' ').title()}**")
                st.write(f"   {defect['description']}")
                st.progress(defect['confidence'], text=f"Confidence: {defect['confidence']:.0%}")
                st.write("---")
    
    # Summary statistics
    st.subheader("📊 Inspection Summary")
    col1, col2, col3 = st.columns(3)
    
    total_defects = len(st.session_state.all_defects)
    unique_defect_types = len(set(defect['type'] for defect in st.session_state.all_defects))
    areas_inspected = len(defects_by_area)
    
    col1.metric("Total Defects", total_defects)
    col2.metric("Unique Defect Types", unique_defect_types)
    col3.metric("Areas Inspected", areas_inspected)
    
    # Defect type distribution
    if total_defects > 0:
        st.write("**Defect Type Distribution:**")
        defect_counts = {}
        for defect in st.session_state.all_defects:
            defect_type = defect['type'].replace('_', ' ').title()
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        for defect_type, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"• {defect_type}: {count} instance(s)")
    
    # Export option
    st.download_button(
        label="📄 Export Defect Report",
        data="\n".join([f"{d['area']}: {d['description']}" for d in st.session_state.all_defects]),
        file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Upload photos of warehouse areas to begin inspection. Add descriptive names for each area for better organization.")

# Quick guide
with st.expander("ℹ️ Inspection Guide"):
    st.write("""
    **How to use:**
    1. Upload photos of different warehouse areas (floors, walls, columns, etc.)
    2. Provide descriptive names for each area (e.g., "North wall near entrance")
    3. Click "Analyze All Images" to process all photos
    4. Review the consolidated defect list with concise descriptions
    
    **Defect Severity Guidelines:**
    - **Hairline cracks**: Monitor, typically cosmetic
    - **Medium cracks**: Requires professional assessment
    - **Wide cracks**: Immediate attention needed
    - **Spalling/Corrosion**: Structural assessment recommended
    - **Staining/Efflorescence**: Investigate moisture sources
    """)

# Add footer
st.markdown("---")
st.caption("Warehouse Defect Inspection System • Automated defect detection with concise reporting")
