import os
import streamlit as st
import requests
import time
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Warehouse Defect Detector",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .defect-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ff6b6b;
    }
    .success-card {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🏗️ Warehouse Concrete Defect Detection</h1>', unsafe_allow_html=True)
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

# Check if we're in a server environment (no GUI)
IS_SERVER_ENVIRONMENT = not os.path.exists("/.dockerenv") and 'AWS_EXECUTION_ENV' not in os.environ

if IS_SERVER_ENVIRONMENT:
    st.info("ℹ️ Running in server environment. Using compatible mode.")

# Try to use Ultralytics/YOLO if available, otherwise use demo mode
DEMO_MODE = True
try:
    # Try to import ultralytics - this will fail if OpenCV has GUI dependencies
    from ultralytics import YOLO
    import gdown
    
    # If we get here, the import succeeded
    DEMO_MODE = False
    
    # Model setup
    MODEL_PATH = "best.pt"
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1LIAP45Ab8diOfvT2HVL1pwmZyspeXEDi"

    @st.cache_resource
    def load_model():
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading YOLO model, please wait..."):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        return YOLO(MODEL_PATH)

    model = load_model()
    
except ImportError as e:
    st.warning(f"⚠️ Import error: {e}")
    st.info("Falling back to demo mode...")
    DEMO_MODE = True
except Exception as e:
    st.warning(f"⚠️ Error loading model: {e}")
    st.info("Falling back to demo mode...")
    DEMO_MODE = True

# Sample defect data for demo mode
SAMPLE_DEFECTS = [
    {"type": "crack", "confidence": 0.85, "location": [100, 150, 30, 10], "severity": "medium"},
    {"type": "spall", "confidence": 0.72, "location": [200, 80, 40, 40], "severity": "low"},
    {"type": "discoloration", "confidence": 0.68, "location": [150, 250, 60, 60], "severity": "low"}
]

def analyze_image_demo(image):
    """Demo function that returns sample defects"""
    return SAMPLE_DEFECTS

def analyze_image_real(image):
    """Real function that uses YOLO model"""
    try:
        results = model(image)
        defects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Determine severity based on confidence
                if confidence > 0.8:
                    severity = "high"
                elif confidence > 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                defects.append({
                    "type": class_name,
                    "confidence": confidence,
                    "location": [float(coord) for coord in box.xywh[0]],
                    "severity": severity
                })
        return defects
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return []

# Enhanced Local Expert System
def get_expert_analysis(defects, image_name="the image"):
    """Generate comprehensive expert commentary using an enhanced local system"""
    if not defects:
        return {
            "summary": "No defects detected. The concrete surface appears to be in good condition.",
            "severity": "None",
            "recommendations": ["Continue regular maintenance inspections"],
            "risk_factors": ["None identified"],
            "maintenance_schedule": "Continue quarterly inspections"
        }
    
    # Count defects by type and calculate statistics
    defect_counts = {}
    confidence_scores = {}
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    
    for defect in defects:
        defect_type = defect['type']
        defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        if defect_type not in confidence_scores:
            confidence_scores[defect_type] = []
        confidence_scores[defect_type].append(defect['confidence'])
        severity_counts[defect['severity']] += 1
    
    # Calculate statistics
    total_defects = len(defects)
    avg_confidence = sum(sum(scores) for scores in confidence_scores.values()) / total_defects
    
    # Determine overall severity
    if severity_counts["high"] > 0:
        overall_severity = "High"
    elif severity_counts["medium"] > 0:
        overall_severity = "Medium"
    else:
        overall_severity = "Low"
    
    # Generate comprehensive analysis
    analysis = f"## Comprehensive Analysis\n\n"
    analysis += f"**Defect Assessment Summary**:\n"
    analysis += f"- Total defects identified: {total_defects}\n"
    analysis += f"- Overall severity: {overall_severity}\n"
    analysis += f"- Average detection confidence: {avg_confidence:.1%}\n\n"
    
    analysis += "**Defect Breakdown**:\n"
    for defect_type, count in defect_counts.items():
        avg_conf = sum(confidence_scores[defect_type]) / len(confidence_scores[defect_type])
        analysis += f"- {defect_type.capitalize()}: {count} instances ({avg_conf:.1%} confidence)\n"
    
    analysis += "\n**Severity Distribution**:\n"
    analysis += f"- High severity: {severity_counts['high']} defects\n"
    analysis += f"- Medium severity: {severity_counts['medium']} defects\n"
    analysis += f"- Low severity: {severity_counts['low']} defects\n\n"
    
    # Recommendations based on defects
    analysis += "**Recommended Actions**:\n"
    
    if any(d['type'] == 'crack' for d in defects):
        analysis += "- Monitor crack width progression over 4-6 weeks\n"
        analysis += "- Consider epoxy injection for cracks > 0.3mm\n"
        analysis += "- Consult structural engineer for cracks > 3mm\n"
    
    if any(d['type'] == 'spall' for d in defects):
        analysis += "- Remove loose material and treat exposed reinforcement\n"
        analysis += "- Apply corrosion inhibitor to exposed steel\n"
        analysis += "- Patch with appropriate concrete repair materials\n"
    
    if any(d['type'] == 'void' for d in defects):
        analysis += "- Fill voids with non-shrink grout after proper preparation\n"
        analysis += "- Ensure proper compaction around repaired areas\n"
    
    if any(d['type'] == 'discoloration' for d in defects):
        analysis += "- Identify and address moisture source\n"
        analysis += "- Clean surface with appropriate methods\n"
        analysis += "- Consider waterproofing measures if moisture persists\n"
    
    analysis += "\n**Maintenance Schedule**:\n"
    if overall_severity == "High":
        analysis += "- Immediate: Professional assessment required\n"
        analysis += "- 30 days: Implement recommended repairs\n"
        analysis += "- 90 days: Follow-up inspection\n"
    elif overall_severity == "Medium":
        analysis += "- 30 days: Targeted repairs\n"
        analysis += "- 90 days: Progress evaluation\n"
        analysis += "- 6 months: Comprehensive inspection\n"
    else:
        analysis += "- 90 days: Follow-up inspection\n"
        analysis += "- 6 months: Comprehensive review\n"
        analysis += "- Annual: Routine monitoring program\n"
    
    analysis += "\n**Risk Factors**:\n"
    if any(d['severity'] == 'high' for d in defects):
        analysis += "- ⚠️ High severity defects may indicate structural concerns\n"
    if any(d['type'] == 'crack' for d in defects):
        analysis += "- ⚠️ Cracks can allow water infiltration leading to reinforcement corrosion\n"
    if any(d['type'] == 'spall' for d in defects):
        analysis += "- ⚠️ Spalling concrete can fall and create safety hazards\n"
    
    analysis += "\n**When to Consult a Structural Engineer**:\n"
    analysis += "- Defects are rapidly progressing or changing\n"
    analysis += "- Cracks wider than 3mm are present\n"
    analysis += "- Structural deformation is visible\n"
    analysis += "- Uncertainty about defect significance\n"
    
    return {
        "full_analysis": analysis,
        "severity": overall_severity,
        "total_defects": total_defects,
        "avg_confidence": avg_confidence
    }

# Main application logic
def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image for defects..."):
                # Analyze image based on mode
                if DEMO_MODE:
                    defects = analyze_image_demo(image)
                    st.info("🔬 Running in demo mode with sample data")
                else:
                    defects = analyze_image_real(image)
                
                # Display results
                if defects:
                    st.subheader("📊 Detection Results")
                    
                    # Show defects in a table
                    for i, defect in enumerate(defects, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="defect-card">
                                <b>Defect {i}: {defect['type'].capitalize()}</b><br>
                                Confidence: {(defect['confidence']*100):.1f}%<br>
                                Severity: {defect['severity'].capitalize()}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Get expert analysis
                    analysis = get_expert_analysis(defects, uploaded_file.name)
                    
                    # Display analysis
                    st.subheader("🧠 Expert Analysis")
                    st.markdown(analysis["full_analysis"])
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Defects", analysis["total_defects"])
                    with col2:
                        st.metric("Average Confidence", f"{analysis['avg_confidence']:.1%}")
                    with col3:
                        st.metric("Overall Severity", analysis["severity"])
                    
                else:
                    st.markdown("""
                    <div class="success-card">
                        <h3>✅ No defects detected!</h3>
                        <p>The concrete surface appears to be in good condition.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.write("This app detects defects in warehouse concrete surfaces using AI.")
        
        if DEMO_MODE:
            st.warning("Running in Demo Mode")
            st.write("Sample data is being used instead of real AI detection.")
            st.write("To enable full AI detection, install the headless version of OpenCV:")
            st.code("""
pip install opencv-python-headless
pip install ultralytics gdown
            """)
        else:
            st.success("Full AI Mode Enabled")
            st.write("YOLO model is loaded and ready for real defect detection.")
        
        st.header("📋 Supported Defects")
        st.write("- Cracks")
        st.write("- Spalling")
        st.write("- Discoloration") 
        st.write("- Voids")
        st.write("- Other surface defects")
        
        st.header("🔧 Installation Guide")
        st.write("For server environments, use headless OpenCV:")
        st.code("""
# Uninstall regular opencv if installed
pip uninstall opencv-python

# Install headless version
pip install opencv-python-headless
pip install ultralytics gdown streamlit
        """)

if __name__ == "__main__":
    main()
