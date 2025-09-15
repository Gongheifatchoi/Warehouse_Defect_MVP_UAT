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
    .api-test-success {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    .api-test-failure {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🏗️ Warehouse Concrete Defect Detection</h1>', unsafe_allow_html=True)
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

# Get Hugging Face API key from Streamlit secrets
try:
    HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
    st.sidebar.success("✅ Hugging Face API key loaded from secrets")
except (KeyError, FileNotFoundError):
    HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
    if HF_API_KEY:
        st.sidebar.info("ℹ️ Hugging Face API key loaded from environment variables")
    else:
        st.sidebar.warning("⚠️ Hugging Face API key not found. AI analysis will be limited.")

# Hugging Face API Test Function
def test_hugging_face_api(api_key):
    """Test if the Hugging Face API key is valid"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            return True, f"✅ API Key Valid\nConnected as: {user_info.get('name', 'Unknown')}"
        else:
            return False, f"❌ API Error: {response.status_code}"
            
    except Exception as e:
        return False, f"❌ Connection Error: {str(e)}"

# Hugging Face AI Analysis Function
def get_ai_analysis(defects_info, api_key):
    """Get AI commentary using Hugging Face API"""
    if not api_key:
        return None, "Hugging Face API key is required for AI analysis."
    
    prompt = f"""
    As a structural engineering expert, analyze these concrete defects detected in a warehouse:
    {defects_info}
    
    Please provide a professional analysis including:
    1. Severity assessment of the defects
    2. Potential causes for each defect type
    3. Recommended repair actions
    4. Safety implications
    5. Maintenance recommendations
    
    Keep the response concise and focused on structural engineering best practices.
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        # Try multiple models
        models = [
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        ]
        
        for model_url in models:
            try:
                response = requests.post(
                    model_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', 'No analysis generated'), None
                    elif isinstance(result, dict):
                        return result.get('generated_text', 'No analysis generated'), None
                elif response.status_code == 503:
                    continue  # Model loading, try next one
                    
            except requests.exceptions.Timeout:
                continue  # Try next model
            except requests.exceptions.ConnectionError:
                continue  # Try next model
                
        return None, "All AI models are currently unavailable. Please try again later."
        
    except Exception as e:
        return None, f"Error accessing AI service: {str(e)}"

# Test API connection in sidebar
if HF_API_KEY:
    if st.sidebar.button("Test API Connection"):
        with st.sidebar:
            with st.spinner("Testing API connection..."):
                success, message = test_hugging_face_api(HF_API_KEY)
                if success:
                    st.markdown(f'<div class="api-test-success">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="api-test-failure">{message}</div>', unsafe_allow_html=True)

# Check if we're in Streamlit sharing environment
IS_STREAMLIT_SHARING = 'SHARING' in os.environ

if IS_STREAMLIT_SHARING:
    st.warning("⚠️ Running in Streamlit Sharing environment. Using demo mode.")
    DEMO_MODE = True
else:
    DEMO_MODE = False

try:
    if not DEMO_MODE:
        from ultralytics import YOLO
        import gdown
        
        # Model setup
        MODEL_PATH = "best.pt"
        MODEL_URL = "https://drive.google.com/uc?export=download&id=1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1"

        @st.cache_resource
        def load_model():
            if not os.path.exists(MODEL_PATH):
                with st.spinner("Downloading YOLO model, please wait..."):
                    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
            return YOLO(MODEL_PATH)

        model = load_model()
        
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Falling back to demo mode...")
    DEMO_MODE = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
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
        return "No defects detected. The concrete surface appears to be in good condition."
    
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
    
    return analysis

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
                    
                    # Prepare defect information for AI analysis
                    defects_info = "\n".join([
                        f"- {d['type']} (confidence: {d['confidence']:.2f}, severity: {d['severity']})"
                        for d in defects
                    ])
                    
                    # Get expert analysis
                    st.subheader("🧠 Expert Analysis")
                    
                    # Try to get AI analysis if API key is available
                    ai_analysis = None
                    if HF_API_KEY:
                        with st.spinner("Getting AI analysis..."):
                            ai_analysis, error = get_ai_analysis(defects_info, HF_API_KEY)
                            if error:
                                st.warning(f"AI Analysis Unavailable: {error}")
                    
                    # Display AI analysis if available, otherwise use local analysis
                    if ai_analysis:
                        st.write("**AI-Powered Analysis:**")
                        st.write(ai_analysis)
                        st.divider()
                        st.write("**Local Expert Analysis:**")
                    
                    # Always show local expert analysis as fallback
                    local_analysis = get_expert_analysis(defects, uploaded_file.name)
                    st.markdown(local_analysis)
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    total_defects = len(defects)
                    avg_confidence = sum(d['confidence'] for d in defects) / total_defects if defects else 0
                    
                    # Determine overall severity
                    severity_counts = {"low": 0, "medium": 0, "high": 0}
                    for d in defects:
                        severity_counts[d['severity']] += 1
                    
                    if severity_counts["high"] > 0:
                        overall_severity = "High"
                    elif severity_counts["medium"] > 0:
                        overall_severity = "Medium"
                    else:
                        overall_severity = "Low"
                    
                    with col1:
                        st.metric("Total Defects", total_defects)
                    with col2:
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                    with col3:
                        st.metric("Overall Severity", overall_severity)
                    
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
        else:
            st.success("Full AI Mode Enabled")
            st.write("YOLO model is loaded and ready for real defect detection.")
        
        if HF_API_KEY:
            st.success("Hugging Face API Available")
            st.write("AI-powered analysis is enabled.")
        else:
            st.warning("Hugging Face API Not Available")
            st.write("Using local expert analysis only.")
        
        st.header("📋 Supported Defects")
        st.write("- Cracks")
        st.write("- Spalling")
        st.write("- Discoloration") 
        st.write("- Voids")
        st.write("- Other surface defects")

if __name__ == "__main__":
    main()
