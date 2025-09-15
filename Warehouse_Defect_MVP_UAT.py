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
        # Remove existing file to allow overwrite
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
# 2. Hugging Face LLM Integration with OpenAI-compatible API
# ----------------------------
def get_llm_commentary(defects, inspection_context):
    """
    Get professional engineering analysis with pre/post tenancy comparison
    """
    # Get API key from Streamlit secrets
    try:
        # Try different possible secret names
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            api_key = st.secrets['HUGGINGFACEHUB_API_TOKEN']
        elif 'HUGGINGFACE_API_KEY' in st.secrets:
            api_key = st.secrets['HUGGINGFACE_API_KEY']
        elif 'HF_TOKEN' in st.secrets:
            api_key = st.secrets['HF_TOKEN']
        else:
            st.error("Hugging Face API token not found in secrets. Please check your secrets configuration.")
            return "API token configuration error."
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return "Secrets access error."
    
    try:
        # Initialize OpenAI client with Hugging Face endpoint
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        # Prepare detailed defect information
        defects_info = "DETECTED DEFECT TYPES AND QUANTITIES:\n"
        defect_counts = {}
        for defect in defects:
            defect_type = defect['type']
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        for defect_type, count in defect_counts.items():
            defects_info += f"- {defect_type}: {count} instance(s)\n"
        
        defects_info += "\nINDIVIDUAL DEFECT DETAILS:\n"
        for i, defect in enumerate(defects, 1):
            defects_info += f"{i}. {defect['type']} (confidence: {defect['confidence']:.2f})\n"
        
        # Professional engineering prompt with tenancy comparison
        prompt = f"""
        As a licensed structural engineer and building surveyor, provide a comprehensive analysis of these concrete defects with specific focus on tenancy context:
        
        INSPECTION CONTEXT: {inspection_context}
        
        DEFECTS IDENTIFIED:
        {defects_info}
        
        Please provide a detailed analysis including:
        
        1. DEFECT-SPECIFIC ANALYSIS:
           - Precise definition and description of each defect type
           - Engineering significance and severity rating
           - Root cause analysis for each defect type
        
        2. TENANCY TIMELINE ASSESSMENT:
           - **Pre-Tenancy vs During Tenancy Comparison**: 
             * Which defects are likely pre-existing vs tenant-induced?
             * Typical defect progression timelines for each defect type
             * Expected vs accelerated deterioration rates
        
        3. LIABILITY ASSESSMENT:
           - **Landlord vs Tenant Responsibility**: 
             * Defects typically considered landlord responsibility
             * Defects that may be tenant-induced or exacerbated
             * Maintenance obligation boundaries
        
        4. LEGAL AND INSURANCE IMPLICATIONS:
           - Documentation requirements for tenancy disputes
           - Insurance claim considerations
           - Dilapidation schedule implications
        
        5. DEFECT-SPECIFIC MITIGATION WITH TENANCY CONTEXT:
           - Urgency of repairs based on tenancy status
           - Tenant safety considerations
           - Repair scheduling around tenancy arrangements
        
        Please structure your response with clear sections for each defect type and specific tenancy context analysis.
        Reference relevant building codes, tenancy laws, and maintenance standards.
        """
        
        with st.spinner("Conducting tenancy context analysis..."):
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior structural engineer and building surveyor with expertise in tenancy defect analysis, liability assessment, and building maintenance. You provide detailed analysis comparing pre-tenancy vs during-tenancy conditions, clearly distinguishing landlord vs tenant responsibilities, and offering practical advice for tenancy context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1200,
                temperature=0.2,
                top_p=0.8,
                stream=False
            )
            
            return response.choices[0].message.content
                
    except Exception as e:
        return f"Unable to generate engineering analysis: {str(e)}"

# ----------------------------
# 3. Streamlit UI with Tenancy Context
# ----------------------------
st.title("🏗️ Warehouse Concrete Structural Assessment")
st.write("Upload an image for detailed defect analysis with pre-tenancy vs during-tenancy comparison.")

# Tenancy context selection
st.subheader("📋 Inspection Context")
inspection_context = st.radio(
    "Select inspection context:",
    ["Pre-Tenancy (Before Move-In)", "During Tenancy (Occupied)", "Post-Tenancy (Move-Out)"],
    help="This helps determine liability and appropriate repair strategies"
)

# Additional tenancy information
if inspection_context != "Pre-Tenancy (Before Move-In)":
    tenancy_duration = st.slider("Tenancy Duration (months):", 1, 120, 12)
    building_usage = st.selectbox(
        "Building Usage:",
        ["General Storage", "Light Manufacturing", "Heavy Machinery", "Cold Storage", "Distribution Center", "Other"]
    )
else:
    tenancy_duration = 0
    building_usage = "Not Applicable"

# Check if we have the API key set up
try:
    has_api_key = any(key in st.secrets for key in ['HUGGINGFACEHUB_API_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_TOKEN'])
    if not has_api_key:
        st.warning("Hugging Face API token not found in secrets. Professional engineering analysis may not be available.")
    else:
        st.success("Hugging Face API key authenticated. Ready for tenancy context analysis.")
except:
    st.warning("Unable to verify API configuration. Some features may be limited.")

uploaded_file = st.file_uploader("Choose structural inspection image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Inspection Image - {inspection_context}", use_container_width=True)

    # Run detection
    with st.spinner("Analyzing concrete defects..."):
        results = model(image)
    
    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Identified Concrete Defects", use_container_width=True)
    
    # Extract defect information
    defects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "location": [float(coord) for coord in box.xywh[0]]  # x_center, y_center, width, height
            })
    
    # Display defect information
    if defects:
        st.subheader("📊 Concrete Defect Inventory")
        
        # Group defects by type for better presentation
        defect_counts = {}
        for defect in defects:
            defect_type = defect['type']
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        st.write("**Defect Type Summary:**")
        for defect_type, count in defect_counts.items():
            st.write(f"• **{defect_type}**: {count} instance(s) detected")
        
        # Show individual defects in a detailed table
        st.write("**Individual Defect Details:**")
        for i, defect in enumerate(defects, 1):
            st.write(f"{i}. **{defect['type']}** - Detection confidence: {defect['confidence']*100:.1f}%")
        
        # Prepare inspection context for analysis
        context_info = f"""
        Inspection Type: {inspection_context}
        Tenancy Duration: {tenancy_duration} months
        Building Usage: {building_usage}
        Inspection Date: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        # Get and display professional engineering analysis
        st.subheader("🧠 Comprehensive Defect Analysis with Tenancy Context")
        st.info(f"**Analysis includes:** Defect analysis + Tenancy timeline assessment + Liability determination + {inspection_context} recommendations")
        
        with st.spinner("Generating tenancy context analysis..."):
            analysis = get_llm_commentary(defects, context_info)
        
        st.write(analysis)
        
        # Add tenancy-specific references
        with st.expander("📚 Tenancy Defect Liability Guidelines"):
            st.write("""
            **Typical Liability Classifications:**
            
            **LANDLORD RESPONSIBILITY (Usually):**
            - Structural defects pre-dating tenancy
            - Foundation settlement issues
            - Roof leaks and water penetration
            - Pre-existing corrosion or decay
            - Building code compliance issues
            
            **TENANT RESPONSIBILITY (Usually):**
            - Damage from improper use or overload
            - Lack of routine maintenance
            - Accident-related damage
            - Modifications without approval
            - Neglect leading to deterioration
            
            **SHARED RESPONSIBILITY (Case-by-case):**
            - Wear and tear vs actual damage
            - Pre-existing conditions exacerbated by use
            - Maintenance issues that weren't reported
            - Environmental factors affecting both parties
            
            **Documentation Requirements:**
            - Pre-tenancy inspection reports with photos
            - Regular maintenance records
            - Tenant reporting timelines
            - Professional assessment documentation
            """)
        
        # Additional tenancy-specific recommendations
        with st.expander("💼 Practical Tenancy Advice"):
            st.write("""
            **For Pre-Tenancy Inspections:**
            - Document all existing defects thoroughly with photos
            - Create detailed dilapidation schedule
            - Establish baseline condition for future reference
            - Consider professional building survey
            
            **For During-Tenancy Issues:**
            - Report defects to landlord promptly
            - Document communication timelines
            - Maintain records of any repairs undertaken
            - Consider independent assessment for disputes
            
            **For Post-Tenancy Assessments:**
            - Compare with pre-tenancy documentation
            - Assess fair wear and tear vs actual damage
            - Consider depreciation for aged defects
            - Seek professional mediation for disputes
            """)
        
    else:
        st.success("✅ No structural defects detected! The concrete elements appear to be in sound condition.")
        
        # Even with no defects, provide tenancy context advice
        if inspection_context == "Pre-Tenancy (Before Move-In)":
            st.info("**Pre-Tenancy Recommendation:** Document this sound condition with timestamped photos for future reference.")
        elif inspection_context == "During Tenancy (Occupied)":
            st.info("**During Tenancy Status:** No tenant-induced defects detected. Continue regular maintenance schedule.")
        else:
            st.info("**Post-Tenancy Status:** No additional defects beyond normal wear and tear detected.")

# Add professional footer with tenancy context
st.markdown("---")
st.markdown("""
**Disclaimer**: 
- This analysis provides preliminary assessment for informational purposes only
- Final liability determinations should be made by qualified building surveyors
- Local tenancy laws and lease agreements take precedence over general guidelines
- Always consult legal professionals for tenancy dispute resolution
""")

# Add engineering certification note
st.caption("_Analysis generated for informational purposes. Final determinations should be made by qualified building surveyors and legal professionals._")
