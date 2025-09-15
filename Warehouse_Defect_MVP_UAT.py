import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1"

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
# 2. API Key Testing Function
# ----------------------------
def test_api_key(api_key):
    """
    Test if the Hugging Face API key is valid using a reliable model
    """
    if not api_key:
        return False, "No API key provided"
    
    # Use a reliable model for testing
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": "[MASK] is a great framework for machine learning.",
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "API key is valid!"
        elif response.status_code == 401:
            return False, "Invalid API key: Unauthorized (401)"
        elif response.status_code == 403:
            return False, "API key doesn't have access to this resource (403)"
        elif response.status_code == 404:
            return False, "Model not found (404). Try a different model."
        else:
            return False, f"API returned status code: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. The API might be busy."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Check your internet connection."
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"

# ----------------------------
# 3. Hugging Face LLM Integration with reliable model
# ----------------------------
def get_llm_commentary(defects_info, api_key):
    """
    Get AI commentary on detected defects using Hugging Face's API
    Uses a more reliable model for consistent performance
    """
    if not api_key:
        return "API key is required for AI commentary. Please enter your Hugging Face API key."
    
    # Prepare the prompt
    prompt = f"""
    As a structural engineering expert, analyze these concrete defects detected in a warehouse:
    {defects_info}
    
    Please provide:
    1. A brief assessment of the severity
    2. Potential causes
    3. Recommended actions
    4. Safety implications
    
    Keep the response concise and professional (under 200 words).
    """
    
    # Try multiple endpoints - some might work better than others
    endpoints = [
        {
            "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            "params": {"max_new_tokens": 300, "temperature": 0.3}
        },
        {
            "url": "https://api-inference.huggingface.co/models/google/flan-t5-base",
            "params": {"max_new_tokens": 300, "temperature": 0.3}
        },
        {
            "url": "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            "params": {"max_new_tokens": 300, "temperature": 0.3}
        }
    ]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for endpoint in endpoints:
        try:
            with st.spinner(f"Trying endpoint: {endpoint['url'].split('/')[-1]}..."):
                payload = {
                    "inputs": prompt,
                    "parameters": endpoint["params"]
                }
                
                response = requests.post(endpoint["url"], headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        if generated_text:
                            return generated_text
                    elif isinstance(result, dict):
                        generated_text = result.get('generated_text', '')
                        if generated_text:
                            return generated_text
                elif response.status_code == 503:
                    # Model is loading, try the next one
                    continue
                    
        except Exception as e:
            # Try the next endpoint if this one fails
            continue
    
    return "All AI endpoints are currently unavailable. Using local expert analysis instead."

# ----------------------------
# 4. Local Expert System (Fallback)
# ----------------------------
def get_local_expert_commentary(defects):
    """
    Generate expert commentary using a local rule-based system
    """
    if not defects:
        return "No defects detected. The concrete surface appears to be in good condition."
    
    # Count defects by type
    defect_counts = {}
    for defect in defects:
        defect_type = defect['type']
        defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
    
    # Generate commentary
    commentary = "## Expert Analysis\n\n"
    commentary += "**Defects Detected**:\n"
    
    for defect_type, count in defect_counts.items():
        commentary += f"- {count} {defect_type}(s)\n"
    
    # Calculate average confidence
    avg_confidence = sum(d['confidence'] for d in defects) / len(defects)
    commentary += f"\n**Average Detection Confidence**: {avg_confidence:.2f}\n\n"
    
    commentary += "**General Recommendations**:\n"
    
    if any('crack' in d['type'].lower() for d in defects):
        commentary += "- Cracks should be monitored for width progression over time\n"
        commentary += "- Cracks wider than 0.3mm may require professional assessment\n"
        commentary += "- Consider epoxy injection for active cracks\n"
    
    if any('spall' in d['type'].lower() for d in defects):
        commentary += "- Spalling indicates concrete deterioration that may expose reinforcement\n"
        commentary += "- Affected areas should be repaired to prevent further damage\n"
        commentary += "- Clean and treat exposed reinforcement before patching\n"
    
    if any('hole' in d['type'].lower() or 'void' in d['type'].lower() for d in defects):
        commentary += "- Voids and holes should be filled with appropriate repair materials\n"
        commentary += "- Assess whether voids affect structural integrity\n"
    
    if any('stain' in d['type'].lower() or 'discolor' in d['type'].lower() for d in defects):
        commentary += "- Stains may indicate water infiltration or chemical exposure\n"
        commentary += "- Identify and address the source of staining\n"
    
    commentary += "\n**Safety Note**: For a comprehensive assessment, consult a structural engineer."
    
    return commentary

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

# API Key Section
st.sidebar.header("API Key Configuration")
api_key = st.sidebar.text_input("Hugging Face API Key:", type="password", 
                               help="Get your API key from huggingface.co → Settings → Access Tokens")

if api_key:
    if st.sidebar.button("Test API Key"):
        is_valid, message = test_api_key(api_key)
        if is_valid:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
else:
    st.sidebar.info("Enter your Hugging Face API key to enable AI commentary")

# Debug information
with st.sidebar.expander("Debug Info"):
    st.write("If API is not working, possible reasons:")
    st.write("- Model may be loading (try again in 30 seconds)")
    st.write("- API rate limits may be exceeded")
    st.write("- Temporary service outage")
    st.write("The app will automatically use local expert analysis if API fails.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    with st.spinner("Analyzing image for defects..."):
        results = model(image)
    
    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    
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
                "location": [float(coord) for coord in box.xywh[0]]
            })
    
    # Display defect information
    if defects:
        st.subheader("📊 Detection Results")
        
        # Show defects in a table
        for i, defect in enumerate(defects, 1):
            st.write(f"{i}. **{defect['type']}** ({(defect['confidence']*100):.1f}% confidence)")
        
        # Prepare defect information for LLM
        defects_info = "\n".join([
            f"- {d['type']} (confidence: {d['confidence']:.2f})"
            for d in defects
        ])
        
        # Get and display commentary
        st.subheader("🧠 Expert Analysis")
        
        if api_key:
            # Try AI first, fallback to local if it fails
            commentary = get_llm_commentary(defects_info, api_key)
            if "unavailable" in commentary.lower():
                st.info("AI service temporarily unavailable. Using local expert analysis.")
                commentary = get_local_expert_commentary(defects)
            st.write(commentary)
        else:
            commentary = get_local_expert_commentary(defects)
            st.write(commentary)
            st.info("To enable AI-powered analysis, add your Hugging Face API key in the sidebar")
        
    else:
        st.success("✅ No defects detected! The concrete surface appears to be in good condition.")

# Footer
st.markdown("---")
st.markdown("""
**About this app**:
- Defect detection powered by YOLO model
- AI commentary requires a Hugging Face API key
- Local expert system provides fallback analysis when AI is unavailable
- Always consult a qualified engineer for critical structural assessments
""")
