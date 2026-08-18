import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ============ CONFIGURATION ============
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

# 8 Classes (alphabetical order from training)
CLASS_NAMES = [
    'acne',
    'basal_cell_carcinoma',
    'benign_keratosis',
    'eczema',
    'fungal_infection',
    'melanocytic_nevi',
    'melanoma',
    'normal_skin'
]

CLASS_INFO = {
    'acne': {
        'icon': '🔴',
        'description': 'Common skin condition causing pimples and blackheads.',
        'severity': 'Mild',
        'action': 'Usually treatable with topical medications. Consult dermatologist if severe.',
        'color': 'orange'
    },
    'basal_cell_carcinoma': {
        'icon': '⚠️',
        'description': 'Most common type of skin cancer. Grows slowly.',
        'severity': 'Serious',
        'action': '🚨 Requires medical evaluation. Highly treatable when detected early.',
        'color': 'red'
    },
    'benign_keratosis': {
        'icon': '✅',
        'description': 'Non-cancerous skin growth. Common with aging.',
        'severity': 'Benign',
        'action': 'Usually harmless. Consult doctor if changes occur.',
        'color': 'green'
    },
    'eczema': {
        'icon': '🟡',
        'description': 'Inflammatory skin condition causing itchy, red patches.',
        'severity': 'Mild-Moderate',
        'action': 'Manageable with moisturizers and prescribed treatments.',
        'color': 'orange'
    },
    'fungal_infection': {
        'icon': '🟠',
        'description': 'Skin infection caused by fungi (ringworm, athletes foot, etc.)',
        'severity': 'Mild',
        'action': 'Treatable with antifungal medications. Consult doctor.',
        'color': 'orange'
    },
    'melanocytic_nevi': {
        'icon': '✅',
        'description': 'Common mole. Usually harmless.',
        'severity': 'Benign',
        'action': 'Monitor for changes in size, color, or shape.',
        'color': 'green'
    },
    'melanoma': {
        'icon': '🚨',
        'description': 'Serious form of skin cancer. Can spread rapidly.',
        'severity': 'CRITICAL',
        'action': '🚨 URGENT: Consult dermatologist immediately! Early detection saves lives.',
        'color': 'red'
    },
    'normal_skin': {
        'icon': '✅',
        'description': 'Healthy skin with no visible abnormalities.',
        'severity': 'Healthy',
        'action': 'Maintain good skincare routine. Regular self-examination recommended.',
        'color': 'green'
    }
}

IMG_SIZE = 224

# ============ MODEL LOADING ============
@st.cache_resource
def load_skin_model():
    model = load_model('models/efficientnetb0_8class_best.h5')
    return model

# ============ GRAD-CAM ============
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_overlay(img_pil, heatmap, alpha=0.5):
    img = np.array(img_pil)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return heatmap_colored, overlay

# ============ APP UI ============

# Header
st.title("🔬 Advanced Skin Lesion Classification System")
st.markdown("### AI-Powered Analysis for 8 Skin Conditions")

# Model info banner
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Model", "EfficientNetB0")
with col_info2:
    st.metric("Test Accuracy", "81.75%")
with col_info3:
    st.metric("Classes", "8")
with col_info4:
    st.metric("Training Images", "14,711")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🏥 About")
    st.write("""
    Advanced deep learning system trained on **14,711 dermatoscopic images** 
    to classify skin conditions into 8 categories including cancer detection.
    """)
    
    st.header("📊 Classes Detected")
    for cls in CLASS_NAMES:
        info = CLASS_INFO[cls]
        st.markdown(f"**{info['icon']} {cls}**")
        st.caption(f"{info['description']}")
    
    st.markdown("---")
    st.header("⚠️ Medical Disclaimer")
    st.warning("""
    This tool is for **educational purposes only**.  
    Not a replacement for professional medical diagnosis.  
    Always consult a qualified dermatologist for medical concerns.
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Skin Image")
    uploaded_file = st.file_uploader(
        "Upload a clear skin image",
        type=['jpg', 'jpeg', 'png'],
        help="Best results with well-lit, close-up dermatoscopic images"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("🎯 Analysis Results")
    
    if uploaded_file:
        with st.spinner("🧠 AI analyzing image..."):
            model = load_skin_model()
            
            # Preprocess
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            info = CLASS_INFO[predicted_class]
            
            # Confidence-based display
            if confidence >= 80:
                st.success(f"✅ **High Confidence Prediction**")
            elif confidence >= 60:
                st.info(f"ℹ️ **Moderate Confidence**")
            else:
                st.warning(f"⚠️ **Low Confidence** - Consider consulting a doctor")
            
            # Main prediction
            st.markdown(f"### {info['icon']} Predicted: **{predicted_class.replace('_', ' ').title()}**")
            
            # Confidence meter
            st.metric("Confidence Level", f"{confidence:.2f}%")
            st.progress(float(confidence/100))
            
            # Severity badge
            if info['color'] == 'red':
                st.error(f"**Severity: {info['severity']}**")
            elif info['color'] == 'orange':
                st.warning(f"**Severity: {info['severity']}**")
            else:
                st.success(f"**Severity: {info['severity']}**")
            
            # Description
            st.info(f"**Description:** {info['description']}")
            
            # Action
            st.markdown(f"**Recommended Action:** {info['action']}")
    else:
        st.info("👈 Upload an image to get AI analysis")
        st.markdown("### 💡 Tips for Best Results:")
        st.markdown("""
        - Use clear, well-lit images
        - Close-up shots work best
        - Dermatoscopic images give highest accuracy
        - Avoid blurry or dark photos
        """)

# All Probabilities
if uploaded_file:
    st.markdown("---")
    st.header("📊 Detailed Probability Analysis")
    
    # Sort by probability
    prob_data = [(CLASS_NAMES[i], predictions[i]*100, CLASS_INFO[CLASS_NAMES[i]]['icon']) 
                 for i in range(len(CLASS_NAMES))]
    prob_data.sort(key=lambda x: -x[1])
    
    for cls, prob, icon in prob_data:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.markdown(f"**{icon} {cls.replace('_', ' ').title()}**")
        with col_b:
            st.progress(float(prob/100))
            st.caption(f"{prob:.2f}%")

# Grad-CAM Visualization
if uploaded_file:
    st.markdown("---")
    st.header("🔥 Grad-CAM Visualization")
    st.markdown("**See where the AI is focusing to make its decision:**")
    
    with st.spinner("Generating heatmap..."):
        try:
            heatmap = make_gradcam_heatmap(img_array, model)
            heat_colored, overlay = create_overlay(img_resized, heatmap)
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.image(img_resized, caption="Original", use_container_width=True)
            with col4:
                st.image(heat_colored, caption="Heatmap (Red = Focus)", use_container_width=True)
            with col5:
                st.image(overlay, caption="Overlay Analysis", use_container_width=True)
            
            st.info("🔴 **Red areas** show where the model focused most. 🔵 **Blue areas** were less important for the decision.")
        except Exception as e:
            st.warning(f"Grad-CAM generation failed: {str(e)}")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**🎓 Project**")
    st.caption("Multi-Class Image Classifier")
with col_f2:
    st.markdown("**🔬 Technology**")
    st.caption("EfficientNetB0 + TensorFlow")
with col_f3:
    st.markdown("**📊 Performance**")
    st.caption("81.75% Test Accuracy on 8 Classes")

st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This AI tool is for educational purposes only. Always consult a qualified dermatologist for medical diagnosis.")