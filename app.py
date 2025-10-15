import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Space Station Safety", layout="wide")
st.title("🛸 Space Station Safety Object Detection")
st.markdown("**GenIgnite 2025** | Priya Gupta | mAP@0.5: **73.85%** ✅")

with st.sidebar:
    st.success("**Model Performance**\n- mAP@0.5: 73.85%\n- Target: 70% ✅")
    conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)

@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

tab1, tab2 = st.tabs(["📸 Detection", "📊 Performance"])

with tab1:
    st.header("Upload Image")
    uploaded = st.file_uploader("Choose image", type=['jpg','png'])
    
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded)
            st.image(image, caption="Original")
        with col2:
            img_array = np.array(image)
            results = model(img_array, conf=conf, verbose=False)[0]
            result_img = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            st.image(result_img, caption="Detected")
            
            if results.boxes and len(results.boxes) > 0:
                st.success(f"✅ Found {len(results.boxes)} objects!")
                data = []
                for i, (cls, conf_val) in enumerate(zip(
                    results.boxes.cls.cpu().numpy(),
                    results.boxes.conf.cpu().numpy()
                ), 1):
                    data.append({'ID': i, 'Object': model.names[int(cls)], 'Confidence': f'{conf_val:.2%}'})
                df = pd.DataFrame(data)
                st.dataframe(df, hide_index=True)
            else:
                st.warning("No objects detected")

with tab2:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("mAP@0.5", "73.85%", "+3.85%")
    col2.metric("Precision", "90.73%")
    col3.metric("Recall", "65.07%")
