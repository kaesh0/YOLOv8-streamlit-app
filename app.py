#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Aayush Yadav
   @Date:          2025/10/30
   @Description:   Real-Time Vehicle Detection using YOLOv8 and Streamlit
-------------------------------------------------
"""
from pathlib import Path
import streamlit as st
from PIL import Image

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# -------------------------------------------------
# 🔧 PAGE CONFIG (must be the first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="🚗 Real-Time Vehicle Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 🏠 MAIN TITLE + DESCRIPTION
# -------------------------------------------------
st.title("🚗 Real-Time Vehicle Detection using YOLOv8")
st.markdown(
    """
    Detects **vehicles** and **pedestrians** from live video or webcam feed in real time.  
    Built using **Ultralytics YOLOv8** and **Streamlit**.
    """
)

# -------------------------------------------------
# 🧠 SIDEBAR: MODEL CONFIGURATION
# -------------------------------------------------
st.sidebar.header("Model Configuration")

task_type = st.sidebar.selectbox("Select Task", ["Detection"])

if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select YOLO Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented.")
    st.stop()

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) / 100

if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please select a model in the sidebar.")
    st.stop()

# -------------------------------------------------
# 🧩 LOAD YOLO MODEL
# -------------------------------------------------
try:
    model = load_model(model_path)
    st.sidebar.success(f"✅ Model loaded successfully: {model_type}")
except Exception as e:
    st.sidebar.error(f"❌ Unable to load model. Check path: {model_path}")
    st.stop()

# -------------------------------------------------
# 🎥 IMAGE / VIDEO / WEBCAM OPTIONS
# -------------------------------------------------
st.sidebar.header("Input Source")
source_selectbox = st.sidebar.selectbox(
    "Select Source Type",
    config.SOURCES_LIST
)

if source_selectbox == "Image":
    infer_uploaded_image(confidence, model)

elif source_selectbox == "Video":
    infer_uploaded_video(confidence, model)

elif source_selectbox == "Webcam":
    infer_uploaded_webcam(confidence, model)

else:
    st.error("Please select a valid input source.")
