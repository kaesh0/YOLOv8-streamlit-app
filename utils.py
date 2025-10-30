from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

# ------------------ VEHICLE CLASSES ------------------
# YOLOv8 COCO dataset indices for vehicles
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}

# -----------------------------------------------------
def _display_detected_frames(conf, model, st_frame, image):
    """
    Display detected objects on a video frame using the YOLOv8 model.
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    results = model.predict(image, conf=conf)

    boxes = results[0].boxes
    names = model.names  # class labels (dict like {0: 'person', 1: 'bicycle', ...})

    # Filter only vehicles
    vehicle_boxes = []
    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        if label in VEHICLE_CLASSES:
            vehicle_boxes.append(box)

    # Count total vehicles detected
    vehicle_count = len(vehicle_boxes)

    # Plot full results (YOLO draws all, but you can mask non-vehicle if you prefer)
    res_plotted = results[0].plot()
    st_frame.image(
        res_plotted,
        caption=f"Detected Vehicles: {vehicle_count}",
        channels="BGR",
        use_column_width=True
    )

    # Show expanded info if needed
    with st.expander("Detection Results"):
        st.write(f"Total vehicles detected: **{vehicle_count}**")
        for box in vehicle_boxes:
            cls = int(box.cls[0])
            st.write(f"- {names[cls]} (confidence: {float(box.conf[0]):.2f})")


# -----------------------------------------------------
@st.cache_resource
def load_model(model_path):
    """Loads YOLOv8 model from file path."""
    return YOLO(model_path)


# -----------------------------------------------------
def infer_uploaded_image(conf, model):
    """Run detection on uploaded image."""
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)

    if uploaded_file:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting vehicles..."):
                results = model.predict(image, conf=conf)
                boxes = results[0].boxes
                names = model.names

                vehicle_boxes = [b for b in boxes if names[int(b.cls[0])] in VEHICLE_CLASSES]
                count = len(vehicle_boxes)
                plotted = results[0].plot()[:, :, ::-1]

                with col2:
                    st.image(plotted, caption=f"Detected Vehicles: {count}", use_column_width=True)
                    st.success(f"Total Vehicles: {count}")


# -----------------------------------------------------
def infer_uploaded_video(conf, model):
    """Run detection on uploaded video."""
    uploaded_video = st.sidebar.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("Run Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            vid_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            while vid_cap.isOpened():
                success, frame = vid_cap.read()
                if not success:
                    break
                _display_detected_frames(conf, model, st_frame, frame)
            vid_cap.release()


# -----------------------------------------------------
def infer_uploaded_webcam(conf, model):
    """Run detection on webcam feed."""
    st.write("🔴 Press 'Stop' to end webcam feed.")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("No webcam feed detected.")
                break
            _display_detected_frames(conf, model, st_frame, frame)
            run = st.checkbox("Start Webcam", value=True)

        cap.release()
        st.success("Webcam stopped.")
