from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


st.set_page_config(
    page_title="Bahraini Currency Recognition",
    layout="centered",
)

st.title("Bahraini Currency Recognition")
st.write("Upload an image or take a picture to detect Bahraini banknotes and coins.")

MODEL_PATH = Path(__file__).parent / "best_2.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error("Model file not found. Please place your trained best_2.pt next to app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

method = st.radio("Choose an option:", ["Upload Image", "Take Picture"], horizontal=True)

if method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a currency image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a currency picture")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This image could not be opened. Please choose another picture.")
        st.stop()

    st.image(image, caption="Your image", use_container_width=True)

    if st.button("Detect Currency", type="primary", use_container_width=True):
        with st.spinner("Detecting currency..."):
            result = model.predict(source=image, conf=0.25, verbose=False)[0]

        st.subheader("Detection Result")
        annotated_image = result.plot()[..., ::-1]  # Convert YOLO's BGR output to RGB
        st.image(annotated_image, use_container_width=True)

        if result.boxes is None or len(result.boxes) == 0:
            st.warning("No currency detected. Try another image with clearer lighting or a closer view.")

