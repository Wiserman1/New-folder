import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None

st.set_page_config(page_title="UniUyo EE15 Structural Integrity App", layout="centered")
st.title("🧱 Automated Concrete Bridge Deck Inspection System")
st.caption("GET 324: Artificial Intelligence, Machine Learning and Convergent Technologies")
st.write("**Group Assignment Task EE15:** Concrete Bridge Deck Crack Detection")

CLASS_0 = "Non-Cracked (Structurally Sound)"
CLASS_1 = "Cracked (Anomalies Detected / Maintenance Required)"

st.sidebar.header("Prediction mode")
st.sidebar.write("Choose the lighter option for easier Streamlit deployment.")
mode = st.sidebar.radio(
    "Mode",
    ["Lightweight demo mode", "TensorFlow model if available"],
    index=0,
)


def classify_with_lightweight_features(image: Image.Image) -> float:
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32) / 255.0

    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    edge_strength = np.sqrt(gx * gx + gy * gy)

    edge_density = float(np.mean(edge_strength))
    contrast = float(np.std(arr))
    score = float(np.clip(0.25 + 0.6 * edge_density + 0.25 * contrast, 0.0, 1.0))
    return score


@st.cache_resource
def load_tensorflow_model():
    if tf is None:
        raise RuntimeError("TensorFlow is not installed")
    if not os.path.exists("model.h5"):
        raise FileNotFoundError("model.h5 was not found in the repository")
    return tf.keras.models.load_model("model.h5")


if mode == "TensorFlow model if available":
    if tf is not None and os.path.exists("model.h5"):
        try:
            model = load_tensorflow_model()
            st.success("🤖 TensorFlow model loaded successfully.")
            use_tensorflow = True
        except Exception as exc:
            st.warning(f"Falling back to lightweight mode: {exc}")
            use_tensorflow = False
    else:
        st.info("TensorFlow model was not found or TensorFlow is unavailable. Using the lightweight demo mode instead.")
        use_tensorflow = False
else:
    use_tensorflow = False

st.caption("How to use: click the upload button below, choose a JPG or PNG image, and the app will show the result.")
uploaded_file = st.file_uploader("Upload concrete surface photograph...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Submitted Surface Specimen", use_container_width=True)
    st.write("🔍 Analyzing the uploaded image...")

    if use_tensorflow:
        img_resized = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch, verbose=0)
        prediction_value = float(prediction[0][0])
        st.info("Using the TensorFlow model loaded from model.h5.")
    else:
        prediction_value = classify_with_lightweight_features(image)
        st.info("Using the lightweight demo classifier for easy deployment.")

    st.divider()

    if prediction_value >= 0.5:
        confidence = prediction_value * 100
        st.error(f"🚨 **Structural Inspection Status: {CLASS_1}**")
        st.metric(label="Crack Probability Index", value=f"{confidence:.2f}%")
    else:
        confidence = (1 - prediction_value) * 100
        st.success(f"🍏 **Structural Inspection Status: {CLASS_0}**")
        st.metric(label="Structural Reliability Margin", value=f"{confidence:.2f}%")

st.divider()
st.info("📌 **Submission Specifications:** Faculty of Engineering, Department of Computer Engineering, University of Uyo.")
