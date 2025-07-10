import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


model = load_model("model_gusture.keras")


class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'Nothing', 'Space', 'Delete'
]


st.set_page_config(page_title="Gesture Prediction", layout="centered")
st.title("🖐️ Sign Language Gesture Prediction")
st.write("Upload an image of a hand sign and get the predicted gesture!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    predictions = model.predict(img_array)[0]
    best_idx = np.argmax(predictions)
    predicted_class = class_names[best_idx]
    confidence = predictions[best_idx] * 100

    st.success(f"### Prediction: {predicted_class}")
    st.info(f"### Confidence: {confidence:.2f}%")


    with st.expander("🔍 Show all class confidences"):
        for i, prob in enumerate(predictions):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
