import os
import time
import streamlit as st
from PIL import Image
from main import generate_summary
from dotenv import load_dotenv

st.title("🍽️Cook My Image")
st.secrets["Google_API_KEY"] = os.getenv("GOOGLE_API_KEY")
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

if st.button("Reset Session",help="Reset this conversation.",type="primary"):
    st.session_state.clear()
    st.rerun()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
UPLOAD_DIR_PERMANENT = "uploaded_images"
os.makedirs(UPLOAD_DIR_PERMANENT, exist_ok=True)
if uploaded_image:
    image = Image.open(uploaded_image)
    saved_image_path = os.path.join(UPLOAD_DIR_PERMANENT, uploaded_image.name)
    image_bytes = uploaded_image.read()
    try:
        with open(saved_image_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        st.error(f"An error occurred while saving the image: {e}")
        st.write("Please ensure the uploaded file is a valid image.")

    st.image(image, caption=uploaded_image.name)
    noisy_title =  st.text_input("Outcast's Name")
    if noisy_title:
        response = generate_summary(test_image_path=saved_image_path,
                                    test_title=noisy_title,
                                    json_path="train/train.json",
                                    img_dir="train",
                                    API_KEY=os.getenv("GOOGLE_API_KEY"))
        st.write_stream(stream_data(response))

if __name__ == "__main__":
    load_dotenv()