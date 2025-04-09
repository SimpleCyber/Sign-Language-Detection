from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Function to classify the sign
def classify_sign(img):
    np.set_printoptions(suppress=True)  # Disable scientific notation

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create input array for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Convert image to RGB and resize
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Streamlit App Configuration
st.set_page_config(layout="wide")

st.title("Sign Language Detection")

# Sidebar: Display sample signs
st.sidebar.header("Sample Images")
st.sidebar.write("Drag and drop images from below for classification.")

# Use columns in the sidebar to align images with spacing
st.sidebar.write("### Different Sign Language")
cols = st.sidebar.columns(4)  # Create 4 columns for images in a row

# Fresh signs
fresh_images = ["images/Alldone.png", "images/camera.png",  "images/help.png", "images/iloveu.png","images/more.png", "images/Please.png", "images/thankyu.png","images/Water.png","images/yes.png"]
fresh_captions = ["All Done", "Camera",  "Help", "I love You","More","Please","Thank You","Water","Yes"]

for idx, img_path in enumerate(fresh_images):
    with cols[idx % 4]:  # Cycle through columns
        st.image(img_path, caption=fresh_captions[idx], use_column_width=True)


# Image Upload
input_img = st.file_uploader("Upload or Drag & Drop an image of a Sign", type=["jpg", "png", "jpeg"])

if input_img is not None:
    if st.button("Classify"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Your Uploaded Image")
            st.image(input_img, use_column_width=False, width=200)

        with col2:
            st.info("Classification Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_sign(image_file)

            # Convert confidence score to percentage
            accuracy = confidence_score * 100
            accuracy_text = f"Accuracy : {accuracy:.2f}%"

            # Extract sign name
            parts = label.strip().split(" ")
            sign_name = parts[1] if len(parts) > 1 else label.strip()

            # Determine background color
            if accuracy > 75:
                st.success(f"The Sign is: **{sign_name}**")
                st.success(accuracy_text)
            elif accuracy > 50:
                st.warning(f"The Sign is: **{sign_name}**")
                st.warning(accuracy_text)
            elif accuracy > 25:
                st.info(f"The Sign is: **{sign_name}**")
                st.info(accuracy_text)
            else:
                st.error("❌ The image could not be classified into any relevant category.")
                st.error(accuracy_text)

