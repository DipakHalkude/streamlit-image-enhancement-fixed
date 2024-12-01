import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Functions for Image Enhancements
def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

def smoothen_image(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_mask(image):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (image.shape[1]//2, image.shape[0]//2), 100, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

# Streamlit UI
st.title("Image Enhancement Tool")
st.write("Upload an image and apply enhancements.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Enhancements
    st.sidebar.title("Enhancement Options")
    contrast = st.sidebar.slider("Adjust Contrast", 1.0, 3.0, 1.5)
    brightness = st.sidebar.slider("Adjust Brightness", -100, 100, 30)
    smoothen = st.sidebar.slider("Smoothen (Blur)", 1, 15, 5, step=2)
    apply_sharpen = st.sidebar.checkbox("Sharpen Image")
    apply_mask_option = st.sidebar.checkbox("Apply Mask")

    # Apply transformations
    enhanced_image = adjust_contrast(image_cv, contrast)
    enhanced_image = adjust_brightness(enhanced_image, brightness)
    enhanced_image = smoothen_image(enhanced_image, kernel_size=smoothen)
    if apply_sharpen:
        enhanced_image = sharpen_image(enhanced_image)
    if apply_mask_option:
        enhanced_image = apply_mask(enhanced_image)

    # Convert back to RGB for display
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    # Display Enhanced Image
    st.image(enhanced_image_rgb, caption="Enhanced Image", use_column_width=True)
