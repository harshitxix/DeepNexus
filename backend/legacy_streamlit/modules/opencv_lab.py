import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def resize_image(image, max_width=400, max_height=400):
    """Resize image to fit display while maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def create_sample_image():
    """Create a sample image with shapes for demonstration."""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), 2)
    cv2.circle(img, (250, 100), 40, (0, 255, 0), 2)
    cv2.ellipse(img, (200, 250), (60, 30), 0, 0, 360, (255, 0, 0), 2)
    cv2.putText(img, "Sample Image", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img


def convert_to_rgb(image):
    """Convert BGR to RGB for display."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image, kernel_size=5):
    """Apply Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_canny_edge_detection(image, threshold1=50, threshold2=150):
    """Apply Canny edge detection."""
    gray = apply_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)


def apply_sobel_edge_detection(image, axis="both"):
    """Apply Sobel edge detection."""
    gray = apply_grayscale(image)
    
    if axis == "both":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    elif axis == "x":
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5).astype(np.uint8)
    else:
        edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5).astype(np.uint8)
    
    return edges


def apply_threshold(image, threshold_value=127):
    """Apply binary threshold."""
    gray = apply_grayscale(image)
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded


def detect_contours(image):
    """Detect contours in image."""
    gray = apply_grayscale(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result, len(contours)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "opencv_image": None,
        "opencv_operation": "Original",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def opencv_page():
    st.title("OpenCV - Image Processing")
    st.caption(
        "Basic image processing operations: filters, edge detection, thresholding, and contour detection."
    )
    
    _init_state()
    
    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE SOURCE
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Image Source")
    
    img_source = st.radio("Select source", ["Sample Image", "Upload Image"], horizontal=True, key="opencv_source")
    
    image = None
    
    if img_source == "Sample Image":
        if st.button("Load Sample Image", key="opencv_load_sample"):
            st.session_state.opencv_image = create_sample_image()
            st.rerun()
        
        if st.session_state.opencv_image is not None:
            image = st.session_state.opencv_image
    
    else:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"], key="opencv_upload")
        if uploaded is not None:
            image_pil = Image.open(uploaded)
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            st.session_state.opencv_image = image
    
    if image is None:
        st.info("Load a sample image or upload an image to get started.")
        return
    
    # ══════════════════════════════════════════════════════════════════════════
    # OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Image Processing Operations")
    
    operation = st.selectbox(
        "Select operation",
        [
            "Original",
            "Grayscale",
            "Blur",
            "Canny Edge Detection",
            "Sobel Edge Detection",
            "Binary Threshold",
            "Contour Detection",
        ],
        key="opencv_operation_select"
    )
    
    # Apply operation with parameters
    result_image = None
    contour_count = None
    
    if operation == "Original":
        result_image = image.copy()
    
    elif operation == "Grayscale":
        result_gray = apply_grayscale(image)
        result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        st.caption("Converted to single-channel grayscale image.")
    
    elif operation == "Blur":
        kernel_size = st.slider("Kernel size", 1, 30, 5, step=2, key="opencv_blur_kernel")
        result_gray = apply_blur(image, kernel_size)
        result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR) if len(result_gray.shape) == 2 else result_gray
        st.caption(f"Gaussian blur with kernel size {kernel_size}x{kernel_size}.")
    
    elif operation == "Canny Edge Detection":
        c1, c2 = st.columns(2)
        with c1:
            threshold1 = st.slider("Lower threshold", 10, 200, 50, key="opencv_canny_low")
        with c2:
            threshold2 = st.slider("Upper threshold", 50, 500, 150, key="opencv_canny_high")
        
        result_gray = apply_canny_edge_detection(image, threshold1, threshold2)
        result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        st.caption(f"Canny edge detection with thresholds {threshold1} and {threshold2}.")
    
    elif operation == "Sobel Edge Detection":
        axis = st.selectbox("Axis", ["both", "x", "y"], key="opencv_sobel_axis")
        result_gray = apply_sobel_edge_detection(image, axis)
        result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        st.caption(f"Sobel edge detection along {axis} axis.")
    
    elif operation == "Binary Threshold":
        threshold_val = st.slider("Threshold value", 0, 255, 127, key="opencv_threshold_val")
        result_gray = apply_threshold(image, threshold_val)
        result_image = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        st.caption(f"Binary threshold at value {threshold_val}.")
    
    elif operation == "Contour Detection":
        result_image, contour_count = detect_contours(image)
        st.caption(f"Contours detected: {contour_count}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Results")
    
    col_original, col_result = st.columns(2)
    
    with col_original:
        st.caption("Original Image")
        original_resized = resize_image(image)
        st.image(convert_to_rgb(original_resized), use_column_width=True)
    
    with col_result:
        st.caption(f"After {operation}")
        result_resized = resize_image(result_image)
        st.image(convert_to_rgb(result_resized), use_column_width=True)
    
    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE INFO
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Image Information")
    
    col_orig_info, col_result_info = st.columns(2)
    
    with col_orig_info:
        st.caption("Original Image")
        st.write(f"Shape: {image.shape}")
        st.write(f"Data type: {image.dtype}")
        st.write(f"Min value: {image.min()}, Max value: {image.max()}")
    
    with col_result_info:
        st.caption("Result Image")
        st.write(f"Shape: {result_image.shape}")
        st.write(f"Data type: {result_image.dtype}")
        st.write(f"Min value: {result_image.min()}, Max value: {result_image.max()}")
    
    if contour_count is not None:
        st.write(f"Contours detected: {contour_count}")
