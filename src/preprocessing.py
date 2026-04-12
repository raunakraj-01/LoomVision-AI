import cv2

def apply_preprocessing(frame):
    """
    Standard preprocessing pipeline for fabric frames.
    Improves contrast and reduces noise so the detection logic works reliably.
    
    Steps:
    1. Convert to Grayscale.
    2. Apply Gaussian Blur to smooth out noise without destroying edges.
    3. Apply Histogram Equalization to standardize brightness/contrast.
    """
    if frame is None:
        return None
        
    # Convert to standard Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Smooth the image to remove camera noise (5x5 kernel)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Standardize lighting/contrast
    equalized = cv2.equalizeHist(blurred)
    
    return equalized
