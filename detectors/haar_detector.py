# ================================================
# Haar Cascade Human Detector
# Author: Chintha Dhanunjaya
# ================================================

import cv2

def get_haar_detector():
    """Initialize and return Haar Cascade full body detector."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    )
    return cascade

def detect_with_haar(gray_frame, cascade):
    """
    Detect humans in grayscale frame using Haar Cascade.
    Returns bounding boxes.
    """
    bodies = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )
    return bodies