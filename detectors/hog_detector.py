# ================================================
# HOG-based Human Detector
# Author: Chintha Dhanunjaya
# ================================================

import cv2

def get_hog_detector():
    """Initialize and return HOG people detector."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_with_hog(frame, hog):
    """
    Detect humans in frame using HOG descriptor.
    Returns bounding boxes and confidence weights.
    """
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    return boxes, weights