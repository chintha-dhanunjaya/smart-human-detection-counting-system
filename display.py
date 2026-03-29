# ================================================
# Display & Overlay Module
# Author: Chintha Dhanunjaya
# ================================================

import cv2
from datetime import datetime

def draw_hog_boxes(frame, boxes, weights):
    """Draw green bounding boxes for HOG detections with confidence."""
    for i, (x, y, w, h) in enumerate(boxes):
        conf = min(100, int(abs(weights[i]) * 50)) if i < len(weights) else 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 210, 0), 2)
        cv2.putText(
            frame, f"Human {conf}%",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 2
        )

def draw_haar_boxes(frame, bodies):
    """Draw blue bounding boxes for Haar Cascade detections."""
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 2)
        cv2.putText(
            frame, "Human (Haar)",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2
        )

def draw_info_panel(frame, hog_count, haar_count, fps):
    """Draw semi-transparent info panel on top-left of frame."""
    total = hog_count + haar_count

    # Semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Total Humans : {total}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"HOG Detected : {hog_count}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 210, 0), 2)
    cv2.putText(frame, f"Haar Detected: {haar_count}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 120, 0), 2)
    cv2.putText(frame, f"FPS          : {fps:.1f}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

def draw_controls(frame):
    """Show keyboard controls at bottom of frame."""
    h = frame.shape[0]
    cv2.putText(
        frame, "S: Screenshot  |  V: Video Mode  |  Q: Quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )

def save_screenshot(frame):
    """Save current frame as PNG in screenshots folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/detection_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")
    return filename