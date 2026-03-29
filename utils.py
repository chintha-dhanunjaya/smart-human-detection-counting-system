# ================================================
# Utility Functions — Logging, Filtering, Confidence
# Author: Chintha Dhanunjaya
# ================================================

import csv
import os
import numpy as np
from datetime import datetime

LOG_FILE = "logs/detection_log.csv"

def initialize_directories():
    """Create required folders if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

def initialize_log():
    """Create CSV log file with headers if not present."""
    initialize_directories()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "hog_count",
                "haar_count",
                "total_detected"
            ])
        print(f"Log file created: {LOG_FILE}")

def log_detection(hog_count, haar_count):
    """Append a detection record to the CSV log."""
    total = hog_count + haar_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, hog_count, haar_count, total])

def filter_boxes(boxes, min_size=40):
    """Filter out bounding boxes smaller than min_size."""
    if len(boxes) == 0:
        return boxes
    return np.array([
        b for b in boxes
        if b[2] >= min_size and b[3] >= min_size
    ])

def compute_confidence(weight):
    """Convert HOG SVM weight to a 0-100 confidence score."""
    return min(100, int(abs(weight) * 50))