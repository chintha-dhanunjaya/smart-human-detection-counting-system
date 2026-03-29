# ================================================
# Human Detection System — Main Entry Point
# Author : Chintha Dhanunjaya
# Tech   : Python, OpenCV, NumPy, HOG, Haar Cascade
# Usage  : python main.py
#          python main.py --video videos/sample.mp4
# ================================================

import cv2
import time
import argparse

from detectors import get_hog_detector, detect_with_hog
from detectors import get_haar_detector, detect_with_haar
from display import (draw_hog_boxes, draw_haar_boxes,
                     draw_info_panel, draw_controls, save_screenshot)
from utils import initialize_log, log_detection

# ── Config ─────────────────────────────────────
USE_WEBCAM           = True
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH          = 720
FRAME_HEIGHT         = 540
LOG_INTERVAL         = 3  # seconds

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Human Detection System by Chintha Dhanunjaya"
    )
    parser.add_argument(
        '--video', type=str, default=None,
        help='Path to video file. If not given, uses webcam.'
    )
    return parser.parse_args()

def run_detection(source):
    """Main detection loop for webcam or video file."""
    initialize_log()

    # ── Load Detectors ─────────────────────────
    hog  = get_hog_detector()
    haar = get_haar_detector()

    # ── Open Video Source ──────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    mode = "VIDEO FILE" if isinstance(source, str) else "WEBCAM"
    print("=" * 45)
    print(f"  Human Detection System — {mode}")
    print("=" * 45)
    print("  Controls: S = Screenshot | Q = Quit")
    print("=" * 45)

    prev_time = time.time()
    last_log  = time.time()

    while True:
        ret, frame = cap.read()

        # ── Loop video when it ends ────────────
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Webcam feed lost.")
                break

        # ── Preprocess Frame ───────────────────
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Run Detections ─────────────────────
        hog_boxes,  weights = detect_with_hog(frame, hog)
        haar_bodies         = detect_with_haar(gray, haar)

        # ── Draw Results ───────────────────────
        draw_hog_boxes(frame, hog_boxes, weights)
        draw_haar_boxes(frame, haar_bodies)

        # ── Calculate FPS ──────────────────────
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # ── Draw Info Panel & Controls ─────────
        draw_info_panel(frame, len(hog_boxes), len(haar_bodies), fps)
        draw_controls(frame)

        # ── Log to CSV every 3 seconds ─────────
        if curr_time - last_log >= LOG_INTERVAL:
            log_detection(len(hog_boxes), len(haar_bodies))
            last_log = curr_time

        # ── Show Frame ─────────────────────────
        cv2.imshow("Human Detection System — Chintha Dhanunjaya", frame)

        # ── Handle Key Presses ─────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting system...")
            break
        elif key == ord('s'):
            save_screenshot(frame)

    # ── Cleanup ────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Detection complete. Log saved to logs/detection_log.csv")

def main():
    args   = parse_args()
    source = args.video if args.video else 0
    run_detection(source)

if __name__ == "__main__":
    main()