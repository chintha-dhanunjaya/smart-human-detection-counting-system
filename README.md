# Smart Human Detection and Counting System

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

A real-time Human Detection and Counting System built using Python and OpenCV.
Detects and counts humans through a live webcam feed using HOG and Haar Cascade algorithms.

Built by **Chintha Dhanunjaya** as a personal project.

---

## Tech Stack

- Python 3.x
- OpenCV
- NumPy
- HOG (Histogram of Oriented Gradients)
- Haar Cascade Classifier
- CSV (Data Logging)
- Git and GitHub

---

## Features

- Real-time human detection and counting via webcam
- Supports video file input for offline testing
- HOG detector with confidence percentage display
- Haar Cascade detector running alongside HOG
- FPS counter for real-time performance monitoring
- Semi-transparent info panel overlay on live video feed
- Press S to save screenshots automatically
- Auto-logs every detection to CSV file with timestamp
- Modular code structure with proper Python packaging

---

## Project Structure
```
smart-human-detection-counting-system/
├── main.py                  ← Entry point
├── detectors/               ← Detection package
│   ├── __init__.py
│   ├── hog_detector.py      ← HOG based detection
│   └── haar_detector.py     ← Haar Cascade detection
├── display.py               ← Overlay and visualization
├── utils.py                 ← Logging, filtering, confidence
├── requirements.txt         ← Dependencies
├── README.md
├── .gitignore
├── logs/                    ← CSV detection logs (auto created)
├── screenshots/             ← Saved screenshots (auto created)
└── videos/                  ← Sample video files
```

---

## How to Run

### Step 1 - Clone the repository
```
git clone https://github.com/chintha-dhanunjaya/smart-human-detection-counting-system.git
cd smart-human-detection-counting-system
```

### Step 2 - Install dependencies
```
pip install -r requirements.txt
```

### Step 3 - Run with webcam
```
python main.py
```

### Step 4 - Run with video file
```
python main.py --video videos/sample.mp4
```

---

## Controls

| Key | Action |
|-----|--------|
| S   | Save screenshot to screenshots folder |
| Q   | Quit the program |

---

## Detection Log

Every detection is automatically saved to logs/detection_log.csv:

| timestamp | hog_count | haar_count | total_detected |
|-----------|-----------|------------|----------------|
| 2025-01-01 10:00:00 | 1 | 0 | 1 |
| 2025-01-01 10:00:03 | 1 | 1 | 2 |

---

## How It Works

1. Webcam captures live video frame by frame
2. Each frame is resized and converted to grayscale
3. HOG descriptor scans the frame and detects human shapes using SVM classifier
4. Haar Cascade simultaneously scans for full body detection
5. Bounding boxes are drawn around detected humans with confidence percentage
6. Detection count, FPS and controls are displayed on screen
7. All detections are logged to CSV automatically every 3 seconds

---

## Known Limitations

- HOG and Haar Cascade require full body to be visible in frame
- Best results when standing 2 to 3 metres away from camera
- Performance depends on lighting conditions and background
- For partial body or face only detection, YOLO or MediaPipe would be more suitable

---

## Future Improvements

- Integrate YOLO for faster and more accurate detection
- Add MediaPipe for upper body and face detection
- Build a web dashboard to visualize detection logs
- Add alert system when human is detected
- Deploy as a web application using Flask

---

## Author

**Chintha Dhanunjaya**

- LinkedIn: https://www.linkedin.com/in/chintha-dhanunjaya/
- GitHub: https://github.com/chintha-dhanunjaya
- Email: chinthadhanu2003@gmail.com
