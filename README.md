# project_wakeup 🛑📱

A real-time Python desktop application that detects when you are using your phone while sitting in front of a computer and triggers a **Doomscrolling Alarm**.

---

## Features

| Feature | Details |
|---|---|
| 📷 Live webcam feed | OpenCV captures and displays frames in real time |
| 👤 Face detection & landmarks | MediaPipe FaceMesh draws a green bounding box + eyes / nose / mouth landmarks |
| 📐 Head-pose estimation | solvePnP computes the pitch angle to determine if you are looking down |
| 📱 Phone detection | YOLOv8-nano (COCO class `cell phone`) detects a phone in the frame |
| 🚨 Doomscrolling Alarm | Visual (flashing red banner) + audio beep fires after **3 seconds** of phone + looking-down |
| 🎛️ Adjustable sensitivity | `+` / `-` keys change the trigger threshold on the fly |
| 📝 Usage logging | Optional timestamped log of alarm events to `phone_usage_log.txt` |
| 🔕 Alarm toggle | Press `a` to enable / disable the alarm at any time |

---

## Requirements

- Python 3.9 or later
- A webcam
- Internet access on first run (YOLOv8 weights are downloaded automatically)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AbdalQader-Sharif/project_wakeup.git
cd project_wakeup

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows (Command Prompt / PowerShell):
.venv\Scripts\activate.bat
# On Windows (Git Bash / MINGW64):
source .venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note – Apple Silicon (M1/M2):** Replace `opencv-python` with
> `opencv-python-headless` if you encounter build issues.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `bash: .venv/bin/activate: No such file or directory` on Windows | Git Bash uses `Scripts/` not `bin/` | Use `source .venv/Scripts/activate` |
| `ERROR: Face detector could not be initialised` | mediapipe incompatibility | Ensure you are using Python 3.9–3.12; mediapipe support for Python 3.13 may be limited |
| `WARNING: Phone detector could not be initialised` | ultralytics/YOLO issue | Phone detection is disabled automatically; face-based detection still works |
| `ERROR: Cannot open webcam` | No camera found | Connect a webcam and ensure no other app is using it |
| Black screen / camera not working on Linux | Missing video device permissions | Run `sudo usermod -aG video $USER` and log out/in |

---

## Running the application

```bash
python main.py
```

A window titled **"Project WakeUp – Doomscrolling Detector"** will open showing the live webcam feed.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `a` | Toggle alarm on / off |
| `l` | Toggle usage logging on / off |
| `+` or `=` | Increase sensitivity (−0.5 s threshold) |
| `-` or `_` | Decrease sensitivity (+0.5 s threshold) |

---

## Project structure

```
project_wakeup/
├── main.py              # Entry point – webcam loop and UI
├── requirements.txt     # Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── config.py        # All tuneable settings
    ├── detector.py      # FaceDetector (MediaPipe) + PhoneDetector (YOLOv8)
    └── alarm.py         # AlarmManager – trigger logic, banner, beep
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_detector.py
    └── test_alarm.py
```

---

## Configuration

All settings are in `src/config.py`.  The most useful ones:

| Setting | Default | Description |
|---|---|---|
| `ALARM_TRIGGER_SECONDS` | `3.0` | Seconds of continuous phone + looking-down before the alarm fires |
| `PHONE_CONFIDENCE_THRESHOLD` | `0.45` | Minimum YOLO confidence to count as a phone detection |
| `LOOKING_DOWN_PITCH_DEGREES` | `15.0` | Head-pitch angle (degrees) that counts as "looking down" |
| `ALARM_SOUND_ENABLED` | `True` | Play an audio beep when the alarm fires |
| `LOG_USAGE` | `False` | Write alarm events to `phone_usage_log.txt` |
| `YOLO_MODEL` | `"yolov8n.pt"` | YOLOv8 model variant (n=nano is fastest) |
| `PHONE_DETECTION_INTERVAL` | `3` | Run phone detection every N frames (higher = faster) |

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## How it works

1. **Face detection** – MediaPipe FaceMesh detects 468 (+ 10 refined) 3-D landmarks every frame.
2. **Head-pose estimation** – OpenCV `solvePnP` fits a canonical 3-D face model to six key landmarks (nose tip, chin, eye corners, mouth corners) and extracts the pitch rotation angle.
3. **Phone detection** – YOLOv8-nano runs on every third frame looking for COCO class `67` (`cell phone`).
4. **Alarm state machine** – If both _looking down_ and _phone detected_ are true for longer than `ALARM_TRIGGER_SECONDS`, the alarm fires:
   - A flashing red banner fills the top of the window.
   - A 880 Hz sine-wave beep repeats via `pygame`.
5. The alarm resets automatically as soon as either condition clears.