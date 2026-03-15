"""
Configuration settings for project_wakeup.

All tuneable parameters live here so users can adjust sensitivity
and behaviour without touching the core logic.
"""

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Number of seconds the user must be looking down at a phone before the alarm
# fires.  Lower values = more sensitive; higher values = more lenient.
ALARM_TRIGGER_SECONDS: float = 3.0

# Minimum YOLO confidence to consider a detection a "phone".
PHONE_CONFIDENCE_THRESHOLD: float = 0.45

# Head-pitch angle (degrees, positive = looking down) above which we consider
# the user to be looking downward.
LOOKING_DOWN_PITCH_DEGREES: float = 15.0

# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------

# Set to False to silence the audio component of the alarm.
ALARM_SOUND_ENABLED: bool = True

# Frequency of the generated beep tone (Hz).
ALARM_BEEP_FREQUENCY: int = 880

# Duration of a single beep cycle (seconds).
ALARM_BEEP_DURATION: float = 0.4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Set to True to write phone-usage events to LOG_FILE.
LOG_USAGE: bool = False

# Path of the usage log file (relative to the working directory).
LOG_FILE: str = "phone_usage_log.txt"

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Colour of the face bounding box and landmarks (BGR).
FACE_BOX_COLOUR: tuple = (0, 255, 0)   # green
PHONE_BOX_COLOUR: tuple = (0, 165, 255)  # orange

# Banner colours (BGR).
WARNING_BANNER_COLOUR: tuple = (0, 0, 200)   # dark red
ALARM_BANNER_COLOUR: tuple = (0, 0, 255)     # bright red

# Window title shown in the OpenCV window.
WINDOW_TITLE: str = "Project WakeUp – Doomscrolling Detector"

# Target display width for the OpenCV window (pixels).  Set to 0 to use the
# camera's native resolution.
DISPLAY_WIDTH: int = 960

# ---------------------------------------------------------------------------
# YOLO model
# ---------------------------------------------------------------------------

# YOLOv8 model variant.  "yolov8n.pt" (nano) is fast; swap for a larger
# variant for better accuracy at the cost of speed.
YOLO_MODEL: str = "yolov8n.pt"

# COCO class index for "cell phone".
PHONE_CLASS_ID: int = 67

# How many frames to skip between phone-detection runs.  Phone detection is
# heavier than face detection; running it every N frames keeps the UI smooth.
PHONE_DETECTION_INTERVAL: int = 3
