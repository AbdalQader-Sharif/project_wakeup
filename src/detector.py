"""
detector.py – Face-landmark detection, head-pose estimation, and phone detection.

Classes
-------
FaceDetector
    Wraps MediaPipe FaceMesh to detect face landmarks and estimate whether the
    user is looking downward by computing a head-pitch angle via solvePnP.

PhoneDetector
    Wraps YOLOv8 to detect a "cell phone" in a video frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Results produced by FaceDetector for a single frame."""

    detected: bool = False
    # Bounding box (x, y, w, h) in pixel coordinates
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    # Head pitch angle in degrees (positive = looking down)
    pitch_degrees: float = 0.0
    # Whether the head is oriented downward beyond the configured threshold
    looking_down: bool = False
    # Raw landmark pixel coordinates for drawing [(x, y), ...]
    landmark_pixels: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class PhoneResult:
    """Results produced by PhoneDetector for a single frame."""

    detected: bool = False
    # Bounding box (x1, y1, x2, y2) in pixel coordinates, or None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# 3-D reference face model used for solvePnP head-pose estimation
# (millimetre scale, canonical face centred at the nose tip)
# ---------------------------------------------------------------------------
_FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip           – landmark 1
        [0.0, -63.6, -12.5],      # Chin               – landmark 152
        [-43.3, 32.7, -26.0],     # Left eye outer     – landmark 33
        [43.3, 32.7, -26.0],      # Right eye outer    – landmark 263
        [-28.9, -28.9, -24.1],    # Left mouth corner  – landmark 61
        [28.9, -28.9, -24.1],     # Right mouth corner – landmark 291
    ],
    dtype=np.float64,
)

_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Subset of landmark indices to draw as "facial landmarks" on the overlay
# (eyes, nose tip, mouth corners + lips outline)
_DRAW_LANDMARK_IDS = [
    # Left eye
    33, 160, 158, 133, 153, 144,
    # Right eye
    362, 385, 387, 263, 373, 380,
    # Nose tip
    1,
    # Mouth corners + lips
    61, 291, 78, 308, 13, 14,
]


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class FaceDetector:
    """Detect face landmarks and estimate head pose using MediaPipe FaceMesh."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> FaceResult:
        """Run face detection + head-pose estimation on *frame* (BGR)."""
        h, w = frame.shape[:2]
        self._ensure_camera_matrix(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceResult(detected=False)

        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark  # list of NormalizedLandmark

        # Convert to pixel coordinates
        px = [(int(p.x * w), int(p.y * h)) for p in lm]

        # Bounding box
        xs = [p[0] for p in px]
        ys = [p[1] for p in px]
        x0, y0 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x1, y1 = min(max(xs) + 10, w), min(max(ys) + 10, h)
        bbox = (x0, y0, x1 - x0, y1 - y0)

        # Head-pose estimation
        pitch = self._estimate_pitch(lm, w, h)
        looking_down = pitch > config.LOOKING_DOWN_PITCH_DEGREES

        # Selected landmarks to draw
        draw_pts = [px[i] for i in _DRAW_LANDMARK_IDS if i < len(px)]

        return FaceResult(
            detected=True,
            bounding_box=bbox,
            pitch_degrees=pitch,
            looking_down=looking_down,
            landmark_pixels=draw_pts,
        )

    def close(self) -> None:
        self._face_mesh.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_camera_matrix(self, w: int, h: int) -> None:
        if self._camera_matrix is not None:
            return
        focal = w  # approximate focal length = image width
        cx, cy = w / 2.0, h / 2.0
        self._camera_matrix = np.array(
            [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64
        )

    def _estimate_pitch(
        self, landmarks, w: int, h: int
    ) -> float:
        """Return the head-pitch angle in degrees (positive = nose down)."""
        try:
            image_points = np.array(
                [
                    (landmarks[i].x * w, landmarks[i].y * h)
                    for i in _POSE_LANDMARK_IDS
                ],
                dtype=np.float64,
            )
            success, rvec, _ = cv2.solvePnP(
                _FACE_3D_MODEL,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return 0.0

            # Convert rotation vector to Euler angles
            rmat, _ = cv2.Rodrigues(rvec)
            pitch_rad = math.asin(-rmat[2, 0])
            return math.degrees(pitch_rad)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# PhoneDetector
# ---------------------------------------------------------------------------

class PhoneDetector:
    """Detect cell phones in video frames using YOLOv8."""

    def __init__(self) -> None:
        # Import here so that the rest of the app can load even if
        # ultralytics is not installed (tests mock this out).
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(config.YOLO_MODEL)
        self._frame_counter = 0
        self._last_result: PhoneResult = PhoneResult()

    def process(self, frame: np.ndarray) -> PhoneResult:
        """Run phone detection on *frame*.

        To keep the UI smooth the model only runs every
        ``config.PHONE_DETECTION_INTERVAL`` frames; the cached result is
        returned on skipped frames.
        """
        self._frame_counter += 1
        if self._frame_counter % config.PHONE_DETECTION_INTERVAL != 0:
            return self._last_result

        results = self._model(
            frame,
            verbose=False,
            conf=config.PHONE_CONFIDENCE_THRESHOLD,
            classes=[config.PHONE_CLASS_ID],
        )

        best: PhoneResult = PhoneResult()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != config.PHONE_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                if conf < config.PHONE_CONFIDENCE_THRESHOLD:
                    continue
                if conf > best.confidence:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    best = PhoneResult(
                        detected=True,
                        bounding_box=(x1, y1, x2, y2),
                        confidence=conf,
                    )

        self._last_result = best
        return best
