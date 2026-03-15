"""
main.py – Entry point for project_wakeup.

Run with:
    python main.py

Keyboard shortcuts (press while the window is focused):
    q  – quit
    a  – toggle alarm on/off
    l  – toggle phone-usage logging on/off
    +  – increase sensitivity (lower alarm trigger time by 0.5 s)
    -  – decrease sensitivity (raise  alarm trigger time by 0.5 s)
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from src import config
from src.alarm import AlarmManager
from src.detector import FaceDetector, PhoneDetector


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_face(frame: np.ndarray, face) -> None:
    """Draw bounding box and selected facial landmarks."""
    if not face.detected:
        return

    x, y, w, h = face.bounding_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), config.FACE_BOX_COLOUR, 2)

    for pt in face.landmark_pixels:
        cv2.circle(frame, pt, 2, config.FACE_BOX_COLOUR, -1)

    # Head-pitch indicator
    pitch_text = f"Pitch: {face.pitch_degrees:.1f}°"
    colour = (0, 0, 255) if face.looking_down else config.FACE_BOX_COLOUR
    cv2.putText(
        frame,
        pitch_text,
        (x, max(y - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        1,
        cv2.LINE_AA,
    )


def _draw_phone(frame: np.ndarray, phone) -> None:
    """Draw phone detection bounding box."""
    if not phone.detected:
        return
    x1, y1, x2, y2 = phone.bounding_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.PHONE_BOX_COLOUR, 2)
    cv2.putText(
        frame,
        f"Phone {phone.confidence:.0%}",
        (x1, max(y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.PHONE_BOX_COLOUR,
        1,
        cv2.LINE_AA,
    )


def _draw_status_bar(frame: np.ndarray, face, phone, alarm: AlarmManager) -> None:
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (40, 40, 40), -1)

    parts = [
        f"Face: {'YES' if face.detected else 'NO'}",
        f"Phone: {'YES' if phone.detected else 'NO'}",
        f"Looking down: {'YES' if face.looking_down else 'NO'}",
        f"Alarm threshold: {config.ALARM_TRIGGER_SECONDS:.1f}s",
        "  [q]uit  [a]larm  [l]og  [+/-] sensitivity",
    ]
    text = "  |  ".join(parts)
    cv2.putText(
        frame,
        text,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    if config.DISPLAY_WIDTH <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == config.DISPLAY_WIDTH:
        return frame
    scale = config.DISPLAY_WIDTH / w
    new_h = int(h * scale)
    return cv2.resize(frame, (config.DISPLAY_WIDTH, new_h))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Make sure a camera is connected.")
        sys.exit(1)

    print("Initialising detectors …")
    face_detector = FaceDetector()
    try:
        phone_detector: PhoneDetector | None = PhoneDetector()
    except Exception as exc:
        print(f"WARNING: Phone detector could not be initialised ({exc}).")
        print("         Phone detection will be disabled.")
        phone_detector = None

    alarm = AlarmManager()

    print(f"Running.  Window: '{config.WINDOW_TITLE}'")
    print("Press 'q' to quit.")

    # Track FPS
    prev_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to capture frame.")
                continue

            # Mirror the frame so the display feels natural
            frame = cv2.flip(frame, 1)

            # --- Detection ---
            face = face_detector.process(frame)
            from src.detector import PhoneResult
            phone = phone_detector.process(frame) if phone_detector else PhoneResult()

            # --- Alarm logic ---
            alarm.update(phone.detected, face.looking_down)

            # --- Draw overlays ---
            _draw_face(frame, face)
            _draw_phone(frame, phone)
            alarm.draw_overlay(frame)
            _draw_status_bar(frame, face, phone, alarm)

            # FPS counter
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (frame.shape[1] - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            display_frame = _resize_for_display(frame)
            cv2.imshow(config.WINDOW_TITLE, display_frame)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                alarm.toggle_alarm_enabled()
                status = "ON" if alarm.alarm_enabled else "OFF"
                print(f"Alarm toggled: {status}")
            elif key == ord("l"):
                config.LOG_USAGE = not config.LOG_USAGE
                status = "ON" if config.LOG_USAGE else "OFF"
                print(f"Usage logging: {status}")
            elif key == ord("+") or key == ord("="):
                config.ALARM_TRIGGER_SECONDS = max(
                    0.5, config.ALARM_TRIGGER_SECONDS - 0.5
                )
                print(f"Alarm threshold: {config.ALARM_TRIGGER_SECONDS:.1f}s")
            elif key == ord("-") or key == ord("_"):
                config.ALARM_TRIGGER_SECONDS += 0.5
                print(f"Alarm threshold: {config.ALARM_TRIGGER_SECONDS:.1f}s")

    finally:
        cap.release()
        face_detector.close()
        alarm.close()
        cv2.destroyAllWindows()
        print("Bye!")


if __name__ == "__main__":
    main()
