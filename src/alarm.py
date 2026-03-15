"""
alarm.py – Alarm management for project_wakeup.

The AlarmManager tracks how long both conditions (phone detected AND looking
down) have been continuously true and fires a visual + optional audio alarm
once the configured threshold is exceeded.
"""

from __future__ import annotations

import datetime
import math
import threading
import time
from typing import Optional

import numpy as np

from src import config


class AlarmManager:
    """Stateful alarm controller.

    Call :meth:`update` every frame.  Inspect :attr:`alarm_active` and call
    :meth:`draw_overlay` to render banners on a frame.
    """

    def __init__(self) -> None:
        self._condition_start: Optional[float] = None  # monotonic timestamp
        self._alarm_active: bool = False
        self._sound_thread: Optional[threading.Thread] = None
        self._sound_running: bool = False
        self._alarm_enabled: bool = True

        # Logging
        self._log_session_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alarm_active(self) -> bool:
        return self._alarm_active

    @property
    def alarm_enabled(self) -> bool:
        return self._alarm_enabled

    @property
    def seconds_until_alarm(self) -> float:
        """Remaining seconds before the alarm fires (0 when already active)."""
        if self._alarm_active:
            return 0.0
        if self._condition_start is None:
            return config.ALARM_TRIGGER_SECONDS
        elapsed = time.monotonic() - self._condition_start
        return max(0.0, config.ALARM_TRIGGER_SECONDS - elapsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, phone_detected: bool, looking_down: bool) -> None:
        """Update alarm state based on current-frame detection results."""
        condition_met = phone_detected and looking_down

        if condition_met:
            if self._condition_start is None:
                self._condition_start = time.monotonic()
            elapsed = time.monotonic() - self._condition_start
            if elapsed >= config.ALARM_TRIGGER_SECONDS and not self._alarm_active:
                self._fire_alarm()
        else:
            if self._alarm_active:
                self._stop_alarm()
            self._condition_start = None

    def toggle_alarm_enabled(self) -> None:
        """Toggle whether the alarm can fire (keyboard shortcut support)."""
        self._alarm_enabled = not self._alarm_enabled
        if not self._alarm_enabled:
            self._stop_alarm()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw warning / alarm banners onto *frame* (in-place) and return it."""
        h, w = frame.shape[:2]
        banner_h = 50

        if self._alarm_active:
            # Flashing alarm banner
            flash = (int(time.monotonic() * 4) % 2 == 0)
            colour = config.ALARM_BANNER_COLOUR if flash else (200, 0, 0)
            cv2 = _get_cv2()
            cv2.rectangle(frame, (0, 0), (w, banner_h), colour, -1)
            cv2.putText(
                frame,
                "!!! DOOMSCROLLING ALARM – PUT DOWN YOUR PHONE !!!",
                (10, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self._condition_start is not None:
            cv2 = _get_cv2()
            remaining = self.seconds_until_alarm
            cv2.rectangle(frame, (0, 0), (w, banner_h), config.WARNING_BANNER_COLOUR, -1)
            cv2.putText(
                frame,
                f"Phone usage detected – alarm in {remaining:.1f}s",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if not self._alarm_enabled:
            cv2 = _get_cv2()
            cv2.putText(
                frame,
                "[ALARM OFF]",
                (w - 160, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def log_event(self, label: str) -> None:
        """Append a timestamped event to the log file if logging is enabled."""
        if not config.LOG_USAGE:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(config.LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{ts}  {label}\n")

    def close(self) -> None:
        self._stop_alarm()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire_alarm(self) -> None:
        if not self._alarm_enabled:
            return
        self._alarm_active = True
        self.log_event("ALARM START – phone usage detected")
        if config.ALARM_SOUND_ENABLED:
            self._start_sound()

    def _stop_alarm(self) -> None:
        if self._alarm_active:
            self.log_event("ALARM STOP")
        self._alarm_active = False
        self._stop_sound()

    def _start_sound(self) -> None:
        if self._sound_running:
            return
        self._sound_running = True
        self._sound_thread = threading.Thread(
            target=self._sound_loop, daemon=True
        )
        self._sound_thread.start()

    def _stop_sound(self) -> None:
        self._sound_running = False

    def _sound_loop(self) -> None:
        """Generate and play beeps in a background thread."""
        try:
            import pygame  # type: ignore
            pygame.mixer.pre_init(44100, -16, 1, 512)
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            while self._sound_running and self._alarm_active:
                buf = _generate_beep(
                    config.ALARM_BEEP_FREQUENCY,
                    config.ALARM_BEEP_DURATION,
                )
                sound = pygame.sndarray.make_sound(buf)
                sound.play()
                pygame.time.wait(int(config.ALARM_BEEP_DURATION * 1000))
        except Exception:
            # Graceful degradation: no sound if pygame is unavailable / fails.
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_beep(frequency: int, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Return a mono 16-bit PCM numpy array for a sine-wave beep."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (np.sin(2 * math.pi * frequency * t) * 32767).astype(np.int16)
    return wave


def _get_cv2():
    """Lazy import of cv2 to allow unit tests to mock it."""
    import cv2 as _cv2_mod  # noqa: PLC0415
    return _cv2_mod
