"""
Tests for src/alarm.py.

Mocks pygame and cv2 so tests run in headless CI environments.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Stub out pygame before importing alarm
pygame_stub = MagicMock()
sys.modules.setdefault("pygame", pygame_stub)


from src.alarm import AlarmManager, _generate_beep  # noqa: E402
import src.config as config  # noqa: E402


# ---------------------------------------------------------------------------
# _generate_beep
# ---------------------------------------------------------------------------

def test_generate_beep_returns_int16_array():
    buf = _generate_beep(440, 0.1)
    assert buf.dtype == np.int16
    assert len(buf) == pytest.approx(44100 * 0.1, rel=0.01)


def test_generate_beep_amplitude_in_range():
    buf = _generate_beep(440, 0.1)
    assert buf.max() <= 32767
    assert buf.min() >= -32768


# ---------------------------------------------------------------------------
# AlarmManager state machine
# ---------------------------------------------------------------------------

class TestAlarmManager:

    def setup_method(self):
        self.alarm = AlarmManager()

    def teardown_method(self):
        self.alarm.close()

    def test_initial_state_not_active(self):
        assert self.alarm.alarm_active is False

    def test_no_phone_no_alarm(self):
        self.alarm.update(phone_detected=False, looking_down=True)
        assert self.alarm.alarm_active is False

    def test_no_looking_down_no_alarm(self):
        self.alarm.update(phone_detected=True, looking_down=False)
        assert self.alarm.alarm_active is False

    def test_condition_met_but_not_yet_threshold(self):
        self.alarm.update(phone_detected=True, looking_down=True)
        # With default 3 s threshold, a single frame should not fire the alarm.
        assert self.alarm.alarm_active is False

    def test_alarm_fires_after_threshold(self):
        # Override threshold to a tiny value so we don't have to sleep 3 s.
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 0.05
        try:
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.1)
            self.alarm.update(phone_detected=True, looking_down=True)
            assert self.alarm.alarm_active is True
        finally:
            config.ALARM_TRIGGER_SECONDS = original

    def test_alarm_stops_when_condition_clears(self):
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 0.05
        try:
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.1)
            self.alarm.update(phone_detected=True, looking_down=True)
            assert self.alarm.alarm_active is True

            # Condition clears
            self.alarm.update(phone_detected=False, looking_down=False)
            assert self.alarm.alarm_active is False
        finally:
            config.ALARM_TRIGGER_SECONDS = original

    def test_toggle_alarm_enabled(self):
        assert self.alarm.alarm_enabled is True
        self.alarm.toggle_alarm_enabled()
        assert self.alarm.alarm_enabled is False
        self.alarm.toggle_alarm_enabled()
        assert self.alarm.alarm_enabled is True

    def test_alarm_disabled_does_not_fire(self):
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 0.05
        try:
            self.alarm.toggle_alarm_enabled()  # disable
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.1)
            self.alarm.update(phone_detected=True, looking_down=True)
            assert self.alarm.alarm_active is False
        finally:
            config.ALARM_TRIGGER_SECONDS = original

    def test_seconds_until_alarm_decreases(self):
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 1.0
        try:
            assert self.alarm.seconds_until_alarm == pytest.approx(1.0)
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.05)
            assert self.alarm.seconds_until_alarm < 1.0
        finally:
            config.ALARM_TRIGGER_SECONDS = original

    def test_seconds_until_alarm_is_zero_when_active(self):
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 0.05
        try:
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.1)
            self.alarm.update(phone_detected=True, looking_down=True)
            assert self.alarm.seconds_until_alarm == pytest.approx(0.0)
        finally:
            config.ALARM_TRIGGER_SECONDS = original

    def test_draw_overlay_does_not_crash(self):
        """draw_overlay must work even without an active alarm."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Just make sure it doesn't raise
        result = self.alarm.draw_overlay(frame)
        assert result is frame

    def test_draw_overlay_alarm_active(self):
        original = config.ALARM_TRIGGER_SECONDS
        config.ALARM_TRIGGER_SECONDS = 0.05
        try:
            self.alarm.update(phone_detected=True, looking_down=True)
            time.sleep(0.1)
            self.alarm.update(phone_detected=True, looking_down=True)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = self.alarm.draw_overlay(frame)
            assert result is frame
            # Banner should have written something (frame is no longer all zeros)
            assert frame.max() > 0
        finally:
            config.ALARM_TRIGGER_SECONDS = original
