"""Tests for src/config.py."""

import pytest

import src.config as config


def test_alarm_trigger_seconds_positive():
    assert config.ALARM_TRIGGER_SECONDS > 0


def test_phone_confidence_threshold_range():
    assert 0.0 < config.PHONE_CONFIDENCE_THRESHOLD < 1.0


def test_looking_down_pitch_degrees_positive():
    assert config.LOOKING_DOWN_PITCH_DEGREES > 0


def test_phone_detection_interval_positive_int():
    assert isinstance(config.PHONE_DETECTION_INTERVAL, int)
    assert config.PHONE_DETECTION_INTERVAL >= 1


def test_face_box_colour_is_bgr_tuple():
    c = config.FACE_BOX_COLOUR
    assert len(c) == 3
    assert all(0 <= v <= 255 for v in c)


def test_phone_box_colour_is_bgr_tuple():
    c = config.PHONE_BOX_COLOUR
    assert len(c) == 3
    assert all(0 <= v <= 255 for v in c)


def test_alarm_beep_frequency_audible():
    assert config.ALARM_BEEP_FREQUENCY >= 100


def test_defaults_match_requirement():
    """The default alarm trigger must be 3 seconds as per spec."""
    assert config.ALARM_TRIGGER_SECONDS == pytest.approx(3.0)
