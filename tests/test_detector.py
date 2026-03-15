"""
Tests for src/detector.py.

These tests avoid real webcam / GPU resources by mocking the heavy
dependencies (mediapipe, ultralytics).
"""

from __future__ import annotations

import types
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# FaceResult / PhoneResult dataclasses
# ---------------------------------------------------------------------------

def test_face_result_defaults():
    from src.detector import FaceResult
    r = FaceResult()
    assert r.detected is False
    assert r.bounding_box is None
    assert r.pitch_degrees == pytest.approx(0.0)
    assert r.looking_down is False
    assert r.landmark_pixels == []


def test_phone_result_defaults():
    from src.detector import PhoneResult
    r = PhoneResult()
    assert r.detected is False
    assert r.bounding_box is None
    assert r.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class TestFaceDetector:
    """Unit-test FaceDetector with mocked MediaPipe."""

    def _make_detector(self):
        """Return a FaceDetector whose MediaPipe internals are mocked."""
        mp_mock = MagicMock()
        face_mesh_instance = MagicMock()
        mp_mock.solutions.face_mesh.FaceMesh.return_value = face_mesh_instance
        with patch.dict(sys.modules, {"mediapipe": mp_mock}):
            from importlib import reload
            import src.detector as det_mod
            reload(det_mod)
            detector = det_mod.FaceDetector()
        return detector, face_mesh_instance

    def test_no_face_returns_not_detected(self):
        detector, mesh = self._make_detector()
        mesh.process.return_value = MagicMock(multi_face_landmarks=None)
        frame = _make_frame()
        with patch("cv2.cvtColor", return_value=frame):
            result = detector.process(frame)
        assert result.detected is False

    def test_face_detected_sets_flag(self):
        detector, mesh = self._make_detector()

        # Build 478 fake landmarks (mediapipe refine gives 478)
        lms = [_make_fake_landmark(0.5, 0.5) for _ in range(478)]
        face_lm_group = MagicMock()
        face_lm_group.landmark = lms
        mesh.process.return_value = MagicMock(
            multi_face_landmarks=[face_lm_group]
        )

        frame = _make_frame()
        with patch("cv2.cvtColor", return_value=frame), \
             patch("cv2.solvePnP", return_value=(False, None, None)):
            result = detector.process(frame)

        assert result.detected is True
        assert result.bounding_box is not None

    def test_bounding_box_within_frame(self):
        detector, mesh = self._make_detector_with_face()
        frame = _make_frame(480, 640)
        with patch("cv2.cvtColor", return_value=frame), \
             patch("cv2.solvePnP", return_value=(False, None, None)):
            result = detector.process(frame)
        if result.detected and result.bounding_box:
            x, y, w, h = result.bounding_box
            assert x >= 0
            assert y >= 0
            assert x + w <= 640
            assert y + h <= 480

    def _make_detector_with_face(self):
        detector, mesh = self._make_detector()
        lms = [_make_fake_landmark(0.5, 0.5) for _ in range(478)]
        face_lm_group = MagicMock()
        face_lm_group.landmark = lms
        mesh.process.return_value = MagicMock(
            multi_face_landmarks=[face_lm_group]
        )
        return detector, mesh


# ---------------------------------------------------------------------------
# PhoneDetector
# ---------------------------------------------------------------------------

class TestPhoneDetector:
    """Unit-test PhoneDetector with mocked ultralytics."""

    def _make_detector(self, detections=None):
        yolo_mock = MagicMock()
        model_instance = MagicMock()
        yolo_mock.return_value = model_instance

        if detections is None:
            model_instance.return_value = []

        with patch.dict(sys.modules, {"ultralytics": MagicMock(YOLO=yolo_mock)}):
            from importlib import reload
            import src.detector as det_mod
            reload(det_mod)
            det = det_mod.PhoneDetector()

        return det, model_instance

    def test_no_phone_returns_not_detected(self):
        detector, model = self._make_detector()
        model.return_value = []
        frame = _make_frame()
        # Ensure the frame counter triggers detection
        from src import config as cfg
        detector._frame_counter = cfg.PHONE_DETECTION_INTERVAL - 1
        result = detector.process(frame)
        assert result.detected is False

    def test_phone_detection_result_structure(self):
        from src.detector import PhoneResult
        import src.config as cfg

        detector, model = self._make_detector()

        # Mock a detection
        box_mock = MagicMock()
        box_mock.cls = [cfg.PHONE_CLASS_ID]
        box_mock.conf = [0.85]
        box_mock.xyxy = [[10, 20, 100, 200]]
        result_mock = MagicMock()
        result_mock.boxes = [box_mock]
        model.return_value = [result_mock]

        detector._frame_counter = cfg.PHONE_DETECTION_INTERVAL - 1
        frame = _make_frame()
        result = detector.process(frame)
        assert result.detected is True
        assert result.confidence == pytest.approx(0.85)
        assert result.bounding_box == (10, 20, 100, 200)

    def test_interval_skips_detection(self):
        detector, model = self._make_detector()
        model.return_value = []
        frame = _make_frame()
        # Reset and call once (counter goes 1, which skips if interval > 1)
        detector._frame_counter = 0
        from src import config as cfg
        if cfg.PHONE_DETECTION_INTERVAL > 1:
            result = detector.process(frame)
            model.assert_not_called()
