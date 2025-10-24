import importlib

import pytest

from backendHelen import camera_probe


def test_list_sources_structure():
    sources = camera_probe.list_sources()
    assert isinstance(sources, dict)
    assert 'v4l2' in sources
    assert 'libcamera' in sources
    assert isinstance(sources['v4l2'], list)
    assert isinstance(sources['libcamera'], list)


def test_parse_resolution_valid_and_invalid():
    assert camera_probe.parse_resolution('640x480') == (640, 480)
    with pytest.raises(ValueError):
        camera_probe.parse_resolution('invalid')


def test_ensure_camera_selection_without_opencv(monkeypatch):
    # Simular ausencia de OpenCV para garantizar manejo seguro
    monkeypatch.setattr(camera_probe, 'cv2', None)
    importlib.reload(camera_probe)
    monkeypatch.setattr(camera_probe, 'cv2', None)
    selection = camera_probe.ensure_camera_selection(force=True, logger=None)
    assert selection is None
    importlib.reload(camera_probe)


def test_probe_prefers_high_resolution(monkeypatch):
    candidate = camera_probe.CameraCandidate(
        identifier="test:/dev/video0",
        label="Test Cam",
        kind="usb",
        backend_hint="v4l2",
        path="/dev/video0",
        index=0,
    )

    attempts = []

    def fake_probe_v4l2(local_candidate, mode):
        attempts.append(mode.label)
        return camera_probe.ProbeResult(
            candidate=local_candidate,
            backend="v4l2",
            mode=mode,
            success=True,
            reason=None,
            latency_ms=10.0,
            resolution=(mode.width, mode.height),
            fps=float(mode.fps),
            orientation="landscape",
            frames_sampled=12,
            pixel_format="MJPG",
        )

    def fake_probe_gstreamer(local_candidate, mode):
        return camera_probe.ProbeResult(
            candidate=local_candidate,
            backend="gstreamer",
            mode=mode,
            success=False,
            reason="skip",
        )

    monkeypatch.setattr(camera_probe, "_list_v4l2_devices", lambda: [candidate])
    monkeypatch.setattr(camera_probe, "_list_libcamera_devices", lambda: [])
    monkeypatch.setattr(camera_probe, "_fallback_candidates", lambda: [])
    monkeypatch.setattr(camera_probe, "_probe_with_v4l2", fake_probe_v4l2)
    monkeypatch.setattr(camera_probe, "_probe_with_gstreamer", fake_probe_gstreamer)
    monkeypatch.setattr(camera_probe, "cv2", object())
    monkeypatch.setattr(camera_probe, "DEFAULT_CAPTURE_FLAG", 0)

    selection = camera_probe.ensure_camera_selection(force=True, logger=None)
    assert selection is not None
    assert attempts, "El flujo de sondeo no se ejecutó"
    assert attempts[0] == "1280x720@30"
    assert selection.width == 1280
    assert selection.height == 720
    assert selection.pixel_format == "MJPG"
