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
