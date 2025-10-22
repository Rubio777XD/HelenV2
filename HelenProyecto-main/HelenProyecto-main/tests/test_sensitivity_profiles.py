"""Tests covering HELEN sensitivity profiles and runtime switching."""

from __future__ import annotations

from typing import Tuple

import pytest

import backendHelen.server as server


@pytest.mark.parametrize("mode", ["STRICT", "BALANCED", "RELAXED"])
def test_sensitivity_profiles_are_available(mode: str) -> None:
    """Every expected sensitivity mode should be parsed from the JSON file."""

    assert mode in server.SENSITIVITY_PROFILES


def test_balanced_profile_matches_configuration() -> None:
    """The BALANCED profile should reflect the calibrated thresholds."""

    balanced = server.SENSITIVITY_PROFILES["BALANCED"]

    assert balanced.quality.blur_laplacian_min == pytest.approx(60.0)
    assert balanced.quality.roi_min_coverage == pytest.approx(0.72)
    assert balanced.quality.hand_range_px == pytest.approx((90.0, 480.0))

    assert balanced.temporal.consensus_n == 2
    assert balanced.temporal.consensus_m == 4
    assert balanced.temporal.cooldown_s == pytest.approx(0.65)
    assert balanced.temporal.listen_window_s == pytest.approx(4.0)
    assert balanced.temporal.min_pos_stability_var == pytest.approx(14.0)

    clima_profile = balanced.classes["Clima"]
    assert clima_profile.score_min == pytest.approx(0.48)
    assert clima_profile.angle_tol_deg == pytest.approx(26.0)
    assert clima_profile.norm_dev_max == pytest.approx(0.28)
    assert clima_profile.curvature_min == pytest.approx(0.32)
    assert clima_profile.gap_ratio_range == pytest.approx((0.28, 0.55))


def test_class_thresholds_apply_hysteresis() -> None:
    """Hysteresis offsets should be respected when computing runtime thresholds."""

    balanced = server.SENSITIVITY_PROFILES["BALANCED"]
    thresholds = server._class_thresholds_from_profile(balanced, server.HYSTERESIS_SETTINGS)

    clima = thresholds["Clima"]
    expected_enter = balanced.classes["Clima"].score_min + server.HYSTERESIS_SETTINGS.on_offset
    expected_release = max(0.0, expected_enter - server.HYSTERESIS_SETTINGS.off_delta)

    assert clima.enter == pytest.approx(expected_enter)
    assert clima.release == pytest.approx(expected_release)


def test_runtime_frameskip_adjusts_with_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Balanced mode should toggle the frameskip based on measured FPS."""

    class DummyFeatureNormalizer:
        def __init__(self, dataset_path):
            self._dataset_path = dataset_path

        def transform(self, features):
            return list(features)

        def snapshot(self):
            return {
                "dataset": str(self._dataset_path),
                "loaded": False,
                "uses_transformer": False,
                "uses_stats": False,
            }

    class DummyStream:
        source = "dummy"

        def __init__(self) -> None:
            self.closed = False

        def next(self, timeout: float = 0.0):  # noqa: ARG002 - signature parity
            raise TimeoutError("no synthetic frames")

        def status(self):
            return {"healthy": True}

        def close(self) -> None:
            self.closed = True

        def measured_fps(self) -> float:
            return 0.0

        def last_landmarks(self):  # pragma: no cover - not exercised
            return None

        def last_roi(self):  # pragma: no cover - not exercised
            return None

    class DummyPipeline:
        def __init__(self, runtime, interval_s: float, frame_stride: int) -> None:
            self.runtime = runtime
            self.interval_s = interval_s
            self.frame_stride = frame_stride
            self.running = False

        def start(self) -> None:  # pragma: no cover - not exercised
            self.running = True

        def stop(self) -> None:
            self.running = False

        def is_running(self) -> bool:
            return self.running

        def set_frame_stride(self, stride: int) -> None:
            self.frame_stride = int(stride)

    class DummyClassifier:
        source = "dummy"

        def predict(self, features):  # pragma: no cover - not exercised
            return server.Prediction(label="Start", score=0.9)

    def fake_create_classifier(self):
        return DummyClassifier(), {"source": DummyClassifier.source, "loaded": True}

    def fake_create_stream(self) -> Tuple[DummyStream, dict]:
        return DummyStream(), {"source": DummyStream.source}

    monkeypatch.setattr(server, "FeatureNormalizer", DummyFeatureNormalizer)
    monkeypatch.setattr(server, "GesturePipeline", DummyPipeline)
    monkeypatch.setattr(server.HelenRuntime, "_create_classifier", fake_create_classifier, raising=False)
    monkeypatch.setattr(server.HelenRuntime, "_load_secondary_classifier", lambda self, path: None, raising=False)
    monkeypatch.setattr(server.HelenRuntime, "_create_stream", fake_create_stream, raising=False)
    monkeypatch.setattr(server.HelenRuntime, "_create_geometry_verifier", lambda self: None, raising=False)
    monkeypatch.setattr(server.HelenRuntime, "_export_session_report", lambda self: None, raising=False)

    config = server.RuntimeConfig(
        enable_camera=False,
        fallback_to_synthetic=True,
        poll_interval_s=server.DEFAULT_POLL_INTERVAL_S,
        process_every_n=3,
        sensitivity_mode="BALANCED",
    )

    runtime = server.HelenRuntime(config)
    try:
        high = runtime.update_frameskip(runtime.rate_limit_profile.fps_threshold + 5.0)
        assert high == runtime.rate_limit_profile.frameskip_balanced

        low = runtime.update_frameskip(runtime.rate_limit_profile.fps_threshold - 5.0)
        assert low == 1
    finally:
        runtime.stop()
