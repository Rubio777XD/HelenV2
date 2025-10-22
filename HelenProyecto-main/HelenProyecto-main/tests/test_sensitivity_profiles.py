"""Tests covering the soft sensitivity profile and runtime behaviour."""

from typing import Tuple

import pytest

import backendHelen.server as server


def test_soft_profile_matches_configuration() -> None:
    """The global soft profile should reflect the calibrated thresholds."""

    profile = server.SENSITIVITY_PROFILE

    assert profile.quality.blur_laplacian_min == pytest.approx(32.0)
    assert profile.quality.roi_min_coverage == pytest.approx(0.58)
    assert profile.quality.hand_range_px == pytest.approx((60.0, 600.0))

    assert profile.temporal.consensus_n == 2
    assert profile.temporal.consensus_m == 4
    assert profile.temporal.cooldown_s == pytest.approx(0.72)
    assert profile.temporal.listen_window_s == pytest.approx(5.6)
    assert profile.temporal.min_pos_stability_var == pytest.approx(10.0)
    assert profile.temporal.activation_delay_s == pytest.approx(0.44)

    clima_profile = profile.classes["Clima"]
    assert clima_profile.score_min == pytest.approx(0.369)
    assert clima_profile.angle_tol_deg == pytest.approx(40.0)
    assert clima_profile.norm_dev_max == pytest.approx(0.39)
    assert clima_profile.curvature_min == pytest.approx(0.20)
    assert clima_profile.gap_ratio_range == pytest.approx((0.20, 0.68))
    assert clima_profile.missing_distal_allowance == 2
    assert clima_profile.missing_distal_strict_curvature == pytest.approx(0.24)
    assert clima_profile.curvature_consistency_boost == pytest.approx(0.03)
    assert clima_profile.curvature_consistency_window == 3
    assert clima_profile.curvature_consistency_min_frames == 2
    assert clima_profile.curvature_consistency_tolerance == pytest.approx(0.08)
    assert clima_profile.curvature_consistency_min_curvature == pytest.approx(0.26)


def test_class_thresholds_apply_hysteresis() -> None:
    """Hysteresis offsets should be respected when computing runtime thresholds."""

    thresholds = server._class_thresholds_from_profile(server.SENSITIVITY_PROFILE, server.HYSTERESIS_SETTINGS)

    clima = thresholds["Clima"]
    expected_enter = server.SENSITIVITY_PROFILE.classes["Clima"].score_min + server.HYSTERESIS_SETTINGS.on_offset
    expected_release = max(0.0, expected_enter - server.HYSTERESIS_SETTINGS.off_delta)

    assert clima.enter == pytest.approx(expected_enter)
    assert clima.release == pytest.approx(expected_release)


def test_runtime_frameskip_adjusts_with_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime should drop to the minimal stride when the FPS drops below the target."""

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
    )

    runtime = server.HelenRuntime(config)
    try:
        high = runtime.update_frameskip(runtime.rate_limit_profile.fps_threshold + 5.0)
        assert high == runtime.base_frameskip

        low = runtime.update_frameskip(runtime.rate_limit_profile.fps_threshold - 5.0)
        assert low == runtime.rate_limit_profile.min_frameskip
    finally:
        runtime.stop()
