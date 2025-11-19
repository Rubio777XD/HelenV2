"""Herramienta de depuración para el backend LSTM sin cámara.

Carga el mismo modelo TensorFlow usado por el servidor, aplica el
preprocesado de ``TensorFlowSequenceGestureClassifier`` y ejecuta la lógica
completa de la ``GestureDecisionEngine`` sobre una secuencia de landmarks
pregrabada o simulada. Útil para reproducir descartes (score bajo,
consenso insuficiente, etc.) sin depender de la webcam.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence

from backendHelen.server import (
    CLIMA_CONSENSUS_OVERRIDE,
    ClassThreshold,
    ConsensusConfig,
    GestureDecisionEngine,
    GestureMetrics,
    GLOBAL_MIN_SCORE,
    MODEL_LABEL_ALIASES,
    TRACKED_GESTURES,
)
from backendHelen.tf_gesture_classifier import TensorFlowSequenceGestureClassifier

LOGGER = logging.getLogger("helen.debug_lstm_offline")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

DEFAULT_SEQUENCE_LEN = 96
DEFAULT_FEATURE_DIM = 42


def _default_model_dir(repo_root: Path) -> Path:
    base = repo_root / "Hellen_model_TF" / "video_gesture_model" / "data" / "models"
    if not base.exists():
        return base
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "saved_model.pb").exists()]
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0] if candidates else base


def _load_sequence(path: Path) -> List[List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de secuencia: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Sequence):
        raise ValueError("La secuencia debe ser una lista de frames")
    return [list(frame) for frame in data]


def _generate_synthetic_sequence(length: int) -> List[List[float]]:
    sequence: List[List[float]] = []
    base_frame = [0.1 * (i % 21) for i in range(DEFAULT_FEATURE_DIM)]
    for _ in range(length):
        jittered = [value + random.uniform(-0.02, 0.02) for value in base_frame]
        sequence.append(jittered)
    return sequence


def _decision_profile(profile: str) -> tuple[dict[str, ClassThreshold] | None, ConsensusConfig, float]:
    if profile.lower() == "debug_lstm":
        thresholds = {
            "Start": ClassThreshold(enter=0.5, release=0.25),
            "Clima": ClassThreshold(enter=0.5, release=0.28),
            "Reloj": ClassThreshold(enter=0.5, release=0.28),
            "Inicio": ClassThreshold(enter=0.5, release=0.28),
        }
        return thresholds, ConsensusConfig(window_size=3, required_votes=1), 0.2
    return None, ConsensusConfig(), GLOBAL_MIN_SCORE


def _pretty_alias(raw_label: str) -> str:
    key = raw_label.lower()
    return MODEL_LABEL_ALIASES.get(key, raw_label)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug LSTM sin cámara")
    parser.add_argument("--model-dir", type=Path, help="Ruta al SavedModel LSTM")
    parser.add_argument(
        "--sequence",
        type=Path,
        help="JSON con lista de frames de 42 o 126 floats; si se omite se genera una secuencia sintética",
    )
    parser.add_argument(
        "--profile",
        choices=["prod", "debug_lstm"],
        default="prod",
        help="Perfil de decisión a usar (prod=umbrales por defecto, debug_lstm=relajado)",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_SEQUENCE_LEN, help="Frames a simular si no hay archivo")

    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or _default_model_dir(repo_root)

    sequence = (
        _load_sequence(args.sequence)
        if args.sequence is not None
        else _generate_synthetic_sequence(max(DEFAULT_SEQUENCE_LEN, args.frames))
    )

    os.environ.setdefault("HELEN_DEBUG", "1")
    LOGGER.info("Cargando modelo TensorFlow desde %s", model_dir)
    classifier = TensorFlowSequenceGestureClassifier(model_dir)

    metrics = GestureMetrics()
    thresholds, consensus, global_min = _decision_profile(args.profile)
    per_label_consensus = {} if args.profile == "debug_lstm" else {"Clima": CLIMA_CONSENSUS_OVERRIDE}
    decision_engine = GestureDecisionEngine(
        metrics=metrics,
        thresholds=thresholds,
        consensus=consensus,
        global_min_score=global_min,
        geometry_verifier=None if args.profile == "debug_lstm" else None,
        per_label_consensus=per_label_consensus,
    )

    LOGGER.info("Procesando %d frames (perfil=%s)", len(sequence), args.profile)
    for idx, frame in enumerate(sequence, start=1):
        prediction = classifier.predict(frame)
        decision = decision_engine.process(
            prediction,
            timestamp=idx * 0.1,
            hint_label=None,
            latency_ms=0.0,
            landmarks=None,
        )
        alias = _pretty_alias(prediction.label)
        LOGGER.info(
            "frame=%03d label=%s alias=%s score=%.3f -> emit=%s reason=%s votos=%d ventana=%.1f", 
            idx,
            prediction.label,
            alias,
            prediction.score,
            decision.emit,
            decision.reason,
            decision.support,
            decision.window_ms,
        )
        if decision.emit:
            LOGGER.info("GESTO EMITIDO: %s (score=%.3f)", decision.label, decision.score)

    snapshot = metrics.generate_report(
        thresholds=decision_engine.thresholds(),
        consensus=decision_engine.consensus_config,
        dataset_info={"path": None, "primary_available": False, "using_fallback": False, "exists": False},
        latency_stats={"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "count": 0},
        label_consensus=decision_engine.consensus_overrides(),
    )
    print("\nResumen de la sesión offline:")
    print(json.dumps({
        "max_scores": snapshot.get("max_scores"),
        "threshold_rejections": snapshot.get("threshold_rejections"),
        "decision_examples": snapshot.get("decision_examples"),
    }, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
