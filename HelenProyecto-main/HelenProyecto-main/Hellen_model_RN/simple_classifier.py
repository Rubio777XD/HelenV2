"""Lightweight gesture classifier used for the offline HELEN environment.

The original project relies on an XGBoost model that cannot be deserialised in
this execution environment because the binary dependency is not available.
This module provides a deterministic nearest-centroid classifier that is fast
enough for real-time inference while keeping the original gesture taxonomy.

The classifier is purposely simple: it computes one centroid per label from the
pre-recorded dataset and classifies new samples based on the closest centroid.
The distance to the centroid is converted into a pseudo-probability score in
the ``[0.0, 1.0]`` range so the frontend can render confidence indicators and
debounce gestures.

Even though this model is much smaller than the original XGBoost version it
maintains the same input/output contract:

* Input: 42 floating point features representing normalised landmark
  coordinates.
* Output: gesture label, canonical character, and a confidence score.

The implementation only depends on the Python standard library, making it safe
to ship inside the repository without external wheels.
"""

from __future__ import annotations

import math
import pickle
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

try:  # pragma: no cover - compatibilidad con ejecuciones como script
    from .helpers import labels_dict
except ImportError:  # pragma: no cover
    from helpers import labels_dict


FeatureVector = Sequence[float]


def _to_canonical_label(raw_label: str | int) -> str:
    """Normalise dataset labels into the public gesture name.

    The training dataset mixes numeric indices and already-normalised strings.
    This helper maps any numeric identifier through ``helpers.labels_dict`` so
    the rest of the code can reason purely with human readable gesture names.
    """

    try:
        numeric = int(raw_label)
    except (TypeError, ValueError):
        return str(raw_label)

    return labels_dict.get(numeric, str(raw_label))


def _compute_centroid(samples: Iterable[FeatureVector]) -> List[float]:
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot compute centroid from an empty sample list")

    length = len(samples[0])
    centroid = [0.0] * length
    for sample in samples:
        if len(sample) != length:
            raise ValueError("All feature vectors must share the same length")
        for index, value in enumerate(sample):
            centroid[index] += float(value)

    for index in range(length):
        centroid[index] /= float(len(samples))

    return centroid


def _euclidean_distance(a: FeatureVector, b: FeatureVector) -> float:
    if len(a) != len(b):
        raise ValueError("Feature vectors must have the same dimensionality")

    total = 0.0
    for idx in range(len(a)):
        diff = float(a[idx]) - float(b[idx])
        total += diff * diff
    return math.sqrt(total)


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


class SimpleGestureClassifier:
    """Deterministic nearest-centroid classifier for HELEN gestures."""

    def __init__(self, dataset_path: Path) -> None:
        self._lock = threading.Lock()

        with dataset_path.open("rb") as handle:
            data_dict = pickle.load(handle)

        raw_data: List[FeatureVector] = data_dict["data"]
        raw_labels: List[str] = data_dict["labels"]

        by_label: Dict[str, List[FeatureVector]] = {}
        for features, label in zip(raw_data, raw_labels):
            canonical = _to_canonical_label(label)
            by_label.setdefault(canonical, []).append(tuple(float(v) for v in features))

        if not by_label:
            raise ValueError("Dataset did not contain any labelled samples")

        self._centroids: Dict[str, List[float]] = {
            label: _compute_centroid(samples) for label, samples in by_label.items()
        }

        # Pre-compute the maximum intra-class distance to normalise scores.
        max_distance = 0.0
        for label, samples in by_label.items():
            centroid = self._centroids[label]
            for sample in samples:
                distance = _euclidean_distance(sample, centroid)
                max_distance = max(max_distance, distance)

        # Guard against perfectly uniform samples.
        self._max_distance = max_distance or 1.0

    # ------------------------------------------------------------------
    def predict(self, features: FeatureVector) -> Prediction:
        """Return the best matching gesture label and a confidence score."""

        with self._lock:
            best_label: str | None = None
            best_distance = float("inf")

            for label, centroid in self._centroids.items():
                distance = _euclidean_distance(features, centroid)
                if distance < best_distance:
                    best_label = label
                    best_distance = distance

            assert best_label is not None, "Classifier must have at least one label"

            score = max(0.0, 1.0 - (best_distance / self._max_distance))
            return Prediction(label=best_label, score=score)


class SyntheticGestureStream:
    """Generates gesture samples from the dataset with light jitter.

    The original system consumes live MediaPipe landmarks. Since a physical
    camera is not available in the execution environment we replay the dataset
    in a loop while injecting a small amount of random noise. This keeps the
    predictions dynamic and exercises the full transport stack end-to-end.
    """

    def __init__(self, dataset_path: Path, *, jitter: float = 0.0125) -> None:
        with dataset_path.open("rb") as handle:
            data_dict = pickle.load(handle)

        raw_data: List[FeatureVector] = data_dict["data"]
        raw_labels: List[str] = data_dict["labels"]

        self._samples: List[Tuple[List[float], str]] = []
        for features, label in zip(raw_data, raw_labels):
            canonical = _to_canonical_label(label)
            # ``tuple`` ensures the underlying list is not modified by callers.
            self._samples.append((list(float(v) for v in features), canonical))

        if not self._samples:
            raise ValueError("Dataset did not contain any samples")

        priority = ["Start", "Clima", "Reloj", "Inicio"]
        if any(label in priority for _, label in self._samples):
            prioritized = [sample for sample in self._samples if sample[1] in priority]
            others = [sample for sample in self._samples if sample[1] not in priority]
            prioritized.sort(key=lambda pair: priority.index(pair[1]))
            self._samples = prioritized + others

        self._index = 0
        self._lock = threading.Lock()
        self._jitter = float(jitter)

    def __iter__(self) -> Iterator[Tuple[List[float], str]]:
        while True:
            yield self.next()

    def next(self) -> Tuple[List[float], str]:
        with self._lock:
            features, label = self._samples[self._index]
            self._index = (self._index + 1) % len(self._samples)

        jittered = [self._apply_jitter(value) for value in features]
        return jittered, label

    # ------------------------------------------------------------------
    def _apply_jitter(self, value: float) -> float:
        if self._jitter <= 0:
            return value

        delta = random.uniform(-self._jitter, self._jitter)
        candidate = value + delta
        # Landmarks are normalised to the [0, 1] range.
        return max(0.0, min(1.0, candidate))


__all__ = [
    "SimpleGestureClassifier",
    "SyntheticGestureStream",
    "Prediction",
]

