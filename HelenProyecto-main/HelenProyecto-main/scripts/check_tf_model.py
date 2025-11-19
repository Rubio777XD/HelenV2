"""Verifica la carga del modelo TensorFlow LSTM de HELEN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_BASE = BASE_DIR / "Hellen_model_TF" / "video_gesture_model" / "data" / "models"


def _find_latest_model(base: Path) -> Path:
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "saved_model.pb").exists()]
    if not candidates:
        raise FileNotFoundError(f"No se encontraron modelos TensorFlow en {base}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_labels(model_dir: Path):
    labels_path = model_dir / "labels.json"
    if labels_path.exists():
        try:
            return json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"labels.json no legible: {exc}")
    return None


def main() -> int:
    model_dir = _find_latest_model(MODEL_BASE)
    print(f"Usando modelo: {model_dir}")

    model = tf.saved_model.load(str(model_dir))
    signature = model.signatures.get("serve") or model.signatures.get("serving_default")
    if not signature:
        print("El modelo no tiene firma 'serve' o 'serving_default'")
        return 1

    input_name = next(iter(signature.structured_input_signature[1].keys()))
    output_name = next(iter(signature.structured_outputs.keys()))
    input_spec = signature.structured_input_signature[1][input_name]

    print(f"Entrada esperada: name={input_name} shape={input_spec.shape} dtype={input_spec.dtype}")
    print(f"Salida esperada: name={output_name} shape={signature.structured_outputs[output_name].shape}")

    labels = _load_labels(model_dir)
    if labels:
        print(f"labels.json detectado con {len(labels)} etiquetas")
        for label, idx in sorted(labels.items(), key=lambda item: item[1]):
            print(f"  {idx}: {label}")
    else:
        print("labels.json no encontrado; la inferencia seguirá funcionando con etiquetas genéricas")

    return 0


if __name__ == "__main__":
    sys.exit(main())
