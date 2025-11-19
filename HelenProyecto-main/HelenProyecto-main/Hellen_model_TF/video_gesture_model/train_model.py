"""Train a TensorFlow model using landmark sequences extracted from gesture videos.

Script encargado de cargar el dataset de landmarks, definir la red en TensorFlow
y guardar el modelo entrenado junto con su historial y etiquetas.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

# Imports robustos: funcionan como paquete o como script directo
try:
    from . import config
    from .cli_utils import summarise_distribution
except Exception:
    import config  # type: ignore
    from cli_utils import summarise_distribution  # type: ignore


def parse_args() -> argparse.Namespace:
    """Configurar los parámetros de entrenamiento recibidos por consola."""
    parser = argparse.ArgumentParser(description="Train a gesture recognition model with TensorFlow")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.FEATURES_DIR / "gesture_dataset.npz",
        help="Path to the .npz file containing arrays X and y.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=config.FEATURES_DIR / "gesture_dataset_labels.json",
        help="JSON file with the gesture to index mapping.",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Número de épocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=24, help="Tamaño del batch durante el entrenamiento.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
        help="Tasa de aprendizaje del optimizador Adam.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.25,
        help="Porcentaje de muestras reservado para validación (0-1).",
    )
    parser.add_argument(
        "--lstm-units",
        nargs=2,
        type=int,
        default=[160, 96],
        metavar=("L1", "L2"),
        help="Cantidad de neuronas en las dos capas LSTM (primera y segunda).",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=96,
        help="Número de neuronas en la capa densa final previa a la salida.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.45,
        help="Proporción de Dropout aplicada después de cada LSTM.",
    )
    return parser.parse_args()


def load_data(dataset_path: Path, validation_split: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Leer el archivo .npz, mezclar las muestras y separarlas en train/validación."""
    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return (X_train, y_train), (X_val, y_val)


def build_model(
    num_classes: int,
    sequence_length: int,
    feature_dim: int,
    lstm_units: Tuple[int, int],
    dense_units: int,
    dropout: float,
) -> tf.keras.Model:
    """Crear la arquitectura LSTM que procesa secuencias de landmarks."""
    inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="landmarks")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(lstm_units[0], return_sequences=True)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(lstm_units[1])(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main() -> None:
    """Ejecutar el entrenamiento y guardar los artefactos generados."""
    args = parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas: {args.labels}")

    label_map = json.loads(args.labels.read_text(encoding="utf-8"))
    idx_to_label = {idx: gesture for gesture, idx in label_map.items()}

    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, args.validation_split)
    sequence_length = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = int(np.max(np.concatenate([y_train, y_val])) + 1)

    model = build_model(
        num_classes=num_classes,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        lstm_units=tuple(args.lstm_units),
        dense_units=args.dense_units,
        dropout=args.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Resumen útil
    train_samples = len(X_train)
    val_samples = len(X_val)
    total_samples = train_samples + val_samples

    def map_distribution(text: str) -> str:
        parts = []
        for segment in text.split(", "):
            if not segment:
                continue
            label_str, count_str = segment.split(": ")
            label_idx = int(label_str)
            parts.append(f"{idx_to_label.get(label_idx, label_idx)}: {count_str}")
        return ", ".join(parts)

    print("Resumen del conjunto de datos:")
    print(f"   • Total de muestras: {total_samples}")
    print(f"   • Entrenamiento: {train_samples} | Validación: {val_samples}")
    train_dist = map_distribution(summarise_distribution(y_train))
    val_dist = map_distribution(summarise_distribution(y_val))
    print(f"   • Distribución etiquetas (train): {train_dist}")
    print(f"   • Distribución etiquetas (val): {val_dist}")
    print("Hiperparámetros seleccionados:")
    print(f"   • Épocas={args.epochs}, Batch={args.batch_size}, LR={args.learning_rate}, Dropout={args.dropout}")
    print(f"   • LSTM={tuple(args.lstm_units)}, Dense={args.dense_units}")

    # ==== Callbacks (Keras 3 friendly) ====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        # Checkpoint de MEJORES PESOS (ligero, evita requerir .keras)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / f"best_weights_{timestamp}.weights.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # Registra los logs para tensorboard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.LOGS_DIR / datetime.now().strftime("logs_%Y%m%d_%H%M%S"))
        ),
        # Detiene el entrenamiento si no hay mejora en validación durante 10 épocas
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy", mode="max"
        ),
        # Reduce la tasa de aprendizaje cuando la val_loss deja de mejorar
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,       # Reduce a la mitad el LR
            patience=4,        # Espera 4 épocas sin mejora antes de reducir
            min_lr=1e-6,       # Nunca baja más allá de este valor
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # ==== Guardado final en SavedModel (carpeta) ====
    model_dir = config.MODELS_DIR / f"gesture_model_{timestamp}"
    model.export(model_dir)  # SavedModel en directorio
    print(f"💾 Modelo (SavedModel) guardado en {model_dir}")

    # # (Opcional) Guarda también los pesos finales como archivo
    # final_weights = model_dir / "final.weights.h5"
    # model.save_weights(final_weights)
    # print(f"🧱 Pesos finales guardados en {final_weights}")

    # (Opcional) Archivo único Keras 3, por portabilidad
    # model.save(model_dir / "final_model.keras")
    # print(f"📦 Modelo completo en archivo único guardado en {model_dir / 'final_model.keras'}")

    # Historial y etiquetas
    history_path = model_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)
    print(f"📝 Historial de entrenamiento guardado en {history_path}")

    labels_dest = model_dir / "labels.json"
    labels_dest.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🗂️  Copia del mapa de etiquetas guardada en {labels_dest}")


if __name__ == "__main__":
    main()