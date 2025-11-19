from pathlib import Path
import sys

import importlib.util

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

config_spec = importlib.util.spec_from_file_location("config", PACKAGE_ROOT / "config.py")
config_module = importlib.util.module_from_spec(config_spec)
assert config_spec.loader is not None
config_spec.loader.exec_module(config_module)
sys.modules.setdefault("config", config_module)

from video_gesture_model import train_model


def test_build_model_output_shape_and_layers():
    num_classes = 4
    sequence_length = 10
    feature_dim = 6
    lstm_units = (8, 4)
    dense_units = 5
    dropout = 0.3

    model = train_model.build_model(
        num_classes=num_classes,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
    )

    batch_size = 3
    dummy_input = tf.random.uniform((batch_size, sequence_length, feature_dim))
    outputs = model(dummy_input, training=False)

    assert outputs.shape == (batch_size, num_classes)

    last_layer = model.get_layer("class_probabilities")
    assert isinstance(last_layer, tf.keras.layers.Dense)
    assert last_layer.activation == tf.keras.activations.softmax
    assert last_layer.units == num_classes

    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    assert len(dropout_layers) == 2
    for layer in dropout_layers:
        assert np.isclose(layer.rate, dropout)

    masking_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Masking)]
    assert masking_layers, "El modelo debe incluir una capa Masking para ignorar padding"


def test_load_data_splits_dataset(tmp_path):
    num_samples = 12
    sequence_length = 5
    feature_dim = 2

    X = np.arange(num_samples * sequence_length * feature_dim, dtype=np.float32).reshape(
        num_samples, sequence_length, feature_dim
    )
    y = np.arange(num_samples, dtype=np.int64)

    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, X=X, y=y)

    validation_split = 0.25
    np.random.seed(42)
    (X_train, y_train), (X_val, y_val) = train_model.load_data(dataset_path, validation_split)

    expected_train = int(num_samples * (1 - validation_split))
    expected_val = num_samples - expected_train

    assert X_train.shape == (expected_train, sequence_length, feature_dim)
    assert X_val.shape == (expected_val, sequence_length, feature_dim)
    assert y_train.shape == (expected_train,)
    assert y_val.shape == (expected_val,)

    combined_X = np.concatenate([X_train, X_val], axis=0)
    combined_y = np.concatenate([y_train, y_val], axis=0)

    original_X = X.reshape(num_samples, -1)
    shuffled_X = combined_X.reshape(num_samples, -1)
    assert {tuple(row) for row in original_X} == {tuple(row) for row in shuffled_X}
    assert set(y.tolist()) == set(combined_y.tolist())
