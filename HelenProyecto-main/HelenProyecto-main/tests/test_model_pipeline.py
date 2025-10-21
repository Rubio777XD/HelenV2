import importlib
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'Hellen_model_RN'
DATASET_PATH = MODEL_DIR / 'data1.pickle'

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import backendConexion  # noqa: E402
import helpers  # noqa: E402


def load_dataset():
    with DATASET_PATH.open('rb') as handle:
        return pickle.load(handle)


@pytest.mark.parametrize('module_name', ['mediapipe', 'cv2', 'xgboost'])
def test_required_dependency_available(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        pytest.skip(f'{module_name} no está instalado en el entorno de pruebas')

    module = importlib.import_module(module_name)
    assert module is not None


def test_dataset_structure_and_labels_align():
    dataset = load_dataset()

    assert set(dataset.keys()) == {'data', 'labels'}
    assert len(dataset['data']) == len(dataset['labels']) > 0

    sample = dataset['data'][0]
    assert isinstance(sample, list)
    assert len(sample) == 42

    for raw_label in dataset['labels']:
        try:
            numeric_label = int(raw_label)
        except ValueError:
            assert raw_label in helpers.labels_dict.values()
        else:
            assert numeric_label in helpers.labels_dict


def test_labels_dict_values_are_unique():
    values = list(helpers.labels_dict.values())
    assert len(values) == len(set(values))


def test_post_gesturekey_payload_structure(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200

    def fake_post(url, json):
        captured['url'] = url
        captured['json'] = json
        return DummyResponse()

    monkeypatch.setattr(backendConexion.requests, 'post', fake_post)

    status = backendConexion.post_gesturekey('Clima')

    assert status == 200
    assert captured['url'].endswith('/gestures/gesture-key')
    assert captured['json'] == {'gesture': 'Clima', 'character': 'Clima'}


def test_post_gesturekey_allows_unknown_labels(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200

    def fake_post(url, json):
        captured['json'] = json
        return DummyResponse()

    monkeypatch.setattr(backendConexion.requests, 'post', fake_post)

    backendConexion.post_gesturekey('Desconocido')

    assert captured['json']['gesture'] is None
    assert captured['json']['character'] == 'Desconocido'


@pytest.mark.skipif(importlib.util.find_spec('xgboost') is None, reason='xgboost no disponible')
def test_model_prediction_output_format():
    with (MODEL_DIR / 'model.p').open('rb') as handle:
        model_dict = pickle.load(handle)

    model = model_dict['model']
    dataset = load_dataset()
    samples = dataset['data'][:5]

    predictions = model.predict(samples)
    assert len(predictions) == len(samples)

    known_labels = helpers.labels_dict

    for raw_prediction in predictions:
        if isinstance(raw_prediction, (bytes, bytearray)):
            raw_prediction = raw_prediction.decode()

        if isinstance(raw_prediction, str):
            assert raw_prediction in known_labels.values()
        else:
            index = int(round(float(raw_prediction)))
            assert index in known_labels

