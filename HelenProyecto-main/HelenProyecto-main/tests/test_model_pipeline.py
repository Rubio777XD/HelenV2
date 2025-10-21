import json
import pickle
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'Hellen_model_RN'
DATASET_PATH = MODEL_DIR / 'data1.pickle'

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import backendConexion  # noqa: E402
import helpers  # noqa: E402
from simple_classifier import SimpleGestureClassifier, SyntheticGestureStream  # noqa: E402


def load_dataset():
    with DATASET_PATH.open('rb') as handle:
        return pickle.load(handle)


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


def test_simple_classifier_predicts_known_labels():
    classifier = SimpleGestureClassifier(DATASET_PATH)
    dataset = load_dataset()

    sample = dataset['data'][5]
    prediction = classifier.predict(sample)

    assert prediction.label in helpers.labels_dict.values()
    assert 0.0 <= prediction.score <= 1.0


def test_synthetic_stream_introduces_variability():
    stream = SyntheticGestureStream(DATASET_PATH, jitter=0.05)
    first_features, first_label = stream.next()
    second_features, second_label = stream.next()

    assert len(first_features) == 42
    assert first_label in helpers.labels_dict.values()
    assert second_label in helpers.labels_dict.values()
    assert any(abs(a - b) > 0 for a, b in zip(first_features, second_features))


def test_post_gesturekey_payload_structure(monkeypatch):
    captured = {}

    class DummyConnection:
        def __init__(self, host, port, timeout):
            captured['address'] = (host, port)
            captured['timeout'] = timeout

        def request(self, method, path, body=None, headers=None):
            captured['method'] = method
            captured['path'] = path
            captured['body'] = body
            captured['headers'] = headers

        def getresponse(self):
            return types.SimpleNamespace(status=200, read=lambda: b'')

        def close(self):
            captured['closed'] = True

    monkeypatch.setattr(backendConexion, 'http_client', types.SimpleNamespace(HTTPConnection=DummyConnection))

    status = backendConexion.post_gesturekey('Clima', score=0.85, session_id='abc123')

    assert status == 200
    assert captured['method'] == 'POST'
    assert captured['path'].endswith('/gestures/gesture-key')
    assert captured.get('closed') is True

    payload = json.loads(captured['body'].decode('utf-8'))
    assert payload['gesture'] == 'Clima'
    assert payload['character'] == 'Clima'
    assert payload['score'] == pytest.approx(0.85)
    assert payload['session_id'] == 'abc123'
    assert payload['sequence'] == 1
    assert 'timestamp' in payload
    assert 'latency_ms' in payload


def test_post_gesturekey_allows_unknown_labels(monkeypatch):
    captured = {}

    class DummyConnection:
        def __init__(self, host, port, timeout):
            pass

        def request(self, method, path, body=None, headers=None):
            captured['body'] = body

        def getresponse(self):
            return types.SimpleNamespace(status=200, read=lambda: b'')

        def close(self):
            pass

    monkeypatch.setattr(backendConexion, 'http_client', types.SimpleNamespace(HTTPConnection=DummyConnection))

    backendConexion.post_gesturekey('Desconocido')

    payload = json.loads(captured['body'].decode('utf-8'))
    assert payload['gesture'] is None
    assert payload['character'] == 'Desconocido'
    assert payload['sequence'] >= 1
