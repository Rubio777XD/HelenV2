import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'Hellen_model_RN'

for path in (str(REPO_ROOT), str(MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


if 'requests' not in sys.modules:
    requests_stub = types.ModuleType('requests')

    class RequestException(Exception):
        """Excepción genérica para emular requests.exceptions.RequestException."""

    requests_stub.exceptions = types.SimpleNamespace(RequestException=RequestException)

    def _unavailable_post(*args, **kwargs):
        raise RequestException('requests.post no está disponible en el entorno de pruebas')

    requests_stub.post = _unavailable_post
    sys.modules['requests'] = requests_stub
