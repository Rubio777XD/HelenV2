from pathlib import Path


def test_shell_scripts_use_lf_endings():
    repo_root = Path(__file__).resolve().parents[1]
    for script in repo_root.rglob('*.sh'):
        data = script.read_bytes()
        assert b'\r' not in data, f"El script {script} contiene finales de línea CRLF"
