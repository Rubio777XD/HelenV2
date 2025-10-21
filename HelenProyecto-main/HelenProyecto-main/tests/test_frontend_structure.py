import re
from pathlib import Path
from typing import List, Set

import pytest

from backendHelen.server import ACTIVATION_ALIASES as BACKEND_ACTIVATION_ALIASES


REPO_ROOT = Path(__file__).resolve().parents[1]
HELEN_DIR = REPO_ROOT / 'helen'
BACKEND_DIR = REPO_ROOT / 'backendHelen'


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def extract_script_sources(html: str) -> List[str]:
    return re.findall(r'<script[^>]+src="([^"]+)"', html)


def extract_activation_aliases(actions_js: str) -> Set[str]:
    aliases_match = re.search(r'const ACTIVATION_ALIASES = \[(.*?)\];', actions_js, re.S)
    assert aliases_match is not None
    return {
        token.strip().strip("'\"")
        for token in aliases_match.group(1).split(',')
        if token.strip()
    }


def assert_socket_stack_order(script_sources):
    targets = [
        'jsSignHandler/SocketIO.js',
        'jsSignHandler/eventConnector.js',
        'jsSignHandler/actions.js',
    ]
    positions = []
    for target in targets:
        for index, source in enumerate(script_sources):
            if target in source:
                positions.append(index)
                break
        else:
            raise AssertionError(f'No se encontró {target} en la pila de scripts')

    assert positions == sorted(positions), 'El orden de los scripts SocketIO/eventConnector/actions es incorrecto'


def test_activation_ring_styles_present():
    globals_css = read_text(HELEN_DIR / 'css' / 'globals.css')

    assert '--activation-ring-duration' in globals_css
    assert '.activation-ring{' in globals_css
    assert '.activation-ring__halo' in globals_css
    assert '@keyframes activation-ring-pulse' in globals_css


@pytest.mark.parametrize(
    'html_path',
    [
        HELEN_DIR / 'index.html',
        HELEN_DIR / 'pages' / 'devices' / 'devices.html',
        HELEN_DIR / 'pages' / 'settings' / 'settings.html',
        HELEN_DIR / 'pages' / 'settings' / 'wifi.html',
        HELEN_DIR / 'pages' / 'settings' / 'help.html',
        HELEN_DIR / 'pages' / 'settings' / 'info.html',
        HELEN_DIR / 'pages' / 'tutorial' / 'tutorial.html',
        HELEN_DIR / 'pages' / 'clock' / 'clock.html',
        HELEN_DIR / 'pages' / 'clock' / 'alarm.html',
        HELEN_DIR / 'pages' / 'clock' / 'timer.html',
        HELEN_DIR / 'pages' / 'clock' / 'stopwatch.html',
        HELEN_DIR / 'pages' / 'weather' / 'weather.html',
    ],
)
def test_globals_css_is_loaded(html_path):
    html = read_text(html_path)
    assert 'globals.css' in html


@pytest.mark.parametrize(
    'html_path',
    [
        HELEN_DIR / 'index.html',
        HELEN_DIR / 'pages' / 'devices' / 'devices.html',
        HELEN_DIR / 'pages' / 'settings' / 'settings.html',
        HELEN_DIR / 'pages' / 'settings' / 'wifi.html',
        HELEN_DIR / 'pages' / 'tutorial' / 'tutorial.html',
        HELEN_DIR / 'pages' / 'clock' / 'clock.html',
        HELEN_DIR / 'pages' / 'weather' / 'weather.html',
    ],
)
def test_socket_stack_order(html_path):
    html = read_text(html_path)
    script_sources = extract_script_sources(html)
    assert_socket_stack_order(script_sources)


def test_wifi_control_identifiers_exist():
    expected_ids = {
        'refresh-wifi', 'scan-label', 'networks-empty', 'wifi-list',
        'password-container', 'wifi-password', 'toggle-password',
        'password-toggle-icon', 'connect-wifi-button', 'connect-loader',
        'connect-text', 'connecting-text', 'wifi-button', 'wifi-modal',
        'close-wifi-modal', 'wifi-status-icon'
    }

    index_ids = set(re.findall(r'id="([^"]+)"', read_text(HELEN_DIR / 'index.html')))
    wifi_page_ids = set(re.findall(r'id="([^"]+)"', read_text(HELEN_DIR / 'pages' / 'settings' / 'wifi.html')))

    for identifier in expected_ids:
        assert identifier in index_ids or identifier in wifi_page_ids, f'Falta el id {identifier} para el control WiFi'


def test_default_socket_url_matches_backend():
    event_connector = read_text(HELEN_DIR / 'jsSignHandler' / 'eventConnector.js')
    server_source = read_text(BACKEND_DIR / 'server.py')

    default_url_match = re.search(r"const DEFAULT_SOCKET_URL = '([^']+)'", event_connector)
    assert default_url_match is not None
    default_url = default_url_match.group(1)

    host_match = re.search(r"host\s*:\s*[^=]+=\s*['\"]([^'\"]+)['\"]", server_source)
    port_match = re.search(r"port\s*:\s*[^=]+=\s*(\d+)", server_source)
    assert host_match is not None and port_match is not None

    host = host_match.group(1)
    if host == '0.0.0.0':
        host = '127.0.0.1'
    port = port_match.group(1)
    assert default_url == f'http://{host}:{port}'


def test_frontend_gesture_mappings_cover_model_labels():
    from Hellen_model_RN import helpers as model_helpers

    actions_js = read_text(HELEN_DIR / 'jsSignHandler' / 'actions.js')

    actions_match = re.search(r'const gestureActions = \{(.*?)\};', actions_js, re.S)
    assert actions_match is not None
    action_block = actions_match.group(1)
    action_keys = {match.strip() for match in re.findall(r'\s*([a-z]+):\s*\(\)\s*=>', action_block)}

    alias_tokens = extract_activation_aliases(actions_js)

    known_labels = {
        value.lower()
        for value in model_helpers.labels_dict.values()
        if not value.isdigit()
    }

    covered = action_keys.union(alias_tokens)
    assert known_labels.issubset(covered)


def test_activation_trigger_is_exposed():
    event_connector = read_text(HELEN_DIR / 'jsSignHandler' / 'eventConnector.js')
    assert 'window.triggerActivationAnimation = triggerActivationAnimation' in event_connector


def test_backend_activation_aliases_match_frontend():
    actions_js = read_text(HELEN_DIR / 'jsSignHandler' / 'actions.js')
    frontend_aliases = extract_activation_aliases(actions_js)
    backend_aliases = {alias.lower() for alias in BACKEND_ACTIVATION_ALIASES}

    assert frontend_aliases == backend_aliases
