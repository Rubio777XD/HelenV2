# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the HELEN desktop bundle."""

import pathlib

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / 'helen'
MODEL_DIR = PROJECT_ROOT / 'Hellen_model_RN'
BACKEND_PACKAGE = 'backendHelen'

hiddenimports = []
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('xgboost')

_datas = []
_datas += collect_data_files(BACKEND_PACKAGE, includes=['templates/*'])
_datas += [
    (str(MODEL_DIR / 'model.p'), 'Hellen_model_RN'),
    (str(MODEL_DIR / 'model.json'), 'Hellen_model_RN'),
    (str(MODEL_DIR / 'data1.pickle'), 'Hellen_model_RN'),
]
_datas += Tree(str(FRONTEND_DIR), prefix='helen')

analysis = Analysis(
    ['backendHelen/__main__.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=None)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='helen-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='helen-backend',
)
