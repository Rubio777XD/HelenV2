# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the HELEN desktop bundle."""

import pathlib
from glob import glob

from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,   # <- para DLLs nativas
)

# --- Rutas robustas (soporta entornos donde __file__ no existe) ---
try:
    SPEC_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    SPEC_DIR = pathlib.Path.cwd()

# Si estamos en .../packaging, la raíz del repo es su padre
PROJECT_ROOT = SPEC_DIR if SPEC_DIR.name != "packaging" else SPEC_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "helen"
MODEL_DIR = PROJECT_ROOT / "Hellen_model_RN"   # carpeta del modelo (nombre exacto)
BACKEND_PACKAGE = "backendHelen"

# Script principal (ruta absoluta) -> lanzador que ejecuta el paquete
MAIN_SCRIPT = (SPEC_DIR / "run_backend.py").resolve()

# --- Dependencias implícitas ---
hiddenimports = []
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("xgboost")
hiddenimports += collect_submodules("backendHelen")

# Forzar submódulos que el pickle del modelo puede requerir:
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("scipy._lib")
hiddenimports += [
    "scipy._lib.array_api_compat",
    "scipy._lib.array_api_compat.numpy",
    "scipy._lib.array_api_compat.numpy.fft",
    "numpy.fft",
]

# --- Archivos a incluir en Analysis (solo pares src/dest) ---
_datas = []
# Plantillas/estáticos del backend
_datas += collect_data_files(BACKEND_PACKAGE, includes=["templates/*", "static/*"])
# Todos los archivos del modelo que empiecen por .p (p, pkl, pickle…)
for file in glob(str(MODEL_DIR / "*.p*")):
    _datas.append((file, "Hellen_model_RN"))

# Assets de MediaPipe (modelos .tflite, configs, etc.)
_datas += collect_data_files("mediapipe")
# (Opcional) Archivos de cv2 si los hooks no los copian: descomenta si hiciera falta
# _datas += collect_data_files("cv2")

# --- DLLs / binarios nativos críticos (cámara, audio, etc.) ---
_binaries = []
_binaries += collect_dynamic_libs("cv2")         # OpenCV (videoio, codecs)
_binaries += collect_dynamic_libs("mediapipe")   # MediaPipe (grafo/ops nativas)
_binaries += collect_dynamic_libs("sounddevice") # PortAudio para micrófono
_binaries += collect_dynamic_libs("xgboost")     # XGBoost (xgboost.dll) ← ¡clave para el modelo!

# --- Análisis ---
analysis = Analysis(
    [str(MAIN_SCRIPT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=_binaries,            # <- incluir DLLs nativas
    datas=_datas,                  # <- (src, dest)
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "pytest"],  # <- evita arrastrarlos innecesariamente
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
    name="helen-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,   # <- mejor sin UPX para evitar problemas/AV con DLLs
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# --- Empaquetado final ---
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    # Frontend completo accesible en /helen
    Tree(str(FRONTEND_DIR), prefix="helen"),
    strip=False,
    upx=False,   # <- idem
    upx_exclude=[],
    name="helen-backend",
)
