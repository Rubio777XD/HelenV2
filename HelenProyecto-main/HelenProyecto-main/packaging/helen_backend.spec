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

# ``collect_submodules`` lanza ImportError si el paquete no está instalado.
# Durante verificaciones locales preferimos mostrar una advertencia en lugar
# de abortar para poder detectar dependencias faltantes en la sección de
# ``REQUIRED_MODULES``.
def safe_collect_submodules(package: str) -> list[str]:
    try:  # pragma: no cover - se ejecuta al congelar
        return collect_submodules(package)
    except Exception as exc:  # pragma: no cover - solo aviso de build
        print(f"[WARN] No se pudieron resolver los submódulos de {package}: {exc}", flush=True)
        return []

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
BACKEND_DIR = PROJECT_ROOT / BACKEND_PACKAGE
FRONTEND_ALT_DIR = PROJECT_ROOT / "frontend"
PRIMARY_DATASET_NAME = "data.pickle"

# Verificación temprana de dependencias críticas para la build
REQUIRED_MODULES = (
    "pickle",
    "xgboost",
    "mediapipe",
    "socketio",
    "flask",
    "flask_socketio",
    "numpy",
    "cv2",
)

missing_modules = []
for module_name in REQUIRED_MODULES:
    try:  # pragma: no cover - ejecución en proceso de build
        __import__(module_name)
    except ModuleNotFoundError:
        missing_modules.append(module_name)

if missing_modules:
    missing = ", ".join(sorted(missing_modules))
    raise RuntimeError(
        "Dependencias faltantes para construir HELEN: {0}. Ejecuta 'pip install -r requirements.txt' antes de compilar.".format(
            missing
        )
    )

# Script principal (ruta absoluta) -> lanzador que ejecuta el paquete
MAIN_SCRIPT = (SPEC_DIR / "run_backend.py").resolve()

# --- Dependencias implícitas ---
hiddenimports = []
hiddenimports += safe_collect_submodules("mediapipe")
hiddenimports += safe_collect_submodules("sklearn")
hiddenimports += safe_collect_submodules("xgboost")
hiddenimports += safe_collect_submodules("backendHelen")
hiddenimports += safe_collect_submodules("Hellen_model_RN")
hiddenimports += safe_collect_submodules("flask_socketio")
hiddenimports += safe_collect_submodules("socketio")
hiddenimports += safe_collect_submodules("engineio")
hiddenimports += safe_collect_submodules("eventlet")
hiddenimports += safe_collect_submodules("werkzeug")
hiddenimports += safe_collect_submodules("simple_websocket")
hiddenimports += safe_collect_submodules("wsproto")
hiddenimports += safe_collect_submodules("bidict")

# Forzar submódulos que el pickle del modelo puede requerir:
hiddenimports += safe_collect_submodules("scipy")
hiddenimports += safe_collect_submodules("scipy._lib")
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

# Metadatos adicionales del modelo (por ejemplo ``model.json``)
for file in glob(str(MODEL_DIR / "*.json")):
    _datas.append((file, "Hellen_model_RN"))

primary_dataset = MODEL_DIR / PRIMARY_DATASET_NAME
primary_entry = (str(primary_dataset), "Hellen_model_RN")
if primary_dataset.exists() and primary_entry not in _datas:
    _datas.append(primary_entry)
elif not primary_dataset.exists():  # pragma: no cover - aviso en build
    print(
        f"[WARN] {PRIMARY_DATASET_NAME} no encontrado en {MODEL_DIR}. La build continuará con los recursos disponibles.",
        flush=True,
    )

# Assets de MediaPipe (modelos .tflite, configs, etc.)
_datas += collect_data_files("mediapipe")

# Configuración y metadatos de runtime (solo si existen)
_config_targets = [
    (PROJECT_ROOT / "config.json", "."),
    (BACKEND_DIR / "config.json", BACKEND_PACKAGE),
    (SPEC_DIR / "requirements-win.txt", "config"),
]

for config_path, destination in _config_targets:
    if config_path.exists():
        _datas.append((str(config_path), destination))
# (Opcional) Archivos de cv2 si los hooks no los copian: descomenta si hiciera falta
# _datas += collect_data_files("cv2")

# --- DLLs / binarios nativos críticos (cámara, audio, etc.) ---
_binaries = []
_binaries += collect_dynamic_libs("cv2")         # OpenCV (videoio, codecs)
_binaries += collect_dynamic_libs("mediapipe")   # MediaPipe (grafo/ops nativas)
_binaries += collect_dynamic_libs("sounddevice") # PortAudio para micrófono
_binaries += collect_dynamic_libs("xgboost")     # XGBoost (xgboost.dll) ← ¡clave para el modelo!

# PyInstaller a veces no detecta ``xgboost.dll`` automáticamente. Verifica de forma
# proactiva que el binario quede empaquetado aunque los hooks fallen.
try:  # pragma: no cover - solo se ejecuta en el proceso de build
    import xgboost  # type: ignore
except Exception:
    pass
else:
    package_dir = pathlib.Path(getattr(xgboost, "__file__", "")).resolve().parent
    candidates = [
        package_dir / "lib" / "xgboost.dll",
        package_dir / "xgboost.dll",
    ]
    existing_sources = {pathlib.Path(src).resolve() for src, _ in _binaries}
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in existing_sources:
            break
        try:
            relative_parent = candidate.parent.relative_to(package_dir)
        except ValueError:
            dest_dir = "xgboost"
        else:
            dest_dir = str(pathlib.Path("xgboost") / relative_parent)
        _binaries.append((str(resolved), dest_dir))
        break

# Directorios completos que deben copiarse intactos
asset_trees = [Tree(str(FRONTEND_DIR), prefix="helen")]

backend_static_dir = BACKEND_DIR / "static"
backend_templates_dir = BACKEND_DIR / "templates"

if backend_templates_dir.exists():
    asset_trees.append(Tree(str(backend_templates_dir), prefix=f"{BACKEND_PACKAGE}/templates"))
if backend_static_dir.exists():
    asset_trees.append(Tree(str(backend_static_dir), prefix=f"{BACKEND_PACKAGE}/static"))
if FRONTEND_ALT_DIR.exists():
    asset_trees.append(Tree(str(FRONTEND_ALT_DIR), prefix="frontend"))

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
    *asset_trees,
    strip=False,
    upx=False,   # <- idem
    upx_exclude=[],
    name="helen-backend",
)
