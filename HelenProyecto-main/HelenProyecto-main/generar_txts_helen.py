import os
from typing import Iterable, List

FRONTEND_EXTENSIONS = [".html", ".css", ".js"]
FRONTEND_DIR_HINTS = ["templates", "static", "frontend", "jsFrontend"]

BACKEND_EXTENSIONS = [".py"]
BACKEND_DIR_HINTS = ["backend", "server", "api"]

MODEL_EXTENSIONS = [".py", ".ipynb", ".tf"]
MODEL_NAME_TOKEN = "TF"

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".exe",
    ".zip",
    ".tar",
    ".gz",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".mkv",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".svg",
    ".webp",
    ".bmp",
    ".bin",
    ".obj",
    ".so",
    ".dll",
    ".dylib",
    ".class",
    ".jar",
    ".psd",
    ".ttf",
    ".woff",
    ".woff2",
    ".eot",
    ".otf",
}

OUTPUT_FILES = {"frontend.txt", "backend.txt", "modelo.txt"}


def path_contains_hint(path_parts: Iterable[str], hints: List[str]) -> bool:
    lower_parts = [part.lower() for part in path_parts]
    lower_hints = [hint.lower() for hint in hints]
    return any(hint in part for part in lower_parts for hint in lower_hints)


def is_frontend_file(rel_path: str, extension: str) -> bool:
    if extension in FRONTEND_EXTENSIONS:
        return True
    parts = rel_path.replace("\\", "/").split("/")
    return path_contains_hint(parts, FRONTEND_DIR_HINTS)


def is_backend_file(rel_path: str, extension: str) -> bool:
    if extension in BACKEND_EXTENSIONS:
        return True
    parts = rel_path.replace("\\", "/").split("/")
    return path_contains_hint(parts, BACKEND_DIR_HINTS)


def is_model_file(filename: str, extension: str) -> bool:
    return MODEL_NAME_TOKEN in filename and extension in MODEL_EXTENSIONS


def escribir_salida(nombre_archivo: str, rutas: List[str], base_dir: str) -> None:
    output_path = os.path.join(base_dir, nombre_archivo)
    with open(output_path, "w", encoding="utf-8") as salida:
        for ruta in sorted(rutas):
            salida.write(f"({ruta})\n")
            archivo_absoluto = os.path.join(base_dir, ruta)
            try:
                with open(archivo_absoluto, "r", encoding="utf-8", errors="ignore") as archivo:
                    contenido = archivo.read()
            except (OSError, UnicodeDecodeError):
                continue
            salida.write(contenido)
            if not contenido.endswith("\n"):
                salida.write("\n")
            salida.write("\n")


def main() -> None:
    base_dir = os.getcwd()
    frontend_rutas: List[str] = []
    backend_rutas: List[str] = []
    modelo_rutas: List[str] = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for nombre in filenames:
            if nombre in OUTPUT_FILES:
                continue
            extension = os.path.splitext(nombre)[1].lower()
            if extension in BINARY_EXTENSIONS:
                continue

            ruta_completa = os.path.join(dirpath, nombre)
            ruta_relativa = os.path.relpath(ruta_completa, base_dir)

            if is_model_file(nombre, extension):
                modelo_rutas.append(ruta_relativa)
                continue

            if is_frontend_file(ruta_relativa, extension):
                frontend_rutas.append(ruta_relativa)
                continue

            if is_backend_file(ruta_relativa, extension):
                backend_rutas.append(ruta_relativa)

    escribir_salida("frontend.txt", frontend_rutas, base_dir)
    escribir_salida("backend.txt", backend_rutas, base_dir)
    escribir_salida("modelo.txt", modelo_rutas, base_dir)


if __name__ == "__main__":
    main()
