"""Helper utilities for interacting with AWS S3.

These functions are not executed automatically but provide a ready-to-use
interface once AWS credentials are configured via environment variables or the
standard AWS CLI configuration. They support uploading datasets, models and
training artefacts to dedicated buckets.

Conjunto de utilidades para subir datasets y modelos al almacenamiento S3 de AWS
cuando el proyecto se conecte a la nube.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

# Imports robustos: paquete o script directo
try:
    from . import config
except Exception:
    import config  # type: ignore


s3_client = boto3.client("s3", region_name=config.AWS_REGION)


def upload_files(paths: Iterable[Path], bucket: str, prefix: str) -> None:
    """Subir una lista de archivos al bucket indicado respetando el prefijo."""
    for path in paths:
        if not path.exists():
            print(f"⚠️  Se omitió {path} porque no existe.")
            continue
        # Aseguramos separadores POSIX en la clave S3
        key = f"{prefix.rstrip('/')}/{path.name}"
        try:
            s3_client.upload_file(str(path), bucket, key)
            print(f"⬆️  {path} -> s3://{bucket}/{key}")
        except (BotoCoreError, NoCredentialsError) as exc:
            raise RuntimeError("No se pudo cargar el archivo a S3. Verifica tus credenciales.") from exc


def upload_dataset(dataset_name: str = "gesture_dataset") -> None:
    """Subir el dataset comprimido y su mapa de etiquetas a S3."""
    dataset_path = config.FEATURES_DIR / f"{dataset_name}.npz"
    label_path = config.FEATURES_DIR / f"{dataset_name}_labels.json"
    upload_files([dataset_path, label_path], config.S3_DATASET_BUCKET, config.S3_DATASET_PREFIX)


def upload_model(model_dir: Path) -> None:
    """Recorrer un directorio SavedModel y subir cada archivo preservando la estructura."""
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"El directorio del modelo {model_dir} no existe o no es una carpeta.")
    files = [p for p in model_dir.rglob("*") if p.is_file()]
    for file_path in files:
        relative_path = file_path.relative_to(model_dir)
        key = f"{config.S3_MODEL_PREFIX.rstrip('/')}/{model_dir.name}/{relative_path.as_posix()}"
        try:
            s3_client.upload_file(str(file_path), config.S3_MODELS_BUCKET, key)
            print(f"⬆️  {file_path} -> s3://{config.S3_MODELS_BUCKET}/{key}")
        except (BotoCoreError, NoCredentialsError) as exc:
            raise RuntimeError("No se pudo cargar el modelo a S3. Verifica tus credenciales.") from exc


def parse_args() -> argparse.Namespace:
    """Crear la interfaz de subcomandos para elegir entre dataset o modelo."""
    parser = argparse.ArgumentParser(description="Upload gesture datasets or models to AWS S3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Sube un dataset de landmarks a S3")
    dataset_parser.add_argument("--name", default="gesture_dataset")

    model_parser = subparsers.add_parser("model", help="Sube un modelo entrenado a S3")
    model_parser.add_argument("model_dir", type=Path, help="Directorio del modelo guardado")

    return parser.parse_args()


def main() -> None:
    """Punto de entrada que decide qué recurso subir según el subcomando."""
    args = parse_args()
    if args.command == "dataset":
        upload_dataset(args.name)
    elif args.command == "model":
        upload_model(args.model_dir)


if __name__ == "__main__":
    main()