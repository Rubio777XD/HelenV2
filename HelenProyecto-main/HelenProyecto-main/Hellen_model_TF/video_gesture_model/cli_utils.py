"""Utilidades comunes para mejorar la interacción en consola.

El objetivo de este módulo es compartir pequeñas rutinas que muestran las
señas disponibles, su número de videos y menús interactivos que permiten
seleccionar opciones sin tener que recordar rutas manualmente.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import config


GestureInventory = List[Tuple[str, int]]


def _count_videos(path: Path) -> int:
    """Contar la cantidad de clips mp4 dentro de una carpeta concreta."""

    return sum(1 for _ in path.glob("*.mp4"))


def gesture_inventory() -> GestureInventory:
    """Listar las señas registradas y cuántos clips tiene cada una."""

    inventory: GestureInventory = []
    if not config.VIDEOS_DIR.exists():
        return inventory

    for gesture_dir in sorted(config.VIDEOS_DIR.iterdir()):
        if gesture_dir.is_dir():
            inventory.append((gesture_dir.name, _count_videos(gesture_dir)))
    return inventory


def print_inventory_table(inventory: GestureInventory) -> None:
    """Mostrar en consola una tabla amigable con la información de las señas."""

    if not inventory:
        print("Aún no hay señas registradas. Empieza creando una nueva.")
        return

    name_width = max(len(name) for name, _ in inventory)
    count_width = max(len(str(count)) for _, count in inventory)

    header = f"{'#':>3}  {'Seña':<{name_width}}  {'Clips':>{count_width}}"
    print(header)
    print("-" * len(header))
    for idx, (name, count) in enumerate(inventory, start=1):
        print(f"{idx:>3}  {name:<{name_width}}  {count:>{count_width}}")


def prompt_for_single_gesture(existing: GestureInventory, show_table: bool = True) -> str:
    """Solicitar al usuario que seleccione una seña existente o que escriba una nueva."""

    if show_table:
        print_inventory_table(existing)

    while True:
        if existing:
            answer = input(
                "Selecciona el número de la seña existente o escribe un nombre para crear una nueva: "
            ).strip()
        else:
            answer = input("Escribe el nombre de la nueva seña que deseas registrar: ").strip()

        if not answer:
            print("⚠️  Debes ingresar una opción válida.")
            continue

        if answer.isdigit() and existing:
            idx = int(answer) - 1
            if 0 <= idx < len(existing):
                return existing[idx][0]
            print("⚠️  Número fuera de rango, intenta de nuevo.")
            continue

        return answer


def prompt_for_multiple_gestures(existing: GestureInventory) -> List[str]:
    """Permitir que el usuario seleccione una o varias señas para procesarlas."""

    if not existing:
        raise RuntimeError("No hay señas registradas. Captura algunos videos primero.")

    print("Señas disponibles para procesar:")
    print_inventory_table(existing)
    print("Ingresa los números separados por coma, '*' para seleccionar todas o escribe los nombres directamente.")

    names = {name.lower(): name for name, _ in existing}

    while True:
        answer = input("Tu selección: ").strip()
        if not answer:
            print("⚠️  Debes ingresar al menos una opción.")
            continue

        if answer in {"*", "todos", "todas"}:
            return [name for name, _ in existing]

        selected: List[str] = []
        valid = True
        for token in (part.strip() for part in answer.split(",")):
            if not token:
                continue
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(existing):
                    selected.append(existing[idx][0])
                else:
                    print(f"⚠️  El número {token} no corresponde a ninguna seña.")
                    valid = False
                    break
            else:
                normalised = names.get(token.lower())
                if normalised:
                    selected.append(normalised)
                else:
                    print(f"⚠️  La seña '{token}' no existe. Selecciónala por número o crea los videos primero.")
                    valid = False
                    break

        if valid and selected:
            # Eliminar duplicados manteniendo el orden de selección.
            seen = set()
            ordered = []
            for item in selected:
                if item not in seen:
                    seen.add(item)
                    ordered.append(item)
            return ordered

        print("Intenta nuevamente con una selección válida.")


def list_saved_models() -> Sequence[Path]:
    """Obtener los modelos guardados ordenados por fecha de modificación (reciente primero)."""

    if not config.MODELS_DIR.exists():
        return []

    models = [
        path for path in config.MODELS_DIR.iterdir() if path.is_dir() and (path / "saved_model.pb").exists()
    ]
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return models


def prompt_for_model_dir(models: Sequence[Path]) -> Path:
    """Solicitar al usuario que elija uno de los modelos guardados."""

    if not models:
        raise RuntimeError(
            "No se encontraron modelos entrenados. Ejecuta el entrenamiento antes de usar la detección en tiempo real."
        )

    print("Modelos disponibles:")
    for idx, model_path in enumerate(models, start=1):
        stat = model_path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        file_count = sum(1 for _ in model_path.glob("*"))
        print(f"{idx:>3}. {model_path.name} (archivos: {file_count}, modificado: {modified})")

    while True:
        answer = input("Selecciona el número del modelo a utilizar: ").strip()
        if not answer:
            print("⚠️  Debes ingresar un valor.")
            continue
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print("⚠️  Selección inválida, intenta de nuevo.")


def summarise_distribution(labels: Iterable[int]) -> str:
    """Crear una cadena que indique cuántas muestras hay por clase."""

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    parts = [f"{label}: {count}" for label, count in sorted(counts.items())]
    return ", ".join(parts)

