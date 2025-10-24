#!/usr/bin/env python3
"""Auxiliary CLI to diagnose Raspberry Pi cameras without launching HELEN."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional, Sequence

from backendHelen import camera_probe

LOGGER = logging.getLogger("helen.camera_check")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _print_json(data: Dict[str, Any]) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _resolve_target(selection: camera_probe.CameraSelection) -> str:
    if selection.device:
        return selection.device
    if selection.index is not None:
        return f"/dev/video{selection.index}"
    if selection.pipeline:
        return selection.pipeline
    return "<sin-definir>"


def _selection_payload(selection: camera_probe.CameraSelection) -> Dict[str, Any]:
    return {
        "device": _resolve_target(selection),
        "backend": selection.backend,
        "width": int(selection.width or 0),
        "height": int(selection.height or 0),
        "fps": float(selection.fps or 0.0),
        "latency_ms": float(selection.latency_ms or 0.0),
        "pipeline": selection.pipeline,
        "kind": selection.kind,
        "mode": selection.mode_name,
        "source": getattr(selection, "_origin", "probe"),
        "verified": bool(getattr(selection, "_verified", False)),
        "timestamp": selection.probed_at,
    }


def _emit_selection(selection: camera_probe.CameraSelection, *, json_mode: bool, detailed: bool) -> None:
    payload = _selection_payload(selection)
    payload["status"] = "ok"

    if json_mode:
        _print_json(payload)
        return

    status_line = "Estado: OK"
    if payload["source"] == "cache":
        status_line += " (cacheada)"
    if not payload["verified"]:
        status_line = "Estado: Revisar (no validada)"

    print(f"Cámara detectada: {payload['device']}")
    print(f"Resolución: {payload['width']}x{payload['height']}")
    print(f"Backend: {payload['backend']}")
    print(status_line)

    if detailed:
        print(f"FPS: {payload['fps']:.2f}")
        if payload["pipeline"]:
            print(f"Pipeline: {payload['pipeline']}")
        print(f"Latencia: {payload['latency_ms']:.1f} ms")
        print(f"Origen: {payload['source']}")
        print(f"Última validación: {payload['timestamp']}")


def _emit_failure(message: str, *, json_mode: bool) -> None:
    if json_mode:
        _print_json({"status": "error", "message": message})
    else:
        print(message)


def _cmd_list(args: argparse.Namespace) -> int:
    sources = camera_probe.list_sources()
    if args.json:
        _print_json(sources)
        return 0

    v4l2 = sources.get("v4l2", [])
    libcamera = sources.get("libcamera", [])

    if not v4l2 and not libcamera:
        _emit_failure(camera_probe.CAMERA_NOT_FOUND_MESSAGE, json_mode=False)
        return 2

    if v4l2:
        print("Fuentes V4L2 detectadas:")
        for entry in v4l2:
            label = entry.get("label", "<sin etiqueta>")
            target = entry.get("path") or entry.get("index")
            kind = entry.get("kind", "desconocido")
            print(f"  - {target}: {label} [{kind}]")
    if libcamera:
        print("Fuentes rpicam/libcamera detectadas:")
        for entry in libcamera:
            label = entry.get("label", "<sin etiqueta>")
            identifier = entry.get("id") or entry.get("metadata", {}).get("description")
            print(f"  - {identifier}: {label}")
    return 0


def _cmd_auto(args: argparse.Namespace) -> int:
    logger = LOGGER if args.verbose else None
    selection = camera_probe.ensure_camera_selection(
        force=args.force,
        preferred=args.preferred,
        logger=logger,
    )
    if not selection:
        _emit_failure(camera_probe.CAMERA_NOT_FOUND_MESSAGE, json_mode=args.json)
        return 2

    _emit_selection(selection, json_mode=args.json, detailed=args.detailed)
    return 0


def _cmd_cached(args: argparse.Namespace) -> int:
    selection = camera_probe.get_cached_selection()
    if not selection:
        _emit_failure("No existe una selección cacheada en disco.", json_mode=args.json)
        return 1
    _emit_selection(selection, json_mode=args.json, detailed=args.detailed)
    return 0


def _cmd_device(args: argparse.Namespace) -> int:
    try:
        width, height = camera_probe.parse_resolution(args.res)
    except ValueError as error:
        _emit_failure(f"Resolución inválida: {error}", json_mode=args.json)
        return 2

    result = camera_probe.probe_specific_device(
        args.device,
        width=width,
        height=height,
        fps=args.fps,
    )

    payload = {
        "device": args.device,
        "backend": result.backend,
        "resolution": result.resolution,
        "fps": result.fps,
        "latency_ms": result.latency_ms,
        "frames_sampled": result.frames_sampled,
        "success": result.success,
        "reason": result.reason,
    }

    if args.json:
        payload["status"] = "ok" if result.success else "error"
        _print_json(payload)
        return 0 if result.success else 1

    target = args.device
    res_width, res_height = result.resolution or (width, height)
    status_line = "Estado: OK" if result.success else f"Estado: ERROR ({result.reason or 'sin-detalle'})"

    print(f"Cámara detectada: {target}")
    print(f"Resolución: {res_width}x{res_height}")
    print(f"Backend: {result.backend}")
    print(status_line)
    if args.detailed:
        print(f"FPS: {result.fps:.2f}")
        print(f"Latencia: {result.latency_ms:.1f} ms")
        print(f"Frames válidos: {result.frames_sampled}")
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnóstico rápido de cámaras HELEN")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Enumerar dispositivos V4L2/rpicam disponibles")
    group.add_argument("--auto", action="store_true", help="Ejecutar la autodetección y validar la cámara preferida")
    group.add_argument("--cached", action="store_true", help="Mostrar únicamente la cámara cacheada si existe")
    group.add_argument("--device", help="Ruta (/dev/videoX) o índice numérico a probar manualmente")
    parser.add_argument("--res", default="640x480", help="Resolución objetivo para pruebas manuales (por defecto 640x480)")
    parser.add_argument("--fps", type=int, default=30, help="FPS objetivo para pruebas manuales (por defecto 30)")
    parser.add_argument("--force", action="store_true", help="Forzar reproceso completo en modo auto")
    parser.add_argument("--preferred", help="Forzar backend/dispositivo preferido en modo auto")
    parser.add_argument("--json", action="store_true", help="Emitir resultados en JSON")
    parser.add_argument("--detailed", action="store_true", help="Mostrar métricas adicionales en la salida en texto")
    parser.add_argument("--verbose", action="store_true", help="Habilitar logs de depuración durante el auto-probe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        return _cmd_list(args)
    if args.auto:
        return _cmd_auto(args)
    if args.cached:
        return _cmd_cached(args)
    if args.device is not None:
        return _cmd_device(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
