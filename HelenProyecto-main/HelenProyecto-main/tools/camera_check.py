#!/usr/bin/env python3
"""Utility to enumerate and validate camera sources on Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional, Sequence

from backendHelen import camera_probe

LOGGER = logging.getLogger("helen.camera_check")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _print_json(data: Dict[str, Any]) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _format_selection(selection: camera_probe.CameraSelection) -> str:
    target = selection.device or selection.index
    return (
        f"backend={selection.backend} target={target} "
        f"mode={selection.mode_name} latency={selection.latency_ms:.1f}ms"
    )


def _cmd_list(args: argparse.Namespace) -> int:
    sources = camera_probe.list_sources()
    if args.json:
        _print_json(sources)
        return 0

    v4l2 = sources.get("v4l2", [])
    libcamera = sources.get("libcamera", [])

    if not v4l2 and not libcamera:
        print("No se detectaron dispositivos V4L2 ni libcamera.")
        return 1

    if v4l2:
        print("Fuentes V4L2 detectadas:")
        for entry in v4l2:
            target = entry.get("path") or entry.get("index")
            print(f"  - {target}: {entry.get('label')} ({entry.get('kind')})")
    if libcamera:
        print("Fuentes libcamera detectadas:")
        for entry in libcamera:
            print(f"  - {entry.get('id')}: {entry.get('label')}")
    return 0


def _cmd_auto(args: argparse.Namespace) -> int:
    selection = camera_probe.ensure_camera_selection(force=True, logger=LOGGER)
    if not selection:
        print("FAIL: no se pudo determinar una cámara funcional. Revisa los logs en reports/logs/pi.")
        return 1

    payload = selection.to_dict()
    payload["status"] = "PASS"
    if args.json:
        _print_json(payload)
    else:
        print(f"PASS: {_format_selection(selection)}")
    return 0


def _cmd_device(args: argparse.Namespace) -> int:
    try:
        width, height = camera_probe.parse_resolution(args.res)
    except ValueError as error:
        print(f"Resolución inválida: {error}")
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
        "mode": result.mode.label,
        "latency_ms": result.latency_ms,
        "frames_sampled": result.frames_sampled,
        "resolution": result.resolution,
        "fps": result.fps,
        "success": result.success,
        "reason": result.reason,
    }

    if args.json:
        _print_json(payload)
    else:
        status = "PASS" if result.success else "FAIL"
        message = (
            f"{status}: backend={result.backend} frames={result.frames_sampled} "
            f"latency={result.latency_ms:.1f}ms"
        )
        if result.reason and not result.success:
            message += f" reason={result.reason}"
        print(message)
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe HELEN camera sources")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Enumerar dispositivos V4L2/libcamera")
    group.add_argument("--auto", action="store_true", help="Ejecutar el auto-probe y mostrar la selección cacheada")
    group.add_argument("--device", help="Ruta (/dev/videoX) o índice numérico a probar")
    parser.add_argument("--res", default="640x480", help="Resolución objetivo (por defecto 640x480)")
    parser.add_argument("--fps", type=int, default=30, help="Frames por segundo objetivo")
    parser.add_argument("--json", action="store_true", help="Emitir resultados en JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        return _cmd_list(args)
    if args.auto:
        return _cmd_auto(args)
    if args.device is not None:
        return _cmd_device(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
