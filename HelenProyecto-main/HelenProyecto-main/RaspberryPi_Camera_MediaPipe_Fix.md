# HELEN – Raspberry Pi MediaPipe & Cámara

Este documento resume los ajustes introducidos para estabilizar la instalación en Raspberry Pi, la auto-detección de cámara USB/CSI y el redimensionamiento del aro visual en modo kiosko.

## Compatibilidad Python ↔ MediaPipe ↔ OpenCV

| Python | MediaPipe | OpenCV | Notas |
| ------ | --------- | ------ | ----- |
| 3.11.x (aarch64) | `mediapipe==0.10.18` | `opencv-python==4.9.0.80` | Stack soportado y verificado por `setup_pi.sh`. |
| 3.10.x (aarch64) | `mediapipe==0.10.11` | `opencv-python==4.9.0.80` | Se admite solo para instalaciones heredadas; se registrará advertencia. |
| ≥3.12 | _No soportado_ | — | El instalador aborta y solicita instalar Python 3.11 (PEP 668 + ruedas ARM64 disponibles). |

- El script `packaging-pi/setup_pi.sh` crea siempre `.venv`, actualiza `pip`, instala `numpy==1.26.4`, `opencv-python==4.9.0.80` y resuelve la versión adecuada de MediaPipe según el intérprete detectado y la arquitectura (`uname -m`).
- Al finalizar se ejecuta `mediapipe` → `Hands()` como prueba de humo y se escribe un snapshot estructurado en `reports/logs/pi/vision-stack-*.json`.
- En caso de incompatibilidad, el log detalla la causa probable (rueda inexistente, build de OpenCV sin V4L2/GStreamer, etc.) y la acción sugerida.

## Forzar o relajar versiones

1. Para reinstalar el stack exacto ejecuta:
   ```bash
   ./packaging-pi/setup_pi.sh
   ```
2. Si necesitas mantener Python 3.10, exporta `PYTHON=/usr/bin/python3.10` antes de lanzar el script. Se instalará `mediapipe==0.10.11` con advertencia.
3. Ante una actualización futura de MediaPipe, edita el bloque `resolve_mediapipe_spec()` en `setup_pi.sh` y vuelve a ejecutar el script.

Los registros completos de instalación y comprobaciones viven en `reports/logs/pi/setup-*.log` y `reports/logs/pi/vision-stack-*.json`.

## Auto-probe de cámara (USB / CSI)

- El módulo `backendHelen/camera_probe.py` realiza:
  - Enumeración de `/dev/video*` (V4L2) y sensores libcamera (`libcamera-hello --list-cameras`).
  - Validación con OpenCV (`cv2.VideoCapture`) tanto para V4L2 como pipelines GStreamer (`libcamerasrc` y `v4l2src`).
  - Medición de latencia al primer frame válido y descarte de fotogramas negros.
  - Selección prioritaria: 1) CSI/libcamera estable, 2) USB UVC estable, 3) cualquier otro fallback disponible.
  - Cacheo en `reports/config/camera_selection.json` junto con la firma de hardware (para reprobar cuando se cambien dispositivos).
- El backend invoca automáticamente este módulo durante el arranque. Si la cámara falla, se hace un re-probe con backoff (1s → 3s → 5s) antes de degradar a flujo sintético.
- Los resultados de cada prueba quedan en `reports/logs/pi/camera-probe-*.json` y aparecen en `/engine/status` bajo `camera_selection`.

### CLI de verificación

```
python tools/camera_check.py --list
python tools/camera_check.py --auto
python tools/camera_check.py --device /dev/video0 --res 640x480 --fps 30
```

- `--list` muestra las fuentes detectadas.
- `--auto` ejecuta el probe completo, escribe la caché y devuelve `PASS/FAIL`.
- `--device` permite comprobar directamente un índice o ruta concreta; si falla se proporcionan sugerencias (dispositivo ocupado, permisos, pipeline inválido, etc.).

## Integración en el backend

- `HelenRuntime.engine_status()` expone `vision` (stack Python/MediaPipe/OpenCV) y `camera_selection` (backend elegido, resolución, latencia inicial).
- `CameraGestureStream` ahora respeta la pipeline seleccionada (V4L2 o GStreamer), preserva la ruta del dispositivo y reintenta con la alternativa automáticamente si el backend primario falla durante la captura.
- El runtime mantiene la caché en memoria y solo reproba cuando cambia el hardware o al forzar `--auto` desde el CLI.

## Configuración del aro de activación

- El aro se adapta dinámicamente al contenedor activo (`[data-raspberry-fit-root]`), mantiene un margen de seguridad configurable y nunca sobrepasa el contenido (z-index inferior y blur ajustado).
- API disponible en el navegador:
  ```js
  window.HelenActivationRing.configure({
    enabled: true,
    maxScale: 0.92,
    safePadding: 12,
    zIndexBase: -1,
  });
  window.HelenActivationRing.refresh();
  ```
- La posición se recalcula con `ResizeObserver`, `Resize` y `orientationchange`. En modo Windows el aro conserva el comportamiento original.
- Nuevas variables CSS expuestas: `--activation-ring-diameter`, `--activation-ring-safe-padding`, `--activation-ring-blur`, `--activation-ring-radius`.

## Checklist de pruebas en Raspberry Pi (objetivo Pi 4 / Pi 5)

| Resolución | Estado cámara | Ring auto-fit | Notas |
| ---------- | ------------- | ------------- | ----- |
| 800×480 | _Ejecutar `tools/camera_check.py --auto` y revisar `/engine/status`_ | _Verificar que el aro permanece detrás de las tarjetas sin provocar scroll_ | **Pendiente de validar en hardware**. |
| 1024×600 | _Ídem_ | _Cambiar orientación y comprobar reajuste suave_ | **Pendiente de validar en hardware**. |

Pasos recomendados para la verificación en sitio:
1. Ejecutar `./packaging-pi/setup_pi.sh` y revisar `reports/logs/pi/setup-*.log`.
2. Lanzar `python -c "import mediapipe, cv2; print('ok')"` dentro de `.venv`.
3. Ejecutar `python tools/camera_check.py --auto` y guardar la salida (`PASS` esperado).
4. Arrancar `./packaging-pi/run_pi.sh`, comprobar `/engine/status` y confirmar que `camera_selection.capture_backend` refleja V4L2 o GStreamer.
5. Navegar por Home, Clima, Reloj, Dispositivos, Configuración y Tutorial en 800×480 y 1024×600 confirmando ausencia de solapamientos y sin scroll adicional.

> ℹ️ En esta entrega no se han podido capturar evidencias desde hardware real debido al entorno sin periféricos. Use la checklist anterior como guía de validación final.
