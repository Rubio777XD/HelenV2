# CHANGELOG

## 2024-XX-XX
- Ajuste de compatibilidad: todos los scripts y requisitos fijan Python 3.10 y TensorFlow 2.15 con `py -3.10`/`python3.10`.
- Aceleración del pipeline LSTM: ventana de 32 frames (rellenada a 96), `poll_interval` de 0.03s, `frame_stride`=1 y consenso rápido (ventana 3, voto mínimo 1).
- Documentación actualizada (README y guía Windows) para los nuevos tiempos de detección y comandos.

## 2024-XX-XX
- Forzado el backend LSTM como modo único, ignorando solicitudes de XGBoost con warnings centralizados.
- Limpieza de scripts: nuevos `run_windows_lstm.bat`, `run_pi5_lstm.sh` y `verify_tf_model.py`; se eliminaron scripts legacy.
- Documentación renovada en `README.md` para instalación y ejecución en Windows y Raspberry Pi.
- Ajustes en el clasificador TensorFlow para manejar 42 o 126 features y mantener el buffer de 96 frames.
- Perfil `debug_lstm` añadido con umbrales relajados y logging detallado (`HELEN_DEBUG=1`), más reporte de sesión extendido con ejemplos de decisiones y máximos por clase.
- Nuevo script `scripts/debug_lstm_offline.py` para reproducir inferencias LSTM sin cámara.
