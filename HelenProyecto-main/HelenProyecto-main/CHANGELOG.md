# CHANGELOG

## 2024-11-20
- Compatibilidad fijada en Python 3.10 (shebangs, scripts y documentación) alineada con TensorFlow 2.15.
- Pipeline LSTM equilibrado: `poll_interval`=0.03s, `frame_stride`=1, ventanas de 24/48 frames con padding automático a 96 y umbrales 0.45–0.5 / 0.30 según perfil.
- Perfiles de detección: `fast` (24 frames, 1 voto, umbral 0.45) y `normal` (48 frames, 2 votos, umbral 0.5) seleccionables por variable o CLI.
- Limpieza de parámetros obsoletos de MediaPipe y ajustes de consenso para evitar falsos positivos.
- README/guía Windows actualizados con pasos para cámara en Windows y Raspberry Pi usando Python 3.10.

## 2024-XX-XX
- Forzado el backend LSTM como modo único, ignorando solicitudes de XGBoost con warnings centralizados.
- Limpieza de scripts: nuevos `run_windows_lstm.bat`, `run_pi5_lstm.sh` y `verify_tf_model.py`; se eliminaron scripts legacy.
- Documentación renovada en `README.md` para instalación y ejecución en Windows y Raspberry Pi.
- Ajustes en el clasificador TensorFlow para manejar 42 o 126 features y mantener el buffer de 96 frames.
- Perfil `debug_lstm` añadido con umbrales relajados y logging detallado (`HELEN_DEBUG=1`), más reporte de sesión extendido con ejemplos de decisiones y máximos por clase.
- Nuevo script `scripts/debug_lstm_offline.py` para reproducir inferencias LSTM sin cámara.
