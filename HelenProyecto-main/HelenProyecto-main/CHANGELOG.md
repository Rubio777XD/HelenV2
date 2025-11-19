# CHANGELOG

## 2024-XX-XX
- Forzado el backend LSTM como modo único, ignorando solicitudes de XGBoost con warnings centralizados.
- Limpieza de scripts: nuevos `run_windows_lstm.bat`, `run_pi5_lstm.sh` y `verify_tf_model.py`; se eliminaron scripts legacy.
- Documentación renovada en `README.md` para instalación y ejecución en Windows y Raspberry Pi.
- Ajustes en el clasificador TensorFlow para manejar 42 o 126 features y mantener el buffer de 96 frames.
