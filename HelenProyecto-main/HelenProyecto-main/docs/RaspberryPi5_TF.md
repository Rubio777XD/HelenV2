# HELEN en Raspberry Pi 5 con TensorFlow LSTM

Esta guía resume los pasos mínimos para ejecutar HELEN usando el backend LSTM por defecto en Raspberry Pi OS (Bookworm, 64 bits).

## 1. Preparar el entorno

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev git chromium-browser
cd ~/HelenProyecto-main/HelenProyecto-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backendHelen/requirements.txt
```

## 2. Ejecutar manualmente

```bash
./scripts/start-helen-backend-pi5.sh
# En otra terminal/ventana
./scripts/start-helen-frontend-pi5.sh
```

Notas:
- Si existe `.venv`, el script lo activa automáticamente.
- El backend exporta `HELEN_MODEL_BACKEND=lstm` por defecto; si necesitas forzar XGBoost, ejecuta `export HELEN_MODEL_BACKEND=xgboost` antes de lanzar el servidor.

## 3. Uso en modo servicio (systemd)

1. Copia el servicio de ejemplo y ajusta la ruta si tu clon está en otra carpeta:
   ```bash
   sudo cp system_scripts/helen-pi5.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```
2. (Opcional) Habilita el servicio para que inicie con el sistema:
   ```bash
   sudo systemctl enable helen-pi5.service
   sudo systemctl start helen-pi5.service
   ```
3. Revisa los logs en `reports/logs/pi/` o vía journalctl:
   ```bash
   journalctl -u helen-pi5.service -f
   ```

El servicio ejecuta el backend y, tras un breve retraso, abre Chromium en modo kiosco usando `scripts/start-helen-frontend-pi5.sh`.
