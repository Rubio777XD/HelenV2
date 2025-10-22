# HELEN

HELEN es una experiencia de asistente doméstico controlada por gestos. El backend escrito en Flask/Socket.IO expone los eventos de cámara y la lógica de reconocimiento de gestos (H, C, R, I) mientras que el frontend web muestra el tutorial, alarmas, temporizador y dispositivos conectados. Este repositorio contiene tanto el backend, como el modelo de detección de gestos y los recursos de la interfaz.

## Instalación

Selecciona la guía correspondiente a tu plataforma:

- [Windows 10/11 (ejecutable y instalador Inno Setup)](packaging/README-build-win.md)
- [Raspberry Pi 4/5 con Raspberry Pi OS 64-bit](packaging-pi/README-raspi.md)

Cada guía incluye los requisitos, comandos y validaciones específicas para garantizar que HELEN se ejecute con cámara real y la interfaz web en modo kiosko.
