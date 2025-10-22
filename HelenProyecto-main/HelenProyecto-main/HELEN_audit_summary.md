# Auditoría HELEN – Tutorial + Popup global

## Cambios principales
- **Tutorial (menú):** se remaquetó el contenedor con la misma retícula que Settings (`max-width`, `gap`, `padding-top`) y se dejaron únicamente el título “Tutorial interactivo” y las tres tarjetas de módulos. El contenedor queda centrado y sin scroll en 1366×768, 1440×900 y 1920×1080.
  ![Tutorial centrado](browser:/invocations/esoikblo/artifacts/artifacts/tutorial-menu.png)
- **Tutorial (flujo de pasos):** se ocultó la franja de íconos decorativos, se compactaron márgenes y se reutilizaron los tokens tipográficos de Settings para alinear encabezado, estado “PASO X DE N” y botón en un bloque centrado.
  ![Flujo compacto](browser:/invocations/jchrsmid/artifacts/artifacts/tutorial-flow.png)
- **Popup global temporizador/alarma:** `alarm-core.js` delega en el nuevo módulo `ui/notifications/notifications.js` para mostrar un único botón **Detener**, gestionar la cola `+N`, reproducir sonido y emitir eventos `timer:finished` / `alarm:triggered` sin listeners duplicados.

## Archivos modificados / añadidos
- `helen/pages/css/tutorial.css`
- `helen/pages/tutorial/tutorial.html`
- `helen/pages/jsFrontend/alarm-core.js`
- `helen/pages/weather/weather.html`
- `helen/pages/clock/{alarm,timer}.html`
- `helen/pages/settings/{settings,help,info,wifi}.html`
- `helen/pages/devices/devices.html`
- `helen/index.html`
- `helen/css/globals.css`
- `helen/ui/notifications/{notifications.js,notifications.css,README.md}`
- `tests/frontend/notifications-modal.test.js`

## Pruebas de aceptación
- Tutorial menú y flujo verificados manualmente a 1366×768 mediante Playwright (capturas adjuntas) y contraste visual con Settings.
- `npm test` (suite Node) → verifica popup, cola de notificaciones, emisión de eventos y flujos del tutorial.

## Estado y estabilidad
- Eventos SocketIO existentes (`timer:finished`, `alarm:triggered`, `helen:timekeeper:fired`) mantienen nombres y payload.
- El popup libera el foco con `Escape`, recupera el elemento activo anterior y sólo registra listeners una vez por carga.
- No se detectaron errores en consola durante la navegación entre Clima, Reloj, Dispositivos y Tutorial con el popup activo.
