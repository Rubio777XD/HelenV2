# Auditoría HELEN – Tutorial interactivo y notificaciones globales

## Cambios principales
- Maquetado del tutorial rehídratado siguiendo la retícula de `settings.css`: header compacto, cards alineadas y contenedor sin desbordes verticales en 1366×768, 1440×900 y 1920×1080.  
  ![Tutorial sin scroll](browser:/invocations/ubjbzjrf/artifacts/artifacts/tutorial-menu.png)
- Normalización tipográfica y de espaciados (`clamp`, `gap`, `max-width`) para que las tres tarjetas queden visibles sin texto extra bajo el título y con íconos centrados.
- Refuerzo del motor de alarmas: popup accesible (`aria-live="assertive"`), textos "Temporizador finalizado" / "Alarma activada", y event-bus global `window.HelenTimekeeperBus` que despacha `helen:timekeeper:fired` con audio auto-gestionado.
- Pipeline backend ajustado a stride configurable (`process_every_n`) con consenso 3/5 frames y CLI `--frame-stride`, evitando ráfagas innecesarias en hardware embebido.
- Documentación y trazabilidad nuevas: `ui/notifications/README.md`, tabla de eventos (`reports/socketio-events.md`), métricas de modelo (`reports/model_metrics.json`).

## Archivos modificados / añadidos
- `helen/pages/tutorial/tutorial.html`
- `helen/pages/css/tutorial.css`
- `helen/pages/jsFrontend/alarm-core.js`
- `tests/frontend/notifications-modal.test.js`
- `backendHelen/server.py`
- `reports/socketio-events.md`
- `helen/ui/notifications/README.md`
- `reports/gesture_session_report.md` / `.json`
- `reports/model_metrics.json`

## Optimizaciones y resultados
- **Layout**: cards con gradiente y sombras reutilizan tokens (`--blue-a`, `--shadow-tile`) y emplean `grid` + `clamp()` para escalar sin overflow.
- **Notificaciones**: modal único en `<body>` evita listeners duplicados; el bus devuelve función de limpieza y emite `CustomEvent` + callbacks internos.
- **Pipeline**: `GesturePipeline` procesa 1 de cada 3 frames (configurable) y la ventana de consenso baja a 5/3, manteniendo histéresis y cooldown.
- **Sonido**: se reutiliza `digital_watch_alarm_long.ogg`; las notas del modal indican bloqueo de autoplay y parada automática.
- **Reportes**: `gesture_session_report` reexportado con consenso 5/3 y nota de dataset sintético para contextualizar las métricas.

## Métricas
| Concepto | Resultado |
| --- | --- |
| Dataset fallback (`data1.pickle`) | 34,302 muestras |
| Precisión simple classifier | 77.56 % (dataset reducido, sin `data.pickle`) |
| Latencia media de inferencia | 0.073 ms (p95 0.107 ms, σ 0.015 ms) |
| Latencia evento → popup | ≤ 50 ms (medido en test `global timekeeper bus emits fired events`) |
| Texto popup | "Temporizador finalizado" / "Alarma activada" |

> Nota: la precisión difiere del reporte (99.78 %) porque en el repositorio sólo está disponible el dataset reducido `data1.pickle`; el modelo de producción (`model.p`) permanece referenciado pero no incluido.

## Pruebas ejecutadas
- `pytest` (API backend, pipeline, modelo) ✔️
- `npm test` (suite Playwright/Node: modal accesible, cola, bus global, tutorial state machine) ✔️
- Verificación manual con Playwright (captura tutorial) ✔️

## Riesgos y próximos pasos
- Reentrenar o validar con `data.pickle` cuando esté disponible para confirmar la métrica de 99.78 %.
- En dispositivos sin audio desbloqueado, revisar mensajes del modal (`unlockAudioOnGesture`) para evitar falsos positivos de sonido bloqueado.
- Considerar una barra de estado visual para el bus global si se añaden más disparadores concurrentes.

## Checklist
- [x] Tutorial: título + 3 tarjetas, sin textos extra ni scroll.
- [x] Espaciado consistente con `settings.css`; íconos normalizados.
- [x] Popup global accesible con sonido y botón de cerrar.
- [x] Sin listeners duplicados / fugas (bus global, SharedWorker reusado).
- [x] Eventos SSE/bus documentados; contratos sin cambios en nombres.
- [x] Pipeline ajustado con rate limiting y consenso reducido.
- [x] Consola/tests limpios (`pytest`, `npm test`).
- [x] Documentación actualizada (`HELEN_audit_summary.md`, `reports/socketio-events.md`, `ui/notifications/README.md`).
