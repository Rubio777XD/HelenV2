# Mapa de eventos (SSE / Bus global)

| Evento / Canal                       | Emisor                                      | Escuchas principales                                                                                                 | Payload (claves)                                                                                     | Pantallas / Módulos |
|-------------------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------|
| `connect` (EventSource `open`)      | Backend `EventStream` al abrir `/events`    | `helen/jsSignHandler/SocketIO.js` → registra log de conexión                                                         | `{ message: 'connected', session_id, sequence: -1, timestamp }`                                         | Todas               |
| `message` (SSE)                     | `HelenRuntime.push_prediction()`             | `eventConnector.js` → reemite en `socket.on('message', …)`<br>`actions.js` y `tutorial-interactive.js` consumen gesto | `{ gesture, character, score, latency_ms, active?, state?, numeric, raw?, source }`                    | Inicio, Clima, Reloj, Dispositivos, Tutorial |
| `helen:timekeeper:fired` (DOM)      | `alarm-core.js` (`queueNotificationSnapshot`) | `window` (CustomEvent) + `window.HelenTimekeeperBus`<br>Modal global de alarmas, pruebas automáticas, listeners UI    | `{ id, eventId, type ('timer'|'alarm'), tone, title, label, detail, meta, secondaryLabel, firedAt }`   | Todas (alarmas/temporizadores persistentes) |
| Bus `update` (`HelenScheduler.on`)  | `alarm-core.js` (cambios en timers)          | Paneles que muestran lista de alarmas (`alarm.js`, `timer.js`), pruebas                                              | `[ { id, type, state, remainingMs, targetEpochMs, metadata… } ]`                                       | Reloj, Ajustes      |
| Bus `fired` (`HelenScheduler.on`)   | Worker de tiempo (`time-worker.js`)          | UI específicas que requieren reaccionar manualmente (opcional)                                                       | `{ id, type, label, state, metadata… }`                                                                | Reloj, Tutorial     |

> Nota: aunque el código conserva adaptadores compatibles con Socket.IO, la distribución actual utiliza Server-Sent Events (`EventSource`). El adaptador (`SocketIO.js`) expone una API con `socket.on/emit` para no romper contratos existentes.

## Ejemplos de uso

```js
// Escuchar gestos en cualquier vista
window.socket.on('message', (payload) => {
  console.log('Gesto recibido', payload.gesture, payload.score);
});

// Reaccionar al final de una alarma globalmente
const stopListening = window.HelenTimekeeperBus.on('helen:timekeeper:fired', (snapshot) => {
  console.log('Tiempo completado', snapshot.label, snapshot.tone);
});
```

## Resiliencia y reconexión
- `SocketIO.js` reintenta la conexión SSE con backoff fijo (1.5 s) tras un `error`.
- `eventConnector.js` reaprovecha `window.socket` para evitar listeners duplicados al cambiar de pestaña.
- `alarm-core.js` persiste notificaciones pendientes en `localStorage` (`helen:timekeeper:pendingQueue:v1`) y las rehidrata en recargas o reconexiones.
