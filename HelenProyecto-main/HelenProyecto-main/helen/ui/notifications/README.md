# Notificaciones globales de alarmas y temporizadores

El motor `alarm-core.js` expone un bus compartido y un evento DOM para reaccionar a las alarmas desde cualquier pantalla sin duplicar listeners.

## API disponible

- **Objeto global**: `window.HelenTimekeeperBus` (versión 1)
  - `on(eventName, handler)` → registra un listener y devuelve una función de limpieza.
  - `off(eventName, handler)` → elimina un listener.
  - `emit(eventName, detail)` → dispara el evento global y el `CustomEvent` correspondiente.
- **Evento DOM**: `helen:timekeeper:fired`
  - Se emite en `window` con `detail` en el siguiente formato:

    ```json
    {
      "id": "timer-123",
      "eventId": "timer-123:1717000000000",
      "type": "timer",
      "tone": "timer",
      "title": "Temporizador finalizado",
      "label": "Sesión de estudio",
      "detail": "Duración: 25:00",
      "meta": "Se activó a las 10:15",
      "secondaryLabel": "Descartar"
    }
    ```

## Integración sugerida

```js
const unsubscribe = window.HelenTimekeeperBus.on('helen:timekeeper:fired', (snapshot) => {
  console.log('Evento global recibido', snapshot);
});

window.addEventListener('helen:timekeeper:fired', (event) => {
  console.log('CustomEvent recibido', event.detail);
});

// Limpieza al desmontar
unsubscribe();
```

La UI modal se ancla automáticamente al `<body>` y define `aria-live="assertive"`, foco controlado y botones de acción (`Detener sonido`, `Descartar`/`Repetir`). No es necesario instanciar elementos adicionales: basta con cargar `helen/pages/jsFrontend/alarm-core.js` una vez por página.

## Accesibilidad

- El popup utiliza `role="alertdialog"` y `aria-live="assertive"`.
- El botón primario recibe el foco inicial y respeta navegación por teclado (ciclo de tabulación y `Escape`).
- El sonido sólo se reproduce tras una interacción previa gracias al desbloqueo de `AudioContext` en `unlockAudioOnGesture()`.

## Estilos

Los estilos se definen en `helen/css/globals.css` (`.helen-global-modal`, `.helen-toast-stack`) y heredan los tokens (`--blue-a`, `--shadow-tile`, etc.). No dupliques reglas: extiende las clases existentes o añade modificadores (`data-tone="alarm"`).
