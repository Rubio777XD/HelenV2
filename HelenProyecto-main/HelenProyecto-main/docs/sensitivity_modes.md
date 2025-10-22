# Modos de sensibilidad de HELEN

Los modos de sensibilidad permiten ajustar de forma declarativa cuán estrictos son los filtros de calidad, el consenso temporal y la post validación geométrica. El objetivo es poder pasar rápidamente de un perfil **STRICT** (ultra conservador) a perfiles **BALANCED** o **RELAXED** que aumentan el recall –especialmente de la seña “C”– sin disparar falsos positivos ni comprometer la latencia.

## Activación rápida

| Vía | Instrucción |
|-----|-------------|
| Variable de entorno | `export HELEN_SENS_MODE=BALANCED` (valores válidos: `STRICT`, `BALANCED`, `RELAXED`). |
| CLI del backend | `python -m backendHelen.server --sensitivity-mode RELAXED` (tiene prioridad sobre la variable). |

> **Predeterminado**: si no se especifica nada el backend utiliza el perfil **BALANCED** cargado desde `config/thresholds.json`.

El modo activo y la versión del perfil se informan al iniciar el runtime:
```
mode=BALANCED profile=version:1
```

## Parámetros por modo (versión 1)

| Modo | blur_laplacian_min | ROI mínima | Rango mano (px) | Consenso (N/M) | Cooldown (s) | Ventana escucha (s) | Var. pos máx. |
|------|--------------------|------------|-----------------|----------------|---------------|----------------------|----------------|
| STRICT   | 80  | 0.80 | 100–440 | 3/5 | 0.80 | 3.0 | ≤10.0 |
| BALANCED | 60  | 0.72 | 90–480  | 2/4 | 0.65 | 4.0 | ≤14.0 |
| RELAXED  | 52  | 0.68 | 80–520  | 2/3 | 0.55 | 4.5 | ≤16.0 |

Umbrales de `score_min` por clase (enter/off se calcula con histéresis `off_delta=0.08`):

| Clase | STRICT | BALANCED | RELAXED |
|-------|--------|----------|---------|
| H/Start | 0.62 | 0.55 | 0.50 |
| C/Clima | 0.60 | 0.48 | 0.44 |
| R/Reloj | 0.60 | 0.52 | 0.48 |
| I/Inicio| 0.60 | 0.52 | 0.48 |

La geometría de “C” incorpora curvatura, ratio de gap palma/arco y tolerancias angulares dependientes del modo. Se permiten hasta dos landmarks distales faltantes siempre que la curvatura promedio supere el mínimo del perfil.

## Telemetría y validación

* Cada sesión escribe `reports/gesture_session_report.(md|json)` con el modo activo, métricas de precisión/recall por clase, varianza posicional y latencias de consenso.
* Se generan snapshots automáticos en `logs/metrics_<MODO>.json`. Úsalos para comparar sesiones STRICT vs BALANCED conforme a los criterios de aceptación.
* Los eventos SSE incluyen `mode`, `profile_version`, `frameskip_used` y `position_variance` para facilitar depuración en vivo.

## Rate limiting adaptativo

* **BALANCED** procesa todos los frames mientras el FPS medio ≤ `rate_limit.fps_threshold` (25). Si la cámara entrega más FPS, se aplica un frameskip 1:3 configurable.
* **STRICT/RELAXED** respetan el frameskip del perfil (`frameskip_strict`/`frameskip_relaxed`). Todos los cambios quedan expuestos en los eventos y métricas.

## Fallback suave

Cuando la predicción principal queda a menos de 0.04 del umbral de entrada se consulta el clasificador centroidal. Si confirma con un score ≥ umbral se marca el evento con `fallback_confirmed`. El log `fallback_confirmed <label>` sirve para auditar cuántas confirmaciones ocurrieron.

## Riesgos y rollback rápido

* **BALANCED** incrementa el recall de “C”, pero ante iluminación muy cambiante puede subir el ruido. Vigila `fp_rate_none` en los JSON de métricas (objetivo ≤ 0.7%).
* **RELAXED** es adecuado solo en sesiones guiadas; aumenta tolerancias geométricas y puede aceptar manos parcialmente fuera de ROI.
* **Volver a STRICT**: `export HELEN_SENS_MODE=STRICT` o usar el selector en la UI (si tu build lo expone) seguido de reinicio del backend.

## Personalización avanzada

1. Edita `config/thresholds.json` para ajustar blur, ROI o scores por clase.
2. Reinicia el backend para que recargue el perfil.
3. Guarda nuevas métricas en `logs/metrics_<MODO>.json` tras validar en campo.

Mantén los tres archivos `logs/metrics_STRICT.json`, `logs/metrics_BALANCED.json` y `logs/metrics_RELAXED.json` como histórico de tus experimentos A/B.
