# Perfil de sensibilidad "suave"

Desde la versión 3 del backend todos los filtros de visión operan con un único
perfil global más permisivo. El objetivo es recuperar la sensibilidad del
pipeline original, aceptar variaciones de luz/ángulo/mano parcial y mantener los
falsos positivos bajos gracias a histéresis y consenso temporal.

## Parámetros activos (versión 3)

| Parámetro | Valor |
|-----------|-------|
| Umbral Laplaciano (blur) | 32.0 |
| Cobertura mínima ROI | 0.58 |
| Rango tamaño mano (px) | 60 – 600 |
| Consenso | 2 de 4 |
| Cooldown tras "H" | 0.72 s |
| Ventana escucha C/R/I | 5.6 s |
| Varianza posicional máx. | 10.0 |

### Umbrales por gesto (enter / release)

Los umbrales dinámicos aplican histéresis con `off_delta = 0.14`.

| Gesto | Score base | Ángulo tol. | Desv. norma | Curvatura min | Gap permitido |
|-------|------------|-------------|--------------|---------------|---------------|
| H/Start | 0.405 | 32° | 0.36 | — | — |
| C/Clima | 0.369 | 40° | 0.39 | 0.20 | 0.20 – 0.68 |
| R/Reloj | 0.387 | 33° | 0.35 | — | — |
| I/Inicio| 0.387 | 33° | 0.35 | — | — |

Se permiten hasta dos landmarks distales oclusos y las tolerancias geométricas
(ángulos, curvaturas, gap y desviaciones) se amplían entre un 15 % y un 25 % para
evitar descartes espurios.

## Rate limiting

El backend procesa un frame de cada tres mientras el FPS efectivo supere 24.0.
Cuando la cámara cae por debajo de ese umbral el stride baja automáticamente al
mínimo (1) para recuperar estabilidad.

## Telemetría y snapshots

* `logs/profile_snapshot.json` registra los parámetros aplicados al iniciar la
  sesión (umbral por clase, histéresis, rate limiting, consenso y rutas de
  modelo/dataset).
* `reports/gesture_session_report.(md|json)` consolida métricas de calidad,
  consensos y latencias.
* `logs/metrics_suave.json` guarda el último informe detallado para comparar
  sesiones.

## Compatibilidad

El selector de modos quedó obsoleto. El CLI y las variables de entorno ignoran
cualquier valor previo (`STRICT/BALANCED/RELAXED`). Los eventos SSE exponen el
campo `profile` (y un alias legado `mode`) con el valor fijo `SUAVE`.
