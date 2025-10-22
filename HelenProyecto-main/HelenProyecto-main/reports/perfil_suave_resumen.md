# Perfil "suave" – resumen de calibración

## Antes ➜ Después

| Componente | Perfil legacy (BALANCED) | Perfil suave |
|------------|--------------------------|--------------|
| Consenso temporal | 2 / 4 | 2 / 4 (sin cambios) |
| Cooldown tras "H" | 0.70 s | 0.72 s |
| Ventana C/R/I | 5.8 s | 5.6 s |
| Varianza posicional máx. | 12.0 | 10.0 |
| Umbral "H" (enter/off) | 0.45 / 0.33 | 0.41 / 0.27 |
| Umbral "C" (enter/off) | 0.41 / 0.29 | 0.37 / 0.23 |
| Umbral "R" (enter/off) | 0.43 / 0.31 | 0.39 / 0.25 |
| Umbral "I" (enter/off) | 0.43 / 0.31 | 0.39 / 0.25 |
| Ángulo tol. "C" | 34° | 40° |
| Desv. norma "C" | 0.34 | 0.39 |
| Rate limiting | 1 de cada 3 cuando FPS > 24 | 1 de cada 3 con fallback a 1 |
| ROI | Bounding box sin margen extra | +3.5% de margen por lado |

## Ejecución de prueba (extracto de log)

```
[2024-07-18 10:14:03] INFO profile=SUAVE version=3
[2024-07-18 10:14:04] INFO Umbrales aplicados perfil=SUAVE v3 => Clima=0.37/0.23, Inicio=0.39/0.25, Reloj=0.39/0.25, Start=0.41/0.27
[2024-07-18 10:14:36] INFO diagnostic session=5c88 profile=SUAVE model=production (model.p) frames_total=864 frames_valid=752 fps=26.4 latency_avg=142.6ms latency_p95=198.4ms thresholds={Clima=0.37/0.23, Inicio=0.39/0.25, Reloj=0.39/0.25, Start=0.41/0.27} reasons={'score_below_threshold': 12, 'geometry_clima_gap': 3}
```

El reporte `reports/gesture_session_report.md` y el snapshot `logs/profile_snapshot.json`
recogen los mismos parámetros para auditoría.
