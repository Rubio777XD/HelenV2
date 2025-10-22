# Informe de sesión de gestos

## Configuración activa
- Ventana de consenso: 4 frames (requiere 2)
- Cooldown tras 'Start': 700 ms
- Ventana de escucha C/R/I: 5.5 s
- Debounce de comandos: 750 ms
- Umbral global mínimo: 0.32
- Latencia consenso p50/p95: 363.2 / 371.6 ms
- Cadencia de inferencia: 1 de cada 3 frames (~0.12 s)

### Umbrales por clase
| Clase | Entrada | Liberación |
|-------|---------|------------|
| Start | 0.37 | 0.26 |
| Clima | 0.34 | 0.24 |
| Reloj | 0.35 | 0.25 |
| Inicio | 0.35 | 0.25 |

## Métricas de sesión
- Frames procesados tras filtros: 186
- Revisiones de calidad: 0
- Descartes por calidad: 0 (0.0%)

### Rendimiento por clase
| Clase | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| Inicio | nan | nan | nan | 0 | 0 | 0 |
| Reloj | nan | nan | nan | 0 | 0 | 0 |
| Clima | 1.00 | 0.32 | 0.49 | 30 | 0 | 63 |
| Start | 1.00 | 0.34 | 0.51 | 32 | 0 | 61 |

### Matriz de confusión
| Actual | Clima | None | Start |
| --- | --- | --- | --- |
| Clima | 30 | 63 | 0 |
| Start | 0 | 61 | 32 |

