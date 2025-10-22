# Informe de sesión de gestos

## Configuración activa
- Ventana de consenso: 12 frames (requiere 8)
- Cooldown tras 'Start': 800 ms
- Ventana de escucha C/R/I: 4.0 s
- Debounce de comandos: 750 ms
- Umbral global mínimo: 0.60

### Umbrales por clase
| Clase | Entrada | Liberación |
|-------|---------|------------|
| Start | 0.75 | 0.65 |
| Clima | 0.80 | 0.70 |
| Reloj | 0.78 | 0.68 |
| Inicio | 0.76 | 0.66 |

## Métricas de sesión
- Frames procesados tras filtros: 51
- Revisiones de calidad: 0
- Descartes por calidad: 0 (0.0%)

### Rendimiento por clase
| Clase | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| Start | 1.00 | 0.04 | 0.08 | 2 | 0 | 49 |
| Inicio | nan | nan | nan | 0 | 0 | 0 |
| Clima | nan | nan | nan | 0 | 0 | 0 |
| Reloj | nan | nan | nan | 0 | 0 | 0 |

### Matriz de confusión
| Actual | None | Start |
| --- | --- | --- |
| Start | 49 | 2 |

