# Informe de sesión de gestos

## Configuración activa
- Ventana de consenso: 4 frames (requiere 2)
- Cooldown tras 'Start': 720 ms
- Ventana de escucha C/R/I: 5.6 s
- Debounce de comandos: 750 ms
- Umbral global mínimo: 0.37
- Perfil: SUAVE (versión 3)
- Latencia consenso p50/p95: 366.6 / 368.2 ms

### Umbrales por clase
| Clase | Entrada | Liberación |
|-------|---------|------------|
| Start | 0.41 | 0.27 |
| Clima | 0.37 | 0.23 |
| Reloj | 0.39 | 0.25 |
| Inicio | 0.39 | 0.25 |

## Métricas de sesión
- Frames procesados tras filtros: 25
- Revisiones de calidad: 0
- Descartes por calidad: 0 (0.0%)

### Rendimiento por clase
| Clase | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| Inicio | nan | nan | nan | 0 | 0 | 0 |
| Clima | nan | nan | nan | 0 | 0 | 0 |
| Start | 1.00 | 0.08 | 0.15 | 2 | 0 | 23 |
| Reloj | nan | nan | nan | 0 | 0 | 0 |

### Matriz de confusión
| Actual | None | Start |
| --- | --- | --- |
| Start | 23 | 2 |

