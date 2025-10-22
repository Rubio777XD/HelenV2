# Informe de sesión de gestos

## Configuración activa
- Ventana de consenso: 5 frames (requiere 3)
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
- Frames procesados tras filtros: 18
- Revisiones de calidad: 0
- Descartes por calidad: 0 (0.0%)

### Rendimiento por clase
| Clase | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| Clima | nan | nan | nan | 0 | 0 | 0 |
| Reloj | nan | nan | nan | 0 | 0 | 0 |
| Inicio | nan | nan | nan | 0 | 0 | 0 |
| Start | 1.00 | 0.11 | 0.20 | 2 | 0 | 16 |

### Matriz de confusión
| Actual | None | Start |
| --- | --- | --- |
| Start | 16 | 2 |
> Nota: La sesión se ejecutó con `process_every_n = 3` (ventana efectiva 5/3) y el dataset sintético `data1.pickle`, por lo que solo se registraron activaciones de inicio.
