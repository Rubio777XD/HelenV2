# Parámetros del motor de decisiones

El backend opera con un único conjunto de valores estáticos pensado para
recuperar la sensibilidad del pipeline original y mantener bajos los falsos
positivos. Los umbrales se definen en `backendHelen/decision_params.py` y se
cargan al arrancar el servidor.

## Puntos clave

- **Consenso temporal:** 2 de 4 frames consecutivos.
- **Cooldown tras "H":** 0,70 s.
- **Ventana de escucha para C/R/I:** 5,5 s.
- **Debounce de comandos:** 0,75 s.
- **Retardo de activación:** 0,45 s.
- **Ventana de movimiento máximo:** varianza ≤ 18.0.
- **Stride fijo:** se procesa 1 de cada 3 frames mientras la cámara se mantenga
  estable.

## Umbrales por gesto (enter/release)

| Gesto | Enter | Release |
|-------|-------|---------|
| H/Start | 0.369 | 0.259 |
| C/Clima | 0.342 | 0.242 |
| R/Reloj | 0.351 | 0.251 |
| I/Inicio| 0.351 | 0.251 |

Los filtros geométricos permiten hasta dos landmarks distales oclusos, amplían
las tolerancias angulares y de curvatura entre un 15 % y un 25 % y mantienen la
normalización de la mano (con chequeos adicionales para evitar divisiones por
cero).

## Calidad de imagen

- Laplaciano mínimo: 24.0.
- Cobertura mínima ROI: 0.52.
- Rango de tamaño de mano: 55–640 px.
- Margen extra al proyectar la ROI: 0.05.

Un snapshot de estos valores se escribe en `logs/params_effective.json` cada vez
que arranca el backend. El endpoint `/engine/status` expone los mismos valores
junto con información de latencia y FPS.
