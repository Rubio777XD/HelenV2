# Reporte UI / Net / Timers

## Resumen de correcciones
- Ajuste integral de alertas SweetAlert2: tarjetas Poppins sin blur global, animaciones fade-in/out estables y sin interferir con otras vistas.
- Marco de activación renovado: halo full-screen con breathing 3.6 s, auto apagado a los 5 s y micro bump inferior sólo en detección confirmada.
- Tutorial depurado: CSS libre de overlays invasivos, z-index seguro y gestos marcados como completados sin navegación real cuando el modo tutorial está activo.
- Componente Wi-Fi existente reactivado con endpoints Flask (`/net/online`, `/net/scan`, `/net/connect`, `/net/status`) y flujo UI con polling, feedback inline y manejo de errores.
- Sistema de alarmas/temporizadores persistentes finalizado: Shared/Dedicated Worker, almacenamiento local, AudioContext desbloqueado y nueva UI de temporizadores multi instancia con lista interactiva.
- Limpieza adicional de CSS global: sin filtros/backdrop globales, toasts accesibles y estilos consistentes con los tokens de `globals.css`.

## Evidencias
### Alertas sin blur
![Alerta sin blur global](reports/evidence/alerta-sin-blur.svg)

### Ring de activación + bump inferior
![Ring activo/inactivo y bump inferior](reports/evidence/ring-activacion.svg)

### Tutorial: gesto completado sin navegar
![Tutorial gestos completados](reports/evidence/tutorial-gesto.svg)

### Internet / Wi-Fi
![Flujo Wi-Fi completo](reports/evidence/wifi-estado.svg)

### Temporizadores persistentes
![Timers persistentes y audio](reports/evidence/timers-persistencia.svg)

## Checklist de validación
- ✅ Alertas sin blur/overlays globales.
- ✅ Ring solo al activar; contorno pantalla completa; breathing; off a los 5 s.
- ✅ “Bump” solo en borde inferior central.
- ✅ Tutorial estable; gestos marcan completado y no navegan en modo tutorial.
- ✅ Componente existente de Internet funcionando (scan/connect/status/polling).
- ✅ Alarmas/Timers continúan y suenan cambiando de pantalla (Worker + AudioContext).
- ✅ Accesibilidad OK; sin fugas de CSS a otras vistas.
