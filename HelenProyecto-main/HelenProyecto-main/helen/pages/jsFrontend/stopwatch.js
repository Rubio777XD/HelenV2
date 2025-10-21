// =========================
// Configuración global
// =========================
const CONFIG = {
  UPDATE_INTERVAL: 10, // 10 ms -> mostramos centésimas
  DIAS: ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'],
  MESES: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
};

// =========================
// Clase principal TimerApp
// =========================
class TimerApp {
  constructor() {
    // Elementos DOM
    this.elements = {
      dateSection: $('.date-section'),
      timerDisplay: $('.timer-display'),
      milliseconds: $('.milliseconds'),
      timerIcon: $('.timer-icon'),

      totalTime: $('.total-time'),
      lapCount: $('.lap-count'),
      avgLap: $('.avg-lap'),
      bestLap: $('.best-lap'),
      worstLap: $('.worstLap'), // opcional si no existe en HTML

      startBtn: $('.start-btn'),
      resetBtn: $('.reset-btn'),
      lapBtn: $('.lap-btn'),

      lapsList: $('.laps-list'),
      lapsCountBadge: $('.laps-count-badge'),

      statsCard: $('.stats-card'),
      loadingIndicator: $('.timer-loading'),
      backBtn: $('.back-btn')
    };

    // Estado
    this.startTime = 0;       // timestamp de inicio (corrige pausas)
    this.elapsedTime = 0;     // ms acumulados cuando se pausa
    this.timerInterval = null;
    this.running = false;
    this.laps = [];           // [{number, time (acumulado), lapTime (duración)}]

    // Init
    this.init();
  }

  // -------------------------
  // Ciclo de vida
  // -------------------------
  init() {
    this.setupEventListeners();
    this.startDateTimer();
    this.resetTimer();

    // Oculta loading si existiera
    setTimeout(() => {
      if (this.elements.loadingIndicator && this.elements.loadingIndicator.length) {
        this.elements.loadingIndicator.fadeOut();
      }
    }, 500);
  }

  setupEventListeners() {
    // Play / Pause
    this.elements.startBtn.on('click', () => {
      if (!this.running) this.startTimer();
      else this.stopTimer();
    });

    // Reset
    this.elements.resetBtn.on('click', () => this.resetTimer());

    // Lap
    this.elements.lapBtn.on('click', () => this.addLap());

    // Back (barra superior)
    if (this.elements.backBtn && this.elements.backBtn.length) {
      this.elements.backBtn.on('click', () => history.back());
    }

    // Teclado
    $(document).on('keydown', (e) => {
      // Espacio → start/stop
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        if (!this.running) this.startTimer();
        else this.stopTimer();
      }
      // L → vuelta (si está corriendo)
      if ((e.key === 'l' || e.key === 'L') && this.running) this.addLap();
      // R → reset (si está detenido)
      if ((e.key === 'r' || e.key === 'R') && !this.running) this.resetTimer();
    });
  }

  // -------------------------
  // Fecha (si decides mostrarla)
  // -------------------------
  startDateTimer() {
    this.updateDate();
    setInterval(() => this.updateDate(), 60000);
  }
  updateDate() {
    const d = new Date();
    const s = `${CONFIG.DIAS[d.getDay()]}, ${d.getDate()} de ${CONFIG.MESES[d.getMonth()]} de ${d.getFullYear()}`;
    this.elements.dateSection.html(s);
  }

  // -------------------------
  // Control del cronómetro
  // -------------------------
  startTimer() {
    if (this.running) return;
    this.running = true;

    const now = Date.now();
    this.startTime = now - this.elapsedTime; // respeta el tiempo acumulado

    this.timerInterval = setInterval(() => this.updateElapsedTime(), CONFIG.UPDATE_INTERVAL);

    // UI
    this.elements.startBtn.html('<i class="bi bi-pause-fill"></i><span>Detener</span>');
    this.elements.startBtn.addClass('stop-btn'); // CSS la vuelve roja
    this.elements.resetBtn.prop('disabled', true);  // oculta Reset (slot)
    this.elements.lapBtn.prop('disabled', false);   // muestra Vuelta (slot)
    this.elements.timerIcon.removeClass('bi-stopwatch').addClass('bi-stopwatch-fill');
  }

  stopTimer() {
    if (!this.running) return;
    this.running = false;

    clearInterval(this.timerInterval);
    this.timerInterval = null;

    // Acumular lo transcurrido hasta ahora
    this.elapsedTime = Date.now() - this.startTime;

    // UI
    this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Continuar</span>');
    this.elements.startBtn.removeClass('stop-btn'); // vuelve a verde
    this.elements.resetBtn.prop('disabled', false); // muestra Reset (slot)
    this.elements.lapBtn.prop('disabled', true);    // oculta Vuelta (slot)
    this.elements.timerIcon.removeClass('bi-stopwatch-fill').addClass('bi-stopwatch');
  }

  resetTimer() {
    // Detener si estaba corriendo
    if (this.running) this.stopTimer();

    this.elapsedTime = 0;
    this.startTime = 0;
    this.laps = [];

    // UI base
    this.updateTimerDisplay();
    this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Iniciar</span>');
    this.elements.resetBtn.prop('disabled', true);
    this.elements.lapBtn.prop('disabled', true);
    this.elements.lapsList.empty();

    // Stats y badge
    this.updateLapStats();
    this.updateLapsCountBadge();
    this.toggleStatsVisibility();
  }

  updateElapsedTime() {
    // tiempo transcurrido = ahora - startTime
    const now = Date.now();
    this.elapsedTime = now - this.startTime;
    this.updateTimerDisplay();
  }

  updateTimerDisplay() {
    const t = this.formatTime(this.elapsedTime);
    this.elements.timerDisplay.text(t.main);
    this.elements.milliseconds.text(t.ms);
    this.elements.totalTime.text(`Tiempo Total: ${t.main}`);
  }

  // -------------------------
  // Vueltas
  // -------------------------
  addLap() {
    if (!this.running) return;

    const lapTime = this.elapsedTime; // acumulado en ms
    const prevCumulative = this.laps.length > 0 ? this.laps[this.laps.length - 1].time : 0;
    const lapDuration = lapTime - prevCumulative;

    this.laps.push({
      number: this.laps.length + 1,
      time: lapTime,        // acumulado total hasta esta vuelta
      lapTime: lapDuration, // duración de la vuelta
    });

    this.updateLapsList();
    this.updateLapStats();
    this.updateLapsCountBadge();
    this.toggleStatsVisibility();
  }

  updateLapsList() {
    this.elements.lapsList.empty();

    // Mostramos la más reciente arriba
    [...this.laps].reverse().forEach(lap => {
      const lapTimeFormatted = this.formatTime(lap.lapTime);
      const totalFormatted = this.formatTime(lap.time);

      // Estructura: # | Vuelta | Total (usa estilos existentes .lap-item / .lap-number / .lap-delta)
      const item = `
        <div class="lap-item">
          <div class="lap-number">#${lap.number}</div>
          <div class="lap-time">${lapTimeFormatted.main}.${lapTimeFormatted.ms}</div>
          <div class="lap-delta">${totalFormatted.main}</div>
        </div>
      `;
      this.elements.lapsList.append(item);
    });
  }

  updateLapStats() {
    const n = this.laps.length;
    this.elements.lapCount.text(`Vueltas: ${n}`);

    if (n > 0) {
      // Promedio (de duraciones de vueltas)
      const avg = this.calculateAverageLapTime();
      const avgF = this.formatTime(avg);
      this.elements.avgLap.text(`Promedio: ${avgF.main}`);

      // Best / Worst (si existen en el HTML)
      const lapsOnly = this.laps.map(l => l.lapTime);
      const best = Math.min(...lapsOnly);
      const worst = Math.max(...lapsOnly);

      if (this.elements.bestLap && this.elements.bestLap.length) {
        this.elements.bestLap.text(`Más Rápida: ${this.formatTime(best).main}`);
      }
      if (this.elements.worstLap && this.elements.worstLap.length) {
        this.elements.worstLap.text(`Más Lenta: ${this.formatTime(worst).main}`);
      }
    } else {
      this.elements.avgLap.text('Promedio: 00:00:00');
      if (this.elements.bestLap && this.elements.bestLap.length) {
        this.elements.bestLap.text('Más Rápida: 00:00:00');
      }
      if (this.elements.worstLap && this.elements.worstLap.length) {
        this.elements.worstLap.text('Más Lenta: 00:00:00');
      }
    }
  }

  updateLapsCountBadge() {
    const n = this.laps.length;
    if (this.elements.lapsCountBadge && this.elements.lapsCountBadge.length) {
      this.elements.lapsCountBadge.text(`(${n})`);
    }
  }

  toggleStatsVisibility() {
    if (!this.elements.statsCard || !this.elements.statsCard.length) return;
    if (this.laps.length > 0) {
      this.elements.statsCard.removeClass('is-hidden');
    } else {
      this.elements.statsCard.addClass('is-hidden');
    }
  }

  // -------------------------
  // Utilidades de tiempo
  // -------------------------
  /**
   * Formatea ms -> { main: "HH:MM:SS", ms: "cc" } (centésimas)
   */
  formatTime(timeInMs) {
    const totalSeconds = Math.floor(timeInMs / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((timeInMs % 1000) / 10); // 0-99

    return {
      main: `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`,
      ms: centiseconds.toString().padStart(2, '0')
    };
  }

  /**
   * Promedio de duración de vueltas (no del acumulado)
   */
  calculateAverageLapTime() {
    if (this.laps.length === 0) return 0;
    const total = this.laps.reduce((sum, lap) => sum + lap.lapTime, 0);
    return total / this.laps.length;
  }
}

// -------------------------
// Bootstrap
// -------------------------
$(document).ready(() => {
  window.timerApp = new TimerApp();
});
