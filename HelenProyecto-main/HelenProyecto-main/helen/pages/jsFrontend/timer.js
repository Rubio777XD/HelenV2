// ============================
// HELEN Timer – JS adaptado a nuevo HTML/CSS
// ============================

// Estado
let timer = null;
let startTime = 0;        // ms timestamp al iniciar/reanudar
let pausedMs = 0;         // ms ya transcurridos (para reanudar)
let totalMs = 0;          // ms a contar (objetivo)
let isRunning = false;

// DOM (nuevo markup)
const display = document.getElementById('timerDisplay');
const startPauseBtn = document.getElementById('startPauseBtn');
const resetBtn = document.getElementById('resetBtn');
const presetsGrid = document.getElementById('presetsGrid');
const openCustomBtn = document.getElementById('openCustomBtn');

// Modal personalizado
const customModal = document.getElementById('customTimeModal');
const modalCloseBtn = customModal ? customModal.querySelector('.close-modal') : null;
const customCancelBtn = document.getElementById('customCancelBtn');
const customSetBtn = document.getElementById('customSetBtn');
const customHours = document.getElementById('customHours');
const customMinutes = document.getElementById('customMinutes');
const customSeconds = document.getElementById('customSeconds');

// ===== Utils de formato =====
function pad2(n) { return String(n).padStart(2, '0'); }

function formatForDisplay(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0 ? `${pad2(h)}:${pad2(m)}:${pad2(s)}` : `${pad2(m)}:${pad2(s)}`;
}

function setDisplayZero() {
  display.textContent = '00:00';
}

// ===== Render =====
function renderRemaining(nowMs) {
  const elapsed = nowMs - startTime;          // ms transcurridos en esta “corrida”
  const acc = pausedMs + elapsed;             // ms acumulados
  const remaining = Math.max(0, totalMs - acc);

  display.textContent = formatForDisplay(remaining);

  // fin
  if (remaining <= 0) {
    stopInterval();
    handleComplete();
    return true;
  }
  return false;
}

// ===== Control del bucle =====
function startInterval() {
  stopInterval();
  timer = setInterval(() => {
    renderRemaining(Date.now());
  }, 100); // 10fps es suficiente y liviano
}

function stopInterval() {
  if (timer) clearInterval(timer);
  timer = null;
}

// ===== Acciones =====
function canStart() { return totalMs > 0; }

function startTimer() {
  if (!canStart()) {
    showToast('error', 'Elige un tiempo para iniciar');
    return;
  }
  if (isRunning) return;

  startTime = Date.now();
  isRunning = true;
  startInterval();
  // UI
  startPauseBtn.innerHTML = '<i class="bi bi-pause-fill"></i>';
  startPauseBtn.classList.add('primary-btn');
  resetBtn.disabled = false;
}

function pauseTimer() {
  if (!isRunning) return;
  // acumular lo transcurrido
  pausedMs += (Date.now() - startTime);
  isRunning = false;
  stopInterval();
  // UI
  startPauseBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
  startPauseBtn.classList.add('primary-btn');
}

function toggleStartPause() {
  if (!isRunning) startTimer(); else pauseTimer();
}

function resetTimer() {
  stopInterval();
  isRunning = false;
  startTime = 0;
  pausedMs = 0;
  totalMs = 0;
  setDisplayZero();
  // UI
  startPauseBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
  resetBtn.disabled = true;
}

// ===== Setear tiempo (desde presets o modal) =====
function setTimer(h, m, s, autoStart = true) {
  h = Number(h) || 0; m = Number(m) || 0; s = Number(s) || 0;
  const ms = (h * 3600 + m * 60 + s) * 1000;
  if (ms <= 0) {
    showToast('error', 'El tiempo debe ser mayor a 0');
    return;
  }
  // Si estaba corriendo, lo pausamos
  stopInterval();
  isRunning = false;
  pausedMs = 0;
  totalMs = ms;
  display.textContent = formatForDisplay(totalMs);
  resetBtn.disabled = false;
  startPauseBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
  if (autoStart) startTimer();
}

// ===== Modal personalizado =====
function openCustomModal() {
  if (!customModal) return;
  // poner valores actuales (si hay) o 0
  const cur = Math.max(0, totalMs);
  const h = Math.floor(cur / 1000 / 3600);
  const m = Math.floor((cur / 1000 % 3600) / 60);
  const s = Math.floor(cur / 1000 % 60);
  customHours.value = h || 0;
  customMinutes.value = m || 0;
  customSeconds.value = s || 0;

  customModal.classList.add('open');
}
function closeCustomModal() { customModal && customModal.classList.remove('open'); }

function clampInputs() {
  if (!customHours || !customMinutes || !customSeconds) return;
  if (customHours.value < 0) customHours.value = 0;
  if (customHours.value > 23) customHours.value = 23;
  if (customMinutes.value < 0) customMinutes.value = 0;
  if (customMinutes.value > 59) customMinutes.value = 59;
  if (customSeconds.value < 0) customSeconds.value = 0;
  if (customSeconds.value > 59) customSeconds.value = 59;
}

// Botones +/− del modal
function setupStepperButtons() {
  document.querySelectorAll('.ct-plus').forEach(b => {
    b.addEventListener('click', () => {
      const t = b.dataset.target;
      const el = t === 'hours' ? customHours : t === 'minutes' ? customMinutes : customSeconds;
      let max = t === 'hours' ? 23 : 59;
      el.value = Math.min(max, Number(el.value || 0) + 1);
    });
  });
  document.querySelectorAll('.ct-minus').forEach(b => {
    b.addEventListener('click', () => {
      const t = b.dataset.target;
      const el = t === 'hours' ? customHours : t === 'minutes' ? customMinutes : customSeconds;
      el.value = Math.max(0, Number(el.value || 0) - 1);
    });
  });
}

// ===== Fin del temporizador =====
function handleComplete() {
  isRunning = false;
  pausedMs = 0;

  // Tarjeta Helen (mismo estilo que alarmas)
  const timeStr = formatForDisplay(0);
  const iconSVG = `
    <svg viewBox="0 0 24 24"><path d="M12 22a2.5 2.5 0 0 0 2.45-2h-4.9A2.5 2.5 0 0 0 12 22Zm7-6V11a7 7 0 0 0-5-6.71V3a2 2 0 1 0-4 0v1.29A7 7 0 0 0 5 11v5l-2 2v1h18v-1l-2-2Z"/></svg>
  `;

  if (window.Swal) {
    Swal.fire({
      customClass: { popup: 'swal-helen' },
      showConfirmButton: false,
      html: `
        <div class="helen-alarm-card">
          <div class="helen-alarm-icon">${iconSVG}</div>
          <div>
            <div class="helen-alarm-title">¡Tiempo terminado!</div>
            <div class="helen-alarm-time">${timeStr}</div>
          </div>
          <div class="helen-alarm-actions">
            <button class="helen-stop-btn" id="helenTimerDone">Aceptar</button>
          </div>
        </div>
      `,
      allowOutsideClick: false,
      allowEscapeKey: false,
      background: 'transparent',
      backdrop: 'rgba(0,0,0,.45)'
    }).then(() => {
      // reset visual
      resetTimer();
    });

    setTimeout(() => {
      const btn = document.getElementById('helenTimerDone');
      if (btn) btn.addEventListener('click', () => Swal.close(), { once: true });
    }, 0);
  } else {
    // Fallback sin SweetAlert
    resetTimer();
  }
}

// ===== Toast (usa tu CSS global si existe, fallback con Swal) =====
function showToast(type, message) {
  const t = document.getElementById('helenToast');
  if (t) {
    t.className = 'helen-toast';
    t.textContent = message || '';
    t.classList.add(type === 'error' ? 'error' : 'success');
    t.style.display = 'block';
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => { t.style.display = 'none'; }, 3000);
  } else if (window.Swal) {
    Swal.fire({ toast:true, icon:type, title:message, timer:2000, showConfirmButton:false, position:'top-end' });
  } else {
    console.log(type?.toUpperCase(), message);
  }
}

// ===== Eventos =====
document.addEventListener('DOMContentLoaded', () => {
  setDisplayZero();

  // Start/Pause y Reset
  if (startPauseBtn) startPauseBtn.addEventListener('click', toggleStartPause);
  if (resetBtn) resetBtn.addEventListener('click', resetTimer);

  // Presets grid
  if (presetsGrid) {
    presetsGrid.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const m = Number(btn.dataset.min || 0);
        setTimer(0, m, 0, true); // set y auto-start
      });
    });
  }

  // Modal abrir/cerrar
  if (openCustomBtn) openCustomBtn.addEventListener('click', openCustomModal);
  if (modalCloseBtn)  modalCloseBtn.addEventListener('click', closeCustomModal);
  if (customCancelBtn) customCancelBtn.addEventListener('click', closeCustomModal);
  if (customSetBtn) {
    customSetBtn.addEventListener('click', () => {
      clampInputs();
      setTimer(customHours.value, customMinutes.value, customSeconds.value, true);
      closeCustomModal();
    });
  }

  // Stepper +/−
  setupStepperButtons();

  // Para accesos por backend/gestos (API simple)
  window.TimerAPI = {
    set: (h=0,m=0,s=0,autostart=true) => setTimer(h,m,s,autostart),
    start: () => startTimer(),
    pause: () => pauseTimer(),
    reset: () => resetTimer(),
    running: () => isRunning,
    remainingMs: () => {
      if (!totalMs) return 0;
      if (!isRunning) return Math.max(0, totalMs - pausedMs);
      return Math.max(0, totalMs - (pausedMs + (Date.now()-startTime)));
    }
  };
});
