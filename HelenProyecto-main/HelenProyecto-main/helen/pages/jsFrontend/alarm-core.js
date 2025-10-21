// alarm-core.js — motor global de alarmas (sin UI)
// Cárgalo en TODAS las páginas (menú, clima, reloj, alarmas, etc.)

(function () {
  if (window.__alarmCoreStarted) return; // evita inicializar dos veces
  window.__alarmCoreStarted = true;

  // ========= Helpers de storage (compat con tu formato) =========
  function getAlarms() {
    try { return JSON.parse(localStorage.getItem('alarms')) || []; }
    catch (e) { return []; }
  }
  function saveAlarms(a) { localStorage.setItem('alarms', JSON.stringify(a || [])); }

  // ========= Evitar disparos duplicados si hay varias páginas abiertas =========
  function makeRingLockKey(alarm, now) {
    // Bloqueo por minuto: id + YYYY-MM-DDTHH:MM
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, '0');
    const d = String(now.getDate()).padStart(2, '0');
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    return `helen:alarm:lock:${alarm.id}:${y}-${m}-${d}T${hh}:${mm}`;
  }

  // ========= Warm de audio (desbloquear autoplay tras primer interacción) =========
  let audioWarmed = false;
  let warmAudioEl = null;
  function warmAudio() {
    if (audioWarmed) return;
    try {
      warmAudioEl = new Audio('data:audio/mp3;base64,//uQZAAAAAAAAAAAAAAAAAAAA'); // silencio
      const p = warmAudioEl.play();
      if (p && p.then) {
        p.then(() => { warmAudioEl.pause(); audioWarmed = true; }).catch(() => { });
      } else { audioWarmed = true; }
    } catch (e) { }
  }
  ['click', 'touchstart', 'keydown'].forEach(evt => {
    window.addEventListener(evt, warmAudio, { once: true, capture: true });
  });


  // ========= Tocar alarma =========
  function ringAlarm(alarm) {
    let audio;
    try {
      audio = new Audio('https://cdn.freesound.org/previews/219/219244_4082826-lq.mp3');
      audio.loop = true;
      const p = audio.play();
      if (p && p.catch) p.catch(() => { }); // por si el navegador bloquea
    } catch (e) { }

    showRingUI(alarm, () => {
      try { audio && audio.pause(); } catch (e) { }
    });
  }
  // ========= UI para ring (SweetAlert con skin HELEN; fallback si no hay Swal) =========
  function showRingUI(alarm, stopFn) {
    const timeStr = `${alarm.hours}:${alarm.minutes} ${alarm.ampm}`;
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
            <div class="helen-alarm-title">${alarm.name || 'Alarma'}</div>
            <div class="helen-alarm-time">${timeStr}</div>
          </div>
          <div class="helen-alarm-actions">
            <button class="helen-stop-btn" id="helenStopSwalCore">Detener</button>
          </div>
        </div>
      `,
        allowOutsideClick: false,
        allowEscapeKey: false,
        background: 'transparent',
        backdrop: 'rgba(0,0,0,.45)'
      }).then(() => stopFn && stopFn());

      // enganchar botón
      setTimeout(() => {
        const btn = document.getElementById('helenStopSwalCore');
        if (btn) btn.addEventListener('click', () => { Swal.close(); stopFn && stopFn(); }, { once: true });
      }, 0);
      return;
    }

    // Fallback sin SweetAlert (usa la misma tarjeta)
    const overlay = document.createElement('div');
    overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.45);
    display:flex;align-items:center;justify-content:center;z-index:99999;backdrop-filter:blur(3px);`;
    const wrap = document.createElement('div');
    wrap.className = 'helen-alarm-card';
    wrap.innerHTML = `
    <div class="helen-alarm-icon">${iconSVG}</div>
    <div>
      <div class="helen-alarm-title">${alarm.name || 'Alarma'}</div>
      <div class="helen-alarm-time">${timeStr}</div>
    </div>
    <div class="helen-alarm-actions">
      <button class="helen-stop-btn" id="helenStopFallback">Detener</button>
    </div>
  `;
    overlay.appendChild(wrap);
    document.body.appendChild(overlay);
    overlay.querySelector('#helenStopFallback').addEventListener('click', () => {
      overlay.remove();
      stopFn && stopFn();
    });
  }


  // ========= Chequeo cada segundo =========
  function checkAlarms(now) {
    const alarms = getAlarms();
    for (const alarm of alarms) {
      if (!alarm.active) continue;

      let hours = parseInt(alarm.hours, 10) || 0;
      const minutes = parseInt(alarm.minutes, 10) || 0;

      // 12h -> 24h
      if (alarm.ampm === 'PM' && hours < 12) hours += 12;
      else if (alarm.ampm === 'AM' && hours === 12) hours = 0;

      const alarmDay = now.getDay(); // 0..6
      const dayOK = (Array.isArray(alarm.days) && alarm.days.length > 0)
        ? alarm.days.includes(alarmDay)
        : true; // si no hay días => una vez/cualquiera

      if (dayOK &&
        now.getHours() === hours &&
        now.getMinutes() === minutes &&
        now.getSeconds() === 0) {

        // Evitar duplicados (si hay varias páginas con el core)
        const lockKey = makeRingLockKey(alarm, now);
        if (localStorage.getItem(lockKey)) continue;
        try { localStorage.setItem(lockKey, '1'); } catch (e) { }
        ringAlarm(alarm);
      }
    }
  }

  // ========= Bucle =========
  // Primer tick alineado al siguiente segundo
  function startLoop() {
    const ms = 1000 - (Date.now() % 1000);
    setTimeout(() => {
      checkAlarms(new Date());
      setInterval(() => checkAlarms(new Date()), 1000);
    }, ms);
  }

  startLoop();
})();
