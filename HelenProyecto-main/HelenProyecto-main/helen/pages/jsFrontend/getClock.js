const ZONAS_HORARIAS = {
  mexicali: "America/Tijuana",
  ensenada: "America/Tijuana",
  tecate: "America/Tijuana",
  rosarito: "America/Tijuana",
  tijuana: "America/Tijuana",
  chiapas: "America/Mexico_City",
};

const CONFIG = {
  STORAGE_KEY: "selectedCity",
  DEFAULT_CITY: "tijuana",
  TIMEZONEDB_API_KEY: "SJABR4Q4XL7D",
};

(function () {
  'use strict';

  const MONTHS = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
  ];
  const ONE_HOUR_MS = 60 * 60 * 1000;
  const RETRY_DELAY_MS = 60 * 1000;

  const state = {
    timezoneKey: CONFIG.DEFAULT_CITY,
    is24h: true,
    localBaseMs: 0,
    syncClientMs: 0,
    tickTimer: null,
    refreshTimer: null,
    lastRenderedMs: 0,
  };

  const elements = {
    clock: null,
    date: null,
    toggleButton: null,
    selector: null,
  };

  const pad2 = (value) => String(value).padStart(2, '0');

  const setElementText = (element, value) => {
    if (!element) {
      return;
    }
    if (element.textContent !== value) {
      element.textContent = value;
    }
  };

  const computeParts = (ms) => {
    const date = new Date(ms);
    return {
      hours: date.getUTCHours(),
      minutes: date.getUTCMinutes(),
      seconds: date.getUTCSeconds(),
      day: date.getUTCDate(),
      monthIndex: date.getUTCMonth(),
      year: date.getUTCFullYear(),
    };
  };

  const formatTime = (parts) => {
    if (state.is24h) {
      return `${pad2(parts.hours)}:${pad2(parts.minutes)}:${pad2(parts.seconds)}`;
    }
    const hour12 = parts.hours % 12 || 12;
    const suffix = parts.hours >= 12 ? 'PM' : 'AM';
    return `${pad2(hour12)}:${pad2(parts.minutes)}:${pad2(parts.seconds)} ${suffix}`;
  };

  const formatDate = (parts) => `${parts.day} de ${MONTHS[parts.monthIndex]}, ${parts.year}`;

  const renderFromMs = (ms) => {
    if (!ms) {
      return;
    }
    const parts = computeParts(ms);
    setElementText(elements.clock, formatTime(parts));
    setElementText(elements.date, formatDate(parts));
    state.lastRenderedMs = ms;
  };

  const stopTick = () => {
    if (state.tickTimer) {
      window.clearTimeout(state.tickTimer);
      state.tickTimer = null;
    }
  };

  const runTick = () => {
    if (!state.localBaseMs) {
      return;
    }
    const now = Date.now();
    const elapsed = now - state.syncClientMs;
    const currentMs = state.localBaseMs + elapsed;
    renderFromMs(currentMs);
    const remainder = currentMs % 1000;
    const delay = remainder ? 1000 - remainder : 1000;
    state.tickTimer = window.setTimeout(runTick, Math.max(200, delay));
  };

  const startTick = () => {
    stopTick();
    runTick();
  };

  const scheduleRefresh = (delayMs) => {
    if (state.refreshTimer) {
      window.clearTimeout(state.refreshTimer);
      state.refreshTimer = null;
    }
    const delay = Math.max(RETRY_DELAY_MS, typeof delayMs === 'number' ? delayMs : ONE_HOUR_MS);
    state.refreshTimer = window.setTimeout(() => {
      state.refreshTimer = null;
      actualizarHoraLocal();
    }, delay);
  };

  const fetchWithRetry = async (timezone) => {
    const url = `https://api.timezonedb.com/v2.1/get-time-zone?key=${CONFIG.TIMEZONEDB_API_KEY}&format=json&by=zone&zone=${encodeURIComponent(timezone)}`;
    let attempt = 0;
    let backoff = 2000;

    while (attempt <= 5) {
      try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Error en la solicitud: ${response.status}`);
        }
        const data = await response.json();
        return {
          formatted: data.formatted,
          timestamp: Number(data.timestamp),
          gmtOffset: Number(data.gmtOffset),
          zoneName: data.zoneName || timezone,
        };
      } catch (error) {
        attempt += 1;
        if (attempt > 5) {
          throw error;
        }
        await new Promise((resolve) => window.setTimeout(resolve, backoff));
        backoff = Math.min(backoff * 1.5, 8000);
      }
    }
    throw new Error('No se pudo obtener la hora.');
  };

  async function actualizarHoraLocal() {
    const timezoneKey = state.timezoneKey || CONFIG.DEFAULT_CITY;
    const timezone = ZONAS_HORARIAS[timezoneKey];
    if (!timezone) {
      console.error('[Helen] Zona horaria no válida:', timezoneKey);
      return;
    }

    let success = false;

    try {
      const result = await fetchWithRetry(timezone);
      const timestamp = Number(result.timestamp);
      const offset = Number(result.gmtOffset);

      if (!Number.isFinite(timestamp) || !Number.isFinite(offset)) {
        throw new Error('Datos de hora incompletos.');
      }

      const localBaseMs = (timestamp + offset) * 1000;
      state.localBaseMs = localBaseMs;
      state.syncClientMs = Date.now();
      renderFromMs(localBaseMs);
      startTick();
      success = true;
    } catch (error) {
      console.error('[Helen] No se pudo sincronizar la hora:', error);
      stopTick();
      setElementText(elements.clock, '--:--:--');
      setElementText(elements.date, 'Sin conexión');
    } finally {
      scheduleRefresh(success ? ONE_HOUR_MS : RETRY_DELAY_MS);
    }
  }

  const loadSavedTimezone = () => {
    try {
      const saved = window.localStorage.getItem(CONFIG.STORAGE_KEY);
      if (saved && ZONAS_HORARIAS[saved]) {
        state.timezoneKey = saved;
      } else {
        state.timezoneKey = CONFIG.DEFAULT_CITY;
      }
    } catch (error) {
      console.warn('[Helen] No se pudo leer localStorage:', error);
      state.timezoneKey = CONFIG.DEFAULT_CITY;
    }
  };

  const handleSelectorChange = (event) => {
    const selected = event.target.value;
    if (!selected || !ZONAS_HORARIAS[selected]) {
      return;
    }
    if (state.timezoneKey === selected) {
      actualizarHoraLocal();
      return;
    }
    state.timezoneKey = selected;
    try {
      window.localStorage.setItem(CONFIG.STORAGE_KEY, selected);
    } catch (error) {
      console.warn('[Helen] No se pudo guardar la zona horaria seleccionada:', error);
    }
    actualizarHoraLocal();
  };

  const init = () => {
    elements.clock = document.querySelector('.clock-item');
    elements.date = document.querySelector('.date-item');
    elements.toggleButton = document.querySelector('.toggle-format-btn');
    elements.selector = document.getElementById('citySelector');

    loadSavedTimezone();

    if (elements.selector) {
      elements.selector.value = state.timezoneKey;
      elements.selector.addEventListener('change', handleSelectorChange);
    }

    if (elements.toggleButton) {
      elements.toggleButton.textContent = state.is24h ? 'Cambiar a 12 horas' : 'Cambiar a 24 horas';
    }

    actualizarHoraLocal();
  };

  const cleanup = () => {
    stopTick();
    if (state.refreshTimer) {
      window.clearTimeout(state.refreshTimer);
      state.refreshTimer = null;
    }
  };

  document.addEventListener('DOMContentLoaded', init);
  window.addEventListener('beforeunload', cleanup);
  window.addEventListener('pagehide', cleanup);

  window.alternarFormatoHora = function alternarFormatoHora() {
    state.is24h = !state.is24h;
    if (elements.toggleButton) {
      elements.toggleButton.textContent = state.is24h ? 'Cambiar a 12 horas' : 'Cambiar a 24 horas';
    }
    if (state.lastRenderedMs) {
      renderFromMs(state.lastRenderedMs);
    }
  };
})();
