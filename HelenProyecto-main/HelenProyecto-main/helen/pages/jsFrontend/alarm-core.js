/* =========================================================
   HELEN – Motor persistente de alarmas y temporizadores
   Ejecuta el cálculo en un Web Worker / SharedWorker para
   evitar throttling y mantiene el estado en localStorage.
   ========================================================= */

(function () {
  'use strict';

  if (window.HelenScheduler && window.HelenScheduler.version === 2) {
    return;
  }

  const STORAGE_KEY = 'helen:timekeeper:v1';
  const VERSION = 2;
  const ONE_SECOND = 1000;
  const ONE_MINUTE = 60 * ONE_SECOND;
  const ONE_HOUR = 60 * ONE_MINUTE;
  const ONE_DAY = 24 * ONE_HOUR;
  const TIMER_TOAST_DURATION = 6500;
  const AUTO_REMOVE_DELAY = TIMER_TOAST_DURATION + 2200;

  const listeners = {
    update: new Set(),
    fired: new Set(),
  };

  const store = new Map();
  const pendingMessages = [];
  const readyResolvers = [];
  let workerPort = null;
  let workerReady = false;
  let audioContext = null;
  let audioUnlocked = false;
  let audioQueueTime = 0;
  let fallbackAudio = null;
  let toastContainer = null;
  let reconciled = false;

  const readyPromise = new Promise((resolve) => {
    readyResolvers.push(resolve);
  });

  const now = () => Date.now();

  const uuid = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    const random = Math.random().toString(16).slice(2, 10);
    const time = now().toString(16);
    return `helen-${time}-${random}`;
  };

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const toNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const clone = (value) => {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      return value;
    }
  };

  const emit = (event, payload) => {
    const callbacks = listeners[event];
    if (!callbacks) return;
    callbacks.forEach((handler) => {
      try {
        handler(payload);
      } catch (error) {
        console.error('[HelenScheduler] Error en listener', error);
      }
    });
  };

  const on = (event, handler) => {
    if (listeners[event]) {
      listeners[event].add(handler);
    }
  };

  const off = (event, handler) => {
    if (listeners[event]) {
      listeners[event].delete(handler);
    }
  };

  const safeLocalStorage = {
    get(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo leer localStorage:', error);
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo guardar localStorage:', error);
      }
    },
  };

  const loadFromStorage = () => {
    const raw = safeLocalStorage.get(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo parsear el estado guardado:', error);
      return [];
    }
  };

  const persistState = () => {
    const snapshot = Array.from(store.values()).map((item) => clone(item));
    safeLocalStorage.set(STORAGE_KEY, JSON.stringify(snapshot));
  };

  const sendToWorker = (message) => {
    if (!workerPort) return;
    if (!workerReady) {
      pendingMessages.push(message);
      return;
    }
    try {
      workerPort.postMessage(message);
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo enviar mensaje al worker:', error);
    }
  };

  const syncWithWorker = () => {
    const payload = Array.from(store.values()).map((item) => ({
      id: item.id,
      type: item.type,
      state: item.state,
      targetEpochMs: typeof item.targetEpochMs === 'number' ? item.targetEpochMs : null,
      remainingMs: typeof item.remainingMs === 'number' ? item.remainingMs : null,
    }));
    sendToWorker({ type: 'set-items', items: payload });
  };

  const notifyUpdate = () => {
    emit('update', list());
    persistState();
    syncWithWorker();
  };

  const ensureToastContainer = () => {
    if (toastContainer || !document || !document.body) return toastContainer;
    const container = document.createElement('div');
    container.className = 'helen-toast-stack';
    container.setAttribute('data-toast-stack', '');
    document.body.appendChild(container);
    toastContainer = container;
    return container;
  };

  const pushToast = ({ title, body, tone = 'info', duration = TIMER_TOAST_DURATION }) => {
    const container = ensureToastContainer();
    if (!container) return;
    const card = document.createElement('div');
    card.className = `helen-toast-card helen-toast-card--${tone}`;

    const icon = document.createElement('div');
    icon.className = 'helen-toast-card__icon';
    icon.innerHTML = tone === 'alarm'
      ? '<i class="bi bi-bell" aria-hidden="true"></i>'
      : '<i class="bi bi-stopwatch" aria-hidden="true"></i>';

    const content = document.createElement('div');
    content.className = 'helen-toast-card__body';
    const heading = document.createElement('div');
    heading.className = 'helen-toast-card__title';
    heading.textContent = title || '';
    const message = document.createElement('div');
    message.className = 'helen-toast-card__message';
    message.textContent = body || '';
    content.append(heading, message);

    card.append(icon, content);
    container.append(card);

    requestAnimationFrame(() => {
      card.classList.add('is-visible');
    });

    const remove = () => {
      card.classList.remove('is-visible');
      setTimeout(() => {
        card.remove();
        if (container.childElementCount === 0) {
          container.remove();
          toastContainer = null;
        }
      }, 320);
    };

    if (duration > 0) {
      setTimeout(remove, duration);
    }

    card.addEventListener('click', remove, { once: true });
  };

  const scheduleAutoRemoval = (item, delay = AUTO_REMOVE_DELAY) => {
    if (!item || !item.id || !item.metadata || !item.metadata.autoRemove) {
      return;
    }
    window.setTimeout(() => {
      const current = store.get(item.id);
      if (!current || current.type !== 'timer') {
        return;
      }
      if (current.state === 'completed' && current.metadata && current.metadata.autoRemove) {
        store.delete(current.id);
        notifyUpdate();
      }
    }, Math.max(1200, delay));
  };

  const ensureAudioContext = () => {
    if (audioUnlocked) {
      return audioContext;
    }
    audioUnlocked = true;
    try {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      if (Ctor) {
        audioContext = new Ctor();
      }
    } catch (error) {
      audioContext = null;
    }
    if (!fallbackAudio) {
      fallbackAudio = new Audio('data:audio/wav;base64,UklGRhQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=');
    }
    return audioContext;
  };

  const unlockAudioOnGesture = () => {
    const unlock = () => {
      ensureAudioContext();
      if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume().catch(() => {});
      }
      window.removeEventListener('pointerdown', unlock, true);
      window.removeEventListener('keydown', unlock, true);
      window.removeEventListener('touchstart', unlock, true);
      window.removeEventListener('click', unlock, true);
    };
    window.addEventListener('pointerdown', unlock, true);
    window.addEventListener('keydown', unlock, true);
    window.addEventListener('touchstart', unlock, true);
    window.addEventListener('click', unlock, true);
  };

  const playChime = () => {
    const context = ensureAudioContext();
    if (context) {
      if (context.state === 'suspended') {
        context.resume().catch(() => {});
      }
      const startAt = Math.max(context.currentTime, audioQueueTime);
      const gain = context.createGain();
      gain.gain.setValueAtTime(0, startAt);
      gain.connect(context.destination);

      const oscillator = context.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(880, startAt);
      gain.gain.linearRampToValueAtTime(0.36, startAt + 0.02);
      gain.gain.linearRampToValueAtTime(0, startAt + 1.1);
      oscillator.connect(gain);
      oscillator.start(startAt);
      oscillator.stop(startAt + 1.2);

      audioQueueTime = startAt + 0.4;
      return;
    }

    if (fallbackAudio) {
      try {
        const cloneAudio = fallbackAudio.cloneNode();
        cloneAudio.play().catch(() => {});
      } catch (error) {
        console.warn('[HelenScheduler] No se pudo reproducir sonido de respaldo:', error);
      }
    }
  };

  const computeAlarmTarget = (item, reference = now()) => {
    const meta = item.metadata || {};
    const hours24 = toNumber(meta.hours24, 0);
    const minutes = clamp(toNumber(meta.minutes, 0), 0, 59);
    if (!Number.isFinite(hours24)) {
      return null;
    }

    const base = new Date(reference);
    base.setMilliseconds(0);
    base.setSeconds(0);
    base.setMinutes(minutes);
    base.setHours(hours24);

    let candidate = base.getTime();
    const days = Array.isArray(meta.repeatDays) ? meta.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day)) : [];
    const currentDay = new Date(reference).getDay();

    if (days.length) {
      let best = null;
      days.sort((a, b) => a - b);
      for (const day of days) {
        let delta = day - currentDay;
        if (delta < 0) delta += 7;
        let scheduled = candidate + delta * ONE_DAY;
        if (delta === 0 && scheduled <= reference + ONE_SECOND) {
          scheduled += 7 * ONE_DAY;
        }
        if (best === null || scheduled < best) {
          best = scheduled;
        }
      }
      return best;
    }

    if (candidate <= reference + ONE_SECOND) {
      candidate += ONE_DAY;
    }
    return candidate;
  };

  const reconcilePastDueItems = () => {
    if (reconciled) return;
    reconciled = true;
    const notifications = [];
    const current = now();
    store.forEach((item) => {
      if (item.type === 'timer' && item.targetEpochMs && item.targetEpochMs <= current) {
        item.state = 'completed';
        item.targetEpochMs = null;
        item.remainingMs = 0;
        item.metadata = item.metadata || {};
        item.metadata.lastTriggeredAt = current;
        notifications.push({ item, tone: 'timer', silent: true });
        scheduleAutoRemoval(item, TIMER_TOAST_DURATION);
      }
      if (item.type === 'alarm' && item.metadata && item.metadata.active && item.targetEpochMs && item.targetEpochMs <= current) {
        notifications.push({ item, tone: 'alarm', silent: true });
        handleAlarmTriggered(item, current, { silent: true });
      }
    });
    if (notifications.length) {
      persistState();
      syncWithWorker();
    }
    notifications.forEach(({ item, tone }) => {
      pushToast({
        title: tone === 'alarm' ? 'Alarma finalizada' : 'Temporizador finalizado',
        body: tone === 'alarm'
          ? `${item.label || 'Alarma'} terminó mientras estabas fuera.`
          : `${item.label || 'Temporizador'} terminó mientras estabas fuera.`,
        tone,
      });
    });
  };

  const handleTimerTriggered = (item, firedAt, { silent } = {}) => {
    item.state = 'completed';
    item.targetEpochMs = null;
    item.remainingMs = 0;
    item.metadata = item.metadata || {};
    item.metadata.lastTriggeredAt = firedAt;
    if (!silent) {
      playChime();
      pushToast({
        title: item.label || 'Temporizador',
        body: 'Tiempo finalizado.',
        tone: 'timer',
      });
    }
    scheduleAutoRemoval(item);
  };

  const handleAlarmTriggered = (item, firedAt, { silent } = {}) => {
    item.metadata = item.metadata || {};
    item.metadata.lastTriggeredAt = firedAt;
    const nextTarget = computeAlarmTarget(item, firedAt + ONE_SECOND);
    if (nextTarget && item.metadata.active) {
      item.targetEpochMs = nextTarget;
      item.remainingMs = nextTarget - now();
      item.state = 'running';
    } else {
      item.metadata.active = false;
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'completed';
    }
    if (!silent) {
      playChime();
      pushToast({
        title: item.label || 'Alarma',
        body: '¡Despierta! La alarma se activó.',
        tone: 'alarm',
      });
    }
  };

  const handleWorkerFired = (payload) => {
    if (!Array.isArray(payload)) return;
    const firedAt = now();
    payload.forEach((entry) => {
      if (!entry || !entry.id) return;
      const item = store.get(entry.id);
      if (!item) return;
      if (item.type === 'timer') {
        handleTimerTriggered(item, firedAt);
      } else if (item.type === 'alarm') {
        handleAlarmTriggered(item, firedAt);
      }
      emit('fired', clone(item));
    });
    notifyUpdate();
  };

  const handleWorkerTick = (payload) => {
    if (!Array.isArray(payload)) return;
    let changed = false;
    payload.forEach((entry) => {
      if (!entry || !entry.id) return;
      const item = store.get(entry.id);
      if (!item) return;
      const remaining = toNumber(entry.remainingMs, 0);
      if (item.remainingMs !== remaining) {
        item.remainingMs = remaining;
        changed = true;
      }
    });
    if (changed) {
      emit('update', list());
    }
  };

  const handleWorkerMessage = (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') {
      return;
    }
    switch (data.type) {
      case 'ready': {
        workerReady = true;
        while (pendingMessages.length) {
          const message = pendingMessages.shift();
          sendToWorker(message);
        }
        syncWithWorker();
        readyResolvers.forEach((resolve) => resolve());
        readyResolvers.length = 0;
        break;
      }
      case 'tick':
        handleWorkerTick(data.payload);
        break;
      case 'fired':
        handleWorkerFired(data.payload);
        break;
      default:
        break;
    }
  };

  const resolveWorkerScriptUrl = () => {
    const script = document.currentScript;
    if (script && script.src) {
      return new URL('./time-worker.js', script.src).toString();
    }
    return new URL('./time-worker.js', window.location.href).toString();
  };

  const setupWorker = () => {
    const url = resolveWorkerScriptUrl();
    try {
      if (typeof SharedWorker !== 'undefined') {
        const shared = new SharedWorker(url);
        workerPort = shared.port;
        workerPort.start();
        workerPort.onmessage = handleWorkerMessage;
        return;
      }
    } catch (error) {
      console.warn('[HelenScheduler] No se pudo crear SharedWorker, usando Worker estándar.', error);
    }
    const dedicated = new Worker(url);
    workerPort = dedicated;
    dedicated.onmessage = handleWorkerMessage;
  };

  const list = (type) => {
    const items = Array.from(store.values());
    if (type) {
      return items.filter((item) => item.type === type).map((item) => clone(item));
    }
    return items.map((item) => clone(item));
  };

  const saveItem = (item) => {
    item.updatedAt = now();
    store.set(item.id, item);
    notifyUpdate();
  };

  const removeItem = (id) => {
    if (store.delete(id)) {
      notifyUpdate();
    }
  };

  const formatTimerLabel = (durationMs) => {
    if (!Number.isFinite(durationMs)) return 'Temporizador';
    const totalSeconds = Math.max(0, Math.round(durationMs / ONE_SECOND));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${hours}h ${minutes.toString().padStart(2, '0')}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds.toString().padStart(2, '0')}s`;
    }
    return `${seconds}s`;
  };

  const createAlarm = (options = {}) => {
    const hoursRaw = options.hours != null ? String(options.hours).padStart(2, '0') : '07';
    const minutesRaw = options.minutes != null ? String(options.minutes).padStart(2, '0') : '00';
    const ampm = String(options.ampm || 'AM').toUpperCase();

    let hours24 = toNumber(hoursRaw, 0);
    const minutes = clamp(toNumber(minutesRaw, 0), 0, 59);
    if (ampm === 'PM' && hours24 < 12) {
      hours24 += 12;
    }
    if (ampm === 'AM' && hours24 === 12) {
      hours24 = 0;
    }

    const metadata = {
      hours: hoursRaw.padStart(2, '0'),
      minutes: minutes.toString().padStart(2, '0'),
      ampm,
      hours24,
      repeatDays: Array.isArray(options.repeatDays) ? options.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day)) : [],
      active: options.active !== false,
      label: options.label ? String(options.label) : '',
    };

    const item = {
      id: options.id || uuid(),
      type: 'alarm',
      label: options.label ? String(options.label) : 'Alarma',
      createdAt: now(),
      updatedAt: now(),
      targetEpochMs: null,
      remainingMs: null,
      state: 'paused',
      metadata,
    };

    if (metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    }

    store.set(item.id, item);
    notifyUpdate();
    return item.id;
  };

  const updateAlarm = (id, updates = {}) => {
    const item = store.get(id);
    if (!item || item.type !== 'alarm') return;
    if (!item.metadata) item.metadata = {};

    if (updates.label != null) {
      item.label = String(updates.label);
    }
    if (updates.hours != null) {
      item.metadata.hours = String(updates.hours).padStart(2, '0');
    }
    if (updates.minutes != null) {
      item.metadata.minutes = String(updates.minutes).padStart(2, '0');
    }
    if (updates.ampm) {
      item.metadata.ampm = String(updates.ampm).toUpperCase();
    }
    if (updates.repeatDays) {
      item.metadata.repeatDays = Array.isArray(updates.repeatDays)
        ? updates.repeatDays.map((day) => Number(day)).filter((day) => Number.isFinite(day))
        : [];
    }

    const ampm = item.metadata.ampm || 'AM';
    let hours24 = toNumber(item.metadata.hours, 0);
    const minutes = clamp(toNumber(item.metadata.minutes, 0), 0, 59);
    if (ampm === 'PM' && hours24 < 12) hours24 += 12;
    if (ampm === 'AM' && hours24 === 12) hours24 = 0;
    item.metadata.hours24 = hours24;
    item.metadata.minutes = minutes.toString().padStart(2, '0');

    if (updates.active != null) {
      item.metadata.active = Boolean(updates.active);
    }

    if (item.metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    } else {
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'paused';
    }

    saveItem(item);
  };

  const toggleAlarm = (id, active) => {
    const item = store.get(id);
    if (!item || item.type !== 'alarm') return;
    item.metadata = item.metadata || {};
    item.metadata.active = Boolean(active);
    if (item.metadata.active) {
      const target = computeAlarmTarget(item);
      if (target) {
        item.targetEpochMs = target;
        item.remainingMs = target - now();
        item.state = 'running';
      }
    } else {
      item.targetEpochMs = null;
      item.remainingMs = null;
      item.state = 'paused';
    }
    saveItem(item);
  };

  const deleteAlarm = (id) => {
    removeItem(id);
  };

  const createTimer = (options = {}) => {
    const durationMs = Math.max(ONE_SECOND, toNumber(options.durationMs, 0));
    const label = options.label || formatTimerLabel(durationMs);
    const autoRemove = options.autoRemove !== false;
    const target = now() + durationMs;
    const item = {
      id: options.id || uuid(),
      type: 'timer',
      label,
      createdAt: now(),
      updatedAt: now(),
      targetEpochMs: target,
      remainingMs: durationMs,
      state: 'running',
      metadata: {
        durationMs,
        autoRemove,
      },
    };
    store.set(item.id, item);
    notifyUpdate();
    return item.id;
  };

  const pauseTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer' || item.state !== 'running') return;
    const remaining = Math.max(0, toNumber(item.targetEpochMs, now()) - now());
    item.state = 'paused';
    item.targetEpochMs = null;
    item.remainingMs = remaining;
    saveItem(item);
  };

  const resumeTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer') return;
    const remaining = Math.max(ONE_SECOND, toNumber(item.remainingMs, 0));
    item.targetEpochMs = now() + remaining;
    item.remainingMs = remaining;
    item.state = 'running';
    saveItem(item);
  };

  const resetTimer = (id) => {
    const item = store.get(id);
    if (!item || item.type !== 'timer') return;
    const duration = toNumber(item.metadata && item.metadata.durationMs, ONE_MINUTE);
    item.targetEpochMs = now() + duration;
    item.remainingMs = duration;
    item.state = 'running';
    saveItem(item);
  };

  const cancelTimer = (id) => {
    removeItem(id);
  };

  const hydrate = () => {
    const data = loadFromStorage();
    if (Array.isArray(data)) {
      data.forEach((item) => {
        if (!item || !item.id) return;
        store.set(item.id, item);
      });
    }
    reconcilePastDueItems();
    notifyUpdate();
  };

  setupWorker();
  hydrate();
  unlockAudioOnGesture();

  window.addEventListener('storage', (event) => {
    if (event.key === STORAGE_KEY && event.newValue) {
      store.clear();
      hydrate();
    }
  });

  window.HelenScheduler = {
    version: VERSION,
    ready: () => readyPromise,
    on,
    off,
    list,
    getItem: (id) => clone(store.get(id)),
    createAlarm,
    updateAlarm,
    toggleAlarm,
    deleteAlarm,
    createTimer,
    pauseTimer,
    resumeTimer,
    resetTimer,
    cancelTimer,
  };
})();
