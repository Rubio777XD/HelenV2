// eventConnector.js - Versión modificada
// Conexión al socket y mejoras en la navegación

const DISPLAY_MODE_STORAGE_KEY = 'helen:display-mode';
const DEFAULT_DISPLAY_MODE = 'windows';
let currentDisplayMode = DEFAULT_DISPLAY_MODE;
const MODE_API_BASE = '/mode';
const MODE_GET_URL = `${MODE_API_BASE}/get`;
const MODE_SET_URL = `${MODE_API_BASE}/set`;

const raspberryFit = (() => {
  if (typeof window === 'undefined') {
    return { schedule: () => {}, measure: () => {}, reset: () => {} };
  }

  const SCALE_VAR = '--raspberry-scale';
  const SCALE_ATTR = 'data-raspberry-scaling';
  const ACTIVE_VALUE = 'active';
  const MIN_SCALE = 0.82;
  let frameId = null;

  const reset = () => {
    if (typeof document === 'undefined') {
      return;
    }
    if (document.body) {
      document.body.removeAttribute(SCALE_ATTR);
    }
    if (document.documentElement) {
      document.documentElement.style.removeProperty(SCALE_VAR);
    }
  };

  const measure = () => {
    if (typeof document === 'undefined' || typeof window === 'undefined') {
      return;
    }

    const body = document.body;
    if (!body || body.getAttribute('data-mode') !== 'raspberry') {
      reset();
      return;
    }

    const candidates = Array.from(document.querySelectorAll('[data-raspberry-fit-root]'));
    const targets = candidates.length ? candidates : (body ? [body] : []);

    if (!targets.length) {
      reset();
      return;
    }

    const viewportWidth = Math.max(window.innerWidth || 0, 1);
    const viewportHeight = Math.max(window.innerHeight || 0, 1);

    let contentWidth = 0;
    let contentHeight = 0;

    targets.forEach((target) => {
      if (!target) return;
      const rect = target.getBoundingClientRect();
      const width = Math.max(rect.width, target.scrollWidth);
      const height = Math.max(rect.height, target.scrollHeight);
      contentWidth = Math.max(contentWidth, width);
      contentHeight = Math.max(contentHeight, height);
    });

    if (!contentWidth || !contentHeight) {
      reset();
      return;
    }

    const scale = Math.min(1, viewportWidth / contentWidth, viewportHeight / contentHeight);
    if (!Number.isFinite(scale) || scale >= 0.995) {
      reset();
      return;
    }

    const finalScale = Math.max(scale, MIN_SCALE);
    if (document.documentElement) {
      document.documentElement.style.setProperty(SCALE_VAR, finalScale.toFixed(4));
    }
    body.setAttribute(SCALE_ATTR, ACTIVE_VALUE);
  };

  const schedule = () => {
    if (typeof window.requestAnimationFrame !== 'function') {
      measure();
      return;
    }
    if (frameId) {
      window.cancelAnimationFrame(frameId);
    }
    frameId = window.requestAnimationFrame(() => {
      frameId = null;
      measure();
    });
  };

  const queue = () => schedule();

  window.addEventListener('resize', queue);
  window.addEventListener('orientationchange', queue);
  window.addEventListener('load', queue);
  window.addEventListener('helen:display-mode', (event) => {
    if (event && event.detail && event.detail.mode === 'raspberry') {
      schedule();
    }
  });

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => schedule(), { once: true });
    } else {
      schedule();
    }
  }

  return { schedule, measure, reset };
})();

if (typeof window !== 'undefined') {
  window.HelenRaspberryFit = {
    refresh: () => raspberryFit.schedule(),
    measure: () => raspberryFit.measure(),
    reset: () => raspberryFit.reset(),
  };
}

const normalizeDisplayMode = (value) => (value === 'raspberry' ? 'raspberry' : DEFAULT_DISPLAY_MODE);

const dispatchDisplayModeChange = (mode) => {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    const event = new CustomEvent('helen:display-mode', { detail: { mode } });
    window.dispatchEvent(event);
  } catch (error) {
    if (typeof window.dispatchEvent === 'function') {
      window.dispatchEvent({ type: 'helen:display-mode', detail: { mode } });
    }
  }
};

const applyDisplayMode = (mode, options = {}) => {
  const { silent = false } = options || {};
  const normalized = normalizeDisplayMode(mode);
  currentDisplayMode = normalized;

  if (typeof document !== 'undefined') {
    if (document.documentElement) {
      document.documentElement.setAttribute('data-mode', normalized);
    }
    if (document.body) {
      document.body.setAttribute('data-mode', normalized);
    } else if (typeof document.addEventListener === 'function') {
      document.addEventListener(
        'DOMContentLoaded',
        () => {
          if (document.body) {
            document.body.setAttribute('data-mode', currentDisplayMode);
            if (currentDisplayMode === 'raspberry') {
              raspberryFit.schedule();
            }
          }
        },
        { once: true },
      );
    }
  }

  if (normalized === 'raspberry') {
    raspberryFit.schedule();
  } else {
    raspberryFit.reset();
  }

  if (!silent) {
    dispatchDisplayModeChange(normalized);
  }

  return normalized;
};

const readStoredDisplayMode = () => {
  if (typeof localStorage === 'undefined') {
    return DEFAULT_DISPLAY_MODE;
  }
  try {
    const stored = localStorage.getItem(DISPLAY_MODE_STORAGE_KEY);
    return stored ? normalizeDisplayMode(stored) : DEFAULT_DISPLAY_MODE;
  } catch (error) {
    console.warn('[Helen] No se pudo leer el modo de visualización:', error);
    return DEFAULT_DISPLAY_MODE;
  }
};

const fetchBackendDisplayMode = () => {
  if (typeof fetch !== 'function') {
    return Promise.resolve(null);
  }
  return fetch(MODE_GET_URL, { cache: 'no-store' })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .catch((error) => {
      console.warn('[Helen] No se pudo sincronizar el modo con el backend:', error);
      return null;
    });
};

const persistDisplayModeToBackend = (mode) => {
  if (typeof fetch !== 'function') {
    return Promise.resolve(null);
  }
  return fetch(MODE_SET_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
    cache: 'no-store',
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => (data && typeof data.mode === 'string' ? normalizeDisplayMode(data.mode) : null))
    .catch((error) => {
      console.warn('[Helen] No se pudo actualizar el modo en el backend:', error);
      return null;
    });
};

const setDisplayMode = (mode) => {
  const normalized = applyDisplayMode(mode);
  if (typeof localStorage !== 'undefined') {
    try {
      localStorage.setItem(DISPLAY_MODE_STORAGE_KEY, normalized);
    } catch (error) {
      console.warn('[Helen] No se pudo persistir el modo de visualización:', error);
    }
  }
  persistDisplayModeToBackend(normalized).then((backendMode) => {
    if (!backendMode) {
      return;
    }
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(DISPLAY_MODE_STORAGE_KEY, backendMode);
      } catch (error) {
        console.warn('[Helen] No se pudo sincronizar el modo almacenado:', error);
      }
    }
    if (backendMode !== currentDisplayMode) {
      applyDisplayMode(backendMode);
    }
  });
  return normalized;
};

const ensureDisplayMode = () => {
  const stored = readStoredDisplayMode();
  applyDisplayMode(stored, { silent: true });
  fetchBackendDisplayMode().then((snapshot) => {
    if (!snapshot || typeof snapshot.active !== 'string') {
      return;
    }
    const backendMode = normalizeDisplayMode(snapshot.active);
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(DISPLAY_MODE_STORAGE_KEY, backendMode);
      } catch (error) {
        console.warn('[Helen] No se pudo sincronizar el modo almacenado:', error);
      }
    }
    if (backendMode !== currentDisplayMode) {
      applyDisplayMode(backendMode);
    }
  });
  return stored;
};

const initialDisplayMode = ensureDisplayMode();

if (typeof window !== 'undefined') {
  window.HelenDisplayMode = {
    current: () => currentDisplayMode,
    set: setDisplayMode,
    apply: (mode, options) => applyDisplayMode(mode, options),
    storageKey: DISPLAY_MODE_STORAGE_KEY,
  };

  window.addEventListener('storage', (event) => {
    if (event && event.key === DISPLAY_MODE_STORAGE_KEY && event.newValue) {
      applyDisplayMode(event.newValue);
    }
  });

  window.addEventListener('helen:display-mode:request-sync', () => {
    dispatchDisplayModeChange(currentDisplayMode);
  });

  dispatchDisplayModeChange(initialDisplayMode);
}

// Inicializar la conexión al socket (SSE)
const DEFAULT_SOCKET_URL = 'http://127.0.0.1:5000';

const resolveSocketUrl = () => {
  if (typeof window === 'undefined') {
    return DEFAULT_SOCKET_URL;
  }

  const { HELEN_SOCKET_URL, HELEN_SOCKET_HOST, HELEN_SOCKET_PORT, HELEN_SOCKET_PROTOCOL } = window;

  if (typeof HELEN_SOCKET_URL === 'string' && HELEN_SOCKET_URL.trim()) {
    return HELEN_SOCKET_URL.trim();
  }

  const hostnameRaw = (typeof HELEN_SOCKET_HOST === 'string' && HELEN_SOCKET_HOST.trim())
    ? HELEN_SOCKET_HOST.trim()
    : (window.location && window.location.hostname ? window.location.hostname : '127.0.0.1');

  if (!hostnameRaw) {
    return DEFAULT_SOCKET_URL;
  }

  let hostnamePart = hostnameRaw;
  let portFromHost = '';

  const isBracketedIPv6 = hostnameRaw.startsWith('[') && hostnameRaw.includes(']');

  if (isBracketedIPv6) {
    const closingIndex = hostnameRaw.indexOf(']');
    hostnamePart = hostnameRaw.slice(0, closingIndex + 1);
    if (hostnameRaw.length > closingIndex + 1 && hostnameRaw[closingIndex + 1] === ':') {
      portFromHost = hostnameRaw.slice(closingIndex + 2);
    }
  } else {
    const colonMatches = hostnameRaw.match(/:/g) || [];
    if (colonMatches.length === 1) {
      const splitIndex = hostnameRaw.lastIndexOf(':');
      hostnamePart = hostnameRaw.slice(0, splitIndex);
      portFromHost = hostnameRaw.slice(splitIndex + 1);
    }
  }

  if (!hostnamePart) {
    hostnamePart = '127.0.0.1';
  }

  portFromHost = portFromHost.trim();

  const hostname = hostnamePart.includes(':') && !hostnamePart.startsWith('[')
    ? `[${hostnamePart}]`
    : hostnamePart;

  const protocolCandidate = (typeof HELEN_SOCKET_PROTOCOL === 'string' && HELEN_SOCKET_PROTOCOL.trim())
    ? HELEN_SOCKET_PROTOCOL.trim().replace(/:$/, '')
    : (window.location && typeof window.location.protocol === 'string' && window.location.protocol.startsWith('http')
      ? window.location.protocol.replace(/:$/, '')
      : 'http');

  const portValue = HELEN_SOCKET_PORT != null ? HELEN_SOCKET_PORT : (portFromHost || 5000);
  const normalizedPort = String(portValue).trim().replace(/^:/, '') || '5000';

  const portSuffix = normalizedPort ? `:${normalizedPort}` : '';

  return `${protocolCandidate}://${hostname}${portSuffix}`.replace(/\/$/, '');
};

const SOCKET_URL = resolveSocketUrl();
const EVENTS_URL = `${SOCKET_URL}/events`;

const createNoopSocket = () => ({
  on(eventName, handler) {
    console.warn(`[Helen] Socket.IO no disponible. No se escuchará "${eventName}".`);
    return this;
  },
  emit(eventName, payload) {
    console.warn('[Helen] Socket.IO no disponible. Emisión ignorada.', eventName, payload);
    return this;
  },
  close() {
    return undefined;
  }
});

const socket = (() => {
  if (typeof createHelenSocket !== 'function') {
    console.warn('[Helen] Adaptador SSE no disponible. Ejecutando en modo sin conexión.');
    return createNoopSocket();
  }

  if (window.socket && typeof window.socket.on === 'function') {
    return window.socket;
  }

  try {
    const instance = createHelenSocket(EVENTS_URL);
    instance.on('connect', () => console.info('[Helen] Conexión SSE establecida:', EVENTS_URL));
    instance.on('disconnect', () => console.warn('[Helen] Conexión SSE perdida. Intentando reconectar...'));
    window.socket = instance;
    return instance;
  } catch (connectionError) {
    console.error('[Helen] No se pudo iniciar la conexión SSE:', connectionError);
    return createNoopSocket();
  }
})();

window.socket = socket;

let isActive = false;
let timeoutId;
let lastNotification = "";
let ringFadeTimer;
let ringErrorTimer;
let currentRingState = 'idle';

const DEACTIVATION_DELAY = 3000;
const ABSOLUTE_URL_REGEX = /^(?:[a-z]+:)?\/\//i;
const RING_ACTIVE_LINGER_MS = 5000;
const RING_ERROR_MS = 900;

const resolveTargetUrl = (targetUrl = '') => {
  if (!targetUrl) return null;

  if (ABSOLUTE_URL_REGEX.test(targetUrl)) {
    return targetUrl;
  }

  const sanitizedTarget = targetUrl.replace(/^\.\/+/, '');
  const pathname = window.location.pathname.replace(/\\/g, '/');
  const segments = pathname.split('/');
  const pagesIndex = segments.lastIndexOf('pages');
  const inPagesDirectory = pagesIndex !== -1;

  const ensureLeadingSlash = (parts) => {
    if (!parts.length) {
      return '';
    }
    const clone = [...parts];
    if (clone[0] !== '') {
      clone.unshift('');
    }
    return clone.join('/');
  };

  if (!sanitizedTarget.includes('/')) {
    if (sanitizedTarget === 'index.html') {
      const rootSegments = inPagesDirectory ? segments.slice(0, pagesIndex) : segments.slice(0, -1);
      if (!rootSegments.length) {
        return sanitizedTarget.startsWith('/') ? sanitizedTarget : `/${sanitizedTarget}`;
      }
      return `${ensureLeadingSlash(rootSegments)}/${sanitizedTarget}`;
    }

    if (inPagesDirectory) {
      const currentDirSegments = segments.slice(0, -1);
      if (!currentDirSegments.length) {
        return sanitizedTarget;
      }
      return `${ensureLeadingSlash(currentDirSegments)}/${sanitizedTarget}`;
    }

    const baseSegments = segments.slice(0, -1);
    if (!baseSegments.length) {
      return sanitizedTarget.startsWith('/') ? sanitizedTarget : `/${sanitizedTarget}`;
    }
    return `${ensureLeadingSlash(baseSegments)}/${sanitizedTarget}`;
  }

  const baseSegments = inPagesDirectory ? segments.slice(0, pagesIndex) : segments.slice(0, -1);
  if (!baseSegments.length) {
    return sanitizedTarget.startsWith('/') ? sanitizedTarget : `/${sanitizedTarget}`;
  }
  return `${ensureLeadingSlash(baseSegments)}/${sanitizedTarget.replace(/^\/+/, '')}`;
};

const updateBodyActivationState = (isActive) => {
  if (!document || !document.body) {
    return;
  }
  document.body.classList.toggle('helen-active', Boolean(isActive));
};

const ensureActivationRingElement = () => {
  let ring = document.querySelector('.activation-ring');
  if (ring) {
    return ring;
  }

  ring = document.createElement('div');
  ring.className = 'activation-ring';
  ring.setAttribute('aria-hidden', 'true');

  const halo = document.createElement('div');
  halo.className = 'activation-ring__halo';

  ring.appendChild(halo);

  const attach = () => {
    if (!document.body) {
      return;
    }
    document.body.appendChild(ring);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach, { once: true });
  } else {
    attach();
  }

  return ring;
};

const clearRingTimers = () => {
  if (ringFadeTimer) {
    clearTimeout(ringFadeTimer);
    ringFadeTimer = undefined;
  }
  if (ringErrorTimer) {
    clearTimeout(ringErrorTimer);
    ringErrorTimer = undefined;
  }
};

const scheduleFadeOut = (delayMs) => {
  if (ringFadeTimer) {
    clearTimeout(ringFadeTimer);
  }
  ringFadeTimer = window.setTimeout(() => {
    if (currentRingState !== 'error') {
      setRingState('idle');
    }
  }, Math.max(0, delayMs));
};

const setRingState = (state, options = {}) => {
  const ring = ensureActivationRingElement();
  if (!ring) {
    return;
  }

  const linger = typeof options.linger === 'number' ? options.linger : RING_ACTIVE_LINGER_MS;
  const persist = Boolean(options.persist);

  if (state === 'idle') {
    clearRingTimers();
    ring.classList.remove('is-active', 'is-error', 'is-visible', 'is-persistent');
    currentRingState = 'idle';
    updateBodyActivationState(false);
    return;
  }

  updateBodyActivationState(true);
  ring.classList.add('is-visible');
  clearRingTimers();

  if (state === 'error') {
    currentRingState = 'error';
    ring.classList.remove('is-active', 'is-persistent');
    ring.classList.add('is-error');
    ringErrorTimer = window.setTimeout(() => {
      if (currentRingState === 'error') {
        setRingState('idle');
      }
    }, RING_ERROR_MS);
    return;
  }

  currentRingState = 'active';
  ring.classList.remove('is-error');
  ring.classList.add('is-active');
  ring.classList.toggle('is-persistent', persist);

  if (!persist && linger > 0) {
    scheduleFadeOut(Math.max(linger, 0));
  }
};

const triggerActivationAnimation = (options = {}) => {
  setRingState('active', options);
};

const triggerRingError = () => {
  setRingState('error');
};

const showPopup = (message, type = 'info') => {
  const normalizedType = String(type || 'info').toLowerCase();

  if (normalizedType === 'error') {
    triggerRingError();
  }

  if (!message) {
    lastNotification = '';
    return;
  }

  if (message === lastNotification) {
    return;
  }

  lastNotification = message;
  console.info(`[Helen][${normalizedType}] ${message}`);
};

const resetDeactivationTimer = () => {
    if (timeoutId) {
        clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        isActive = false;
        console.log('Sistema desactivado automáticamente por inactividad.');
        setRingState('idle');
    }, DEACTIVATION_DELAY);
};
const goToPageWithLoading = (targetUrl, pageName) => {
    const currentUrl = window.location.href;
    if (currentUrl.includes(targetUrl)) {
        console.log(`Ya estás en ${targetUrl}, no se necesita redirección.`);
        return;
    }
    
    const resolvedTarget = resolveTargetUrl(targetUrl);
    if (!resolvedTarget) {
        console.warn('No se pudo resolver la ruta de destino:', targetUrl);
        return;
    }

    let resolvedHref = resolvedTarget;
    try {
        resolvedHref = new URL(resolvedTarget, window.location.href).href;
    } catch (error) {
        console.warn('Error resolviendo URL destino:', error);
    }

    const currentHref = window.location.href;
    if (currentHref === resolvedHref) {
        console.log(`Ya estás en ${resolvedHref}, no se necesita redirección.`);
        return;
    }

    const performNavigation = () => {
        if (window.myAPI && typeof window.myAPI.navigate === 'function') {
            window.myAPI.navigate(targetUrl);
        } else {
            window.location.href = resolvedHref;
        }
    };

    const navigateWithLoading = async () => {
        await new Promise(resolve => setTimeout(resolve, 800));
        console.log(`Navegando a: ${resolvedHref}`);
        performNavigation();
    };

    if (window.loadingScreen && typeof window.loadingScreen.showAndExecute === 'function') {
        console.log(`Mostrando pantalla de carga para: ${pageName}`);
        window.loadingScreen.showAndExecute(navigateWithLoading, `Cargando ${pageName}...`);
    } else {
        console.log(`Navegando a: ${resolvedHref}`);
        performNavigation();
    }
};

const enhancedGoToClock = () => goToPageWithLoading("pages/clock/clock.html", "Reloj");
const enhancedGoToWeather = () => goToPageWithLoading("pages/weather/weather.html", "Clima");
const enhancedGoToDevices = () => goToPageWithLoading("pages/devices/devices.html", "Dispositivos");
const enhancedGoToHome = () => goToPageWithLoading("index.html", "Inicio");
const enhancedGoToAlarm = () => goToPageWithLoading("pages/clock/alarm.html", "Alarma");
const enhancedGoToSettings = () => goToPageWithLoading("pages/settings/settings.html", "Ajustes");

window.goToClock = enhancedGoToClock;
window.goToWeather = enhancedGoToWeather;
window.goToDevices = enhancedGoToDevices;
window.goToHome = enhancedGoToHome;
window.goToAlarm = enhancedGoToAlarm;
window.goToSettings = enhancedGoToSettings;
window.triggerActivationAnimation = triggerActivationAnimation;
window.triggerRingError = triggerRingError;
window.setActivationRingState = setRingState;
window.goToPageWithLoading = goToPageWithLoading;

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    clearRingTimers();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    ensureActivationRingElement();
  }, { once: true });
} else {
  ensureActivationRingElement();
}

