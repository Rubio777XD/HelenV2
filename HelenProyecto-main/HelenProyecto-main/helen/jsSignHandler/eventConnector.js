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

const activationRingLayout = (() => {
  const DEFAULTS = {
    enabled: true,
    maxScale: 0.92,
    safePadding: 12,
    zIndexBase: 0,
  };

  let options = { ...DEFAULTS };
  let resizeObserver = null;
  let observedTarget = null;
  let rafId = null;

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const viewportMetrics = () => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0, left: 0, top: 0 };
    }
    const viewport = window.visualViewport;
    if (viewport) {
      return {
        width: viewport.width || window.innerWidth || 0,
        height: viewport.height || window.innerHeight || 0,
        left: viewport.offsetLeft || 0,
        top: viewport.offsetTop || 0,
      };
    }
    const doc = window.document && window.document.documentElement;
    return {
      width: window.innerWidth || (doc && doc.clientWidth) || 0,
      height: window.innerHeight || (doc && doc.clientHeight) || 0,
      left: 0,
      top: 0,
    };
  };

  const schedule = () => {
    if (rafId !== null) {
      return;
    }
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      rafId = window.requestAnimationFrame(() => {
        rafId = null;
        applyLayout();
      });
      return;
    }
    rafId = window.setTimeout(() => {
      rafId = null;
      applyLayout();
    }, 16);
  };

  const stopObserving = () => {
    if (resizeObserver && observedTarget) {
      resizeObserver.unobserve(observedTarget);
      observedTarget = null;
    }
  };

  const observeContainer = (node) => {
    if (typeof ResizeObserver !== 'function' || !node) {
      observedTarget = node;
      return;
    }
    if (!resizeObserver) {
      resizeObserver = new ResizeObserver(() => schedule());
    }
    if (observedTarget && observedTarget !== node) {
      resizeObserver.unobserve(observedTarget);
    }
    resizeObserver.observe(node);
    observedTarget = node;
  };

  const resetLayout = () => {
    const ring = document.querySelector('.activation-ring');
    if (!ring) {
      return;
    }
    ring.classList.remove('activation-ring--fit');
    ring.classList.toggle('activation-ring--disabled', !options.enabled);
    ring.style.removeProperty('--activation-ring-left');
    ring.style.removeProperty('--activation-ring-top');
    ring.style.removeProperty('--activation-ring-diameter');
    ring.style.removeProperty('--activation-ring-safe-padding');
    ring.style.removeProperty('--activation-ring-blur');
    ring.style.removeProperty('--activation-ring-radius');
    ring.style.removeProperty('--activation-ring-z-index');
    ring.style.removeProperty('--activation-ring-max-width');
    ring.style.removeProperty('--activation-ring-max-height');
  };

  const selectContainer = () => {
    if (typeof document === 'undefined') {
      return null;
    }
    const body = document.body;
    if (!body || body.getAttribute('data-mode') !== 'raspberry') {
      return null;
    }

    const candidates = Array.from(document.querySelectorAll('[data-raspberry-fit-root]'));
    if (!candidates.length) {
      const fallback = document.querySelector('.home-hero');
      return fallback || body;
    }

    let best = null;
    let bestArea = 0;
    candidates.forEach((node) => {
      if (!node || typeof node.getBoundingClientRect !== 'function') {
        return;
      }
      const rect = node.getBoundingClientRect();
      const area = Math.max(rect.width, 0) * Math.max(rect.height, 0);
      if (area > bestArea) {
        bestArea = area;
        best = node;
      }
    });
    return best || candidates[0];
  };

  const applyLayout = () => {
    const ring = ensureActivationRingElement();
    if (!ring) {
      return;
    }
    const halo = ring.querySelector('.activation-ring__halo');
    if (!halo) {
      return;
    }

    if (!options.enabled) {
      stopObserving();
      ring.classList.add('activation-ring--disabled');
      ring.classList.remove('activation-ring--fit');
      return;
    }

    const container = selectContainer();
    if (!container) {
      stopObserving();
      resetLayout();
      return;
    }

    ring.classList.remove('activation-ring--disabled');
    observeContainer(container);
    const rect = container.getBoundingClientRect();
    const metrics = viewportMetrics();

    const basePadding = Math.max(0, Number(options.safePadding) || 0);
    const minDimension = Math.max(0, Math.min(rect.width, rect.height));
    const viewportMinDimension = Math.max(0, Math.min(metrics.width, metrics.height));
    const dynamicPadding = Math.max(basePadding, Math.round(Math.min(minDimension, viewportMinDimension) * 0.045));
    const paddingLimit = Math.max(basePadding, Math.round(Math.max(minDimension, viewportMinDimension) * 0.25));
    const padding = clamp(dynamicPadding, basePadding, paddingLimit || basePadding);

    const rectLeft = rect.left + metrics.left;
    const rectTop = rect.top + metrics.top;
    const rectRight = rect.right + metrics.left;
    const rectBottom = rect.bottom + metrics.top;

    const viewportLeft = metrics.left;
    const viewportTop = metrics.top;
    const viewportRight = metrics.left + metrics.width;
    const viewportBottom = metrics.top + metrics.height;

    const visibleLeft = Math.max(rectLeft, viewportLeft);
    const visibleRight = Math.min(rectRight, viewportRight);
    const visibleTop = Math.max(rectTop, viewportTop);
    const visibleBottom = Math.min(rectBottom, viewportBottom);

    const visibleWidth = Math.max(0, visibleRight - visibleLeft);
    const visibleHeight = Math.max(0, visibleBottom - visibleTop);
    const containerWidth = Math.max(0, rect.width - padding * 2);
    const containerHeight = Math.max(0, rect.height - padding * 2);
    const viewportWidth = Math.max(0, metrics.width - padding * 2);
    const viewportHeight = Math.max(0, metrics.height - padding * 2);

    const constraints = [
      containerWidth,
      containerHeight,
      Math.max(0, visibleWidth - padding * 2),
      Math.max(0, visibleHeight - padding * 2),
      viewportWidth,
      viewportHeight,
    ].filter((value) => Number.isFinite(value) && value > 0);

    if (!constraints.length) {
      resetLayout();
      return;
    }

    const limitingSize = Math.min(...constraints);

    const scale = clamp(Number(options.maxScale) || 1, 0.35, 0.96);
    const diameter = Math.max(0, limitingSize * scale);
    if (!diameter) {
      resetLayout();
      return;
    }

    let centerXAbs = rectLeft + rect.width / 2;
    let centerYAbs = rectTop + rect.height / 2;
    const minCenterX = viewportLeft + padding + diameter / 2;
    const maxCenterX = viewportRight - padding - diameter / 2;
    const minCenterY = viewportTop + padding + diameter / 2;
    const maxCenterY = viewportBottom - padding - diameter / 2;

    if (minCenterX <= maxCenterX) {
      centerXAbs = clamp(centerXAbs, minCenterX, maxCenterX);
    }
    if (minCenterY <= maxCenterY) {
      centerYAbs = clamp(centerYAbs, minCenterY, maxCenterY);
    }

    const centerX = centerXAbs - viewportLeft;
    const centerY = centerYAbs - viewportTop;
    const blurBase = Math.max(18, Math.min(72, padding * 1.2));
    const radius = Math.max(32, diameter / 2);

    let targetZ = typeof options.zIndexBase === 'number' ? options.zIndexBase : 0;
    if (typeof window !== 'undefined' && window.getComputedStyle) {
      const computed = window.getComputedStyle(container);
      const parsed = parseInt(computed && computed.zIndex, 10);
      if (!Number.isNaN(parsed)) {
        targetZ = Math.max(targetZ, parsed + 2);
      } else {
        targetZ = Math.max(targetZ, 12);
      }
    }

    ring.classList.add('activation-ring--fit');
    ring.style.setProperty('--activation-ring-z-index', String(targetZ));
    ring.style.setProperty('--activation-ring-left', `${centerX}px`);
    ring.style.setProperty('--activation-ring-top', `${centerY}px`);
    ring.style.setProperty('--activation-ring-diameter', `${diameter}px`);
    ring.style.setProperty('--activation-ring-safe-padding', `${padding}px`);
    ring.style.setProperty('--activation-ring-blur', `${blurBase}px`);
    ring.style.setProperty('--activation-ring-radius', `${radius}px`);
    ring.style.setProperty('--activation-ring-max-width', `${Math.max(0, Math.min(containerWidth, viewportWidth, visibleWidth))}px`);
    ring.style.setProperty('--activation-ring-max-height', `${Math.max(0, Math.min(containerHeight, viewportHeight, visibleHeight))}px`);
  };

  const configure = (overrides = {}) => {
    if (!overrides || typeof overrides !== 'object') {
      return { ...options };
    }
    const next = { ...options };
    if (Object.prototype.hasOwnProperty.call(overrides, 'enabled')) {
      next.enabled = Boolean(overrides.enabled);
    }
    if (typeof overrides.maxScale === 'number') {
      next.maxScale = overrides.maxScale;
    }
    if (typeof overrides.safePadding === 'number') {
      next.safePadding = overrides.safePadding;
    }
    if (typeof overrides.zIndexBase === 'number') {
      next.zIndexBase = overrides.zIndexBase;
    }
    options = next;
    schedule();
    return { ...options };
  };

  const refresh = () => {
    schedule();
    return { ...options };
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('resize', schedule);
    window.addEventListener('orientationchange', schedule);
    window.addEventListener('helen:display-mode', (event) => {
      const mode = event && event.detail ? event.detail.mode : undefined;
      if (mode === 'raspberry') {
        schedule();
      } else {
        stopObserving();
        resetLayout();
      }
    });
    if (window.visualViewport) {
      ['resize', 'scroll'].forEach((eventName) => {
        window.visualViewport.addEventListener(eventName, schedule);
      });
    }
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => schedule(), { once: true });
    } else {
      schedule();
    }
  }

  return {
    configure,
    refresh,
    options: () => ({ ...options }),
  };
})();

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

  activationRingLayout.refresh();

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
window.HelenActivationRing = {
  configure: (overrides) => activationRingLayout.configure(overrides),
  refresh: () => activationRingLayout.refresh(),
  options: () => activationRingLayout.options(),
};

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    clearRingTimers();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    ensureActivationRingElement();
    activationRingLayout.refresh();
  }, { once: true });
} else {
  ensureActivationRingElement();
  activationRingLayout.refresh();
}

