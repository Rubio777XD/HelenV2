// eventConnector.js - Versión modificada
// Conexión al socket y mejoras en la navegación

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
let ringTransitionTimer;
let ringFadeTimer;
let ringErrorTimer;
let gestureConfirmTimer;
let currentRingState = 'idle';

const DEACTIVATION_DELAY = 3000;
const ABSOLUTE_URL_REGEX = /^(?:[a-z]+:)?\/\//i;
const RING_DETECTION_MS = 1000;
const RING_ACTIVE_LINGER_MS = 2200;
const RING_FADE_MS = 650;
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

const prefersReducedMotion = () => {
  if (typeof document !== 'undefined' && document.body && document.body.classList.contains('is-reduced-motion')) {
    return true;
  }

  if (typeof window.matchMedia !== 'function') return false;
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (error) {
    console.warn('No se pudo evaluar prefers-reduced-motion:', error);
    return false;
  }
};

const updateBodyActivationState = (isActive) => {
  if (!document || !document.body) {
    return;
  }
  document.body.classList.toggle('helen-active', Boolean(isActive));
  if (!isActive) {
    document.body.classList.remove('helen-gesture-confirmed');
  }
};

const triggerGestureConfirmationEffect = () => {
  if (!document || !document.body) {
    return;
  }
  document.body.classList.add('helen-gesture-confirmed');
  if (gestureConfirmTimer) {
    window.clearTimeout(gestureConfirmTimer);
  }
  gestureConfirmTimer = window.setTimeout(() => {
    gestureConfirmTimer = undefined;
    if (document && document.body) {
      document.body.classList.remove('helen-gesture-confirmed');
    }
  }, 600);
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

  const pulse = document.createElement('div');
  pulse.className = 'activation-ring__pulse';

  ring.appendChild(halo);
  ring.appendChild(pulse);

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
  if (ringTransitionTimer) {
    clearTimeout(ringTransitionTimer);
    ringTransitionTimer = undefined;
  }
  if (ringFadeTimer) {
    clearTimeout(ringFadeTimer);
    ringFadeTimer = undefined;
  }
  if (ringErrorTimer) {
    clearTimeout(ringErrorTimer);
    ringErrorTimer = undefined;
  }
  if (gestureConfirmTimer) {
    clearTimeout(gestureConfirmTimer);
    gestureConfirmTimer = undefined;
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

  const reduceMotion = prefersReducedMotion();
  const linger = typeof options.linger === 'number' ? options.linger : RING_ACTIVE_LINGER_MS;

  if (state === 'idle') {
    clearRingTimers();
    ring.classList.remove('is-detected', 'is-active', 'is-error', 'is-visible');
    currentRingState = 'idle';
    updateBodyActivationState(false);
    return;
  }

  updateBodyActivationState(true);
  ring.classList.add('is-visible');

  if (state === 'error') {
    clearRingTimers();
    currentRingState = 'error';
    ring.classList.remove('is-detected', 'is-active');
    ring.classList.add('is-error');
    ringErrorTimer = window.setTimeout(() => {
      if (currentRingState === 'error') {
        setRingState('idle');
      }
    }, RING_ERROR_MS);
    return;
  }

  if (state === 'active') {
    clearRingTimers();
    currentRingState = 'active';
    ring.classList.remove('is-detected', 'is-error');
    ring.classList.add('is-active');
    scheduleFadeOut(Math.max(linger, 0));
    return;
  }

  // state === 'detected'
  clearRingTimers();
  currentRingState = 'detected';
  ring.classList.remove('is-active', 'is-error');
  triggerGestureConfirmationEffect();

  if (reduceMotion) {
    ring.classList.remove('is-detected');
    ring.classList.add('is-active');
    currentRingState = 'active';
    scheduleFadeOut(Math.max(linger / 1.6, 400));
    return;
  }

  // Reiniciar animaciones CSS
  ring.classList.remove('is-detected');
  void ring.offsetWidth; // force reflow
  ring.classList.add('is-detected');

  ringTransitionTimer = window.setTimeout(() => {
    if (currentRingState !== 'detected') {
      return;
    }
    ring.classList.remove('is-detected');
    ring.classList.add('is-active');
    currentRingState = 'active';
    scheduleFadeOut(Math.max(linger, 0));
  }, RING_DETECTION_MS);
};

const triggerActivationAnimation = () => {
  setRingState('detected');
};

const triggerRingError = () => {
  setRingState('error');
};

const ALERT_VARIANTS = {
  success: { icon: 'bi-check2-circle', className: 'helen-alert--success' },
  warning: { icon: 'bi-exclamation-triangle', className: 'helen-alert--warning' },
  error: { icon: 'bi-x-circle', className: 'helen-alert--error' },
  info: { icon: 'bi-info-circle', className: 'helen-alert--info' },
};

const showPopup = (message, type = 'info') => {
  if (message === lastNotification) return;
  lastNotification = message;

  if (typeof Swal === 'undefined') {
    console.warn('Swal no está disponible. Mensaje:', message);
    if (type === 'error') {
      triggerRingError();
    }
    return;
  }

  const normalizedType = String(type || 'info').toLowerCase();
  const variant = ALERT_VARIANTS[normalizedType] || ALERT_VARIANTS.info;
  const safeMessage = Swal.escapeHtml ? Swal.escapeHtml(String(message)) : String(message);
  const iconMarkup = `<i class="bi ${variant.icon}" aria-hidden="true"></i>`;

  Swal.fire({
    toast: true,
    position: 'top',
    timer: 3200,
    timerProgressBar: true,
    showConfirmButton: false,
    background: 'transparent',
    html: `<div class="helen-alert__container"><span class="helen-alert__icon">${iconMarkup}</span><span class="helen-alert__text">${safeMessage}</span></div>`,
    customClass: {
      popup: `helen-alert ${variant.className}`,
      htmlContainer: 'helen-alert__content',
      timerProgressBar: 'helen-alert__progress',
    },
  });

  if (normalizedType === 'error') {
    triggerRingError();
  }
};

const resetDeactivationTimer = () => {
    if (timeoutId) {
        clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        isActive = false;
        console.log('Sistema desactivado automáticamente por inactividad.');
        showPopup('Sistema desactivado por inactividad.', 'warning');
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

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    ensureActivationRingElement();
  }, { once: true });
} else {
  ensureActivationRingElement();
}

