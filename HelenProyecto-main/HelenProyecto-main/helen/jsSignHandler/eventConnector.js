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
let activationTimeoutId;

const DEACTIVATION_DELAY = 3000;
const DEFAULT_ACTIVATION_RING_DURATION = 2600;
const ABSOLUTE_URL_REGEX = /^(?:[a-z]+:)?\/\//i;

const parseCssTimeToMs = (value) => {
  if (typeof value !== 'string') {
    return NaN;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return NaN;
  }

  const millisecondsMatch = trimmed.match(/^(-?\d*\.?\d+)ms$/i);
  if (millisecondsMatch) {
    return Number(millisecondsMatch[1]);
  }

  const secondsMatch = trimmed.match(/^(-?\d*\.?\d+)s$/i);
  if (secondsMatch) {
    return Number(secondsMatch[1]) * 1000;
  }

  const numericValue = Number(trimmed);
  return Number.isFinite(numericValue) ? numericValue : NaN;
};

const getActivationRingDuration = (ringElement) => {
  if (typeof window === 'undefined' || !ringElement) {
    return DEFAULT_ACTIVATION_RING_DURATION;
  }

  try {
    const computedStyle = window.getComputedStyle(ringElement);
    const candidates = [
      computedStyle.getPropertyValue('--activation-ring-duration'),
      computedStyle.getPropertyValue('animation-duration'),
    ];

    for (const candidate of candidates) {
      const values = String(candidate || '')
        .split(',')
        .map((token) => token.trim())
        .filter(Boolean);

      for (const value of values.length ? values : [candidate]) {
        const parsed = parseCssTimeToMs(value);
        if (Number.isFinite(parsed) && parsed > 0) {
          return parsed;
        }
      }
    }
  } catch (error) {
    console.warn('No se pudo determinar la duración del aro de activación:', error);
  }

  return DEFAULT_ACTIVATION_RING_DURATION;
};

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
  if (typeof window.matchMedia !== 'function') return false;
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch (error) {
    console.warn('No se pudo evaluar prefers-reduced-motion:', error);
    return false;
  }
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

const triggerActivationAnimation = () => {
  if (prefersReducedMotion()) {
    return;
  }

  const ring = ensureActivationRingElement();
  if (!ring) {
    return;
  }

  if (activationTimeoutId) {
    clearTimeout(activationTimeoutId);
  }

  ring.classList.remove('is-active');
  // Forzar reflujo para reiniciar las animaciones CSS
  void ring.offsetWidth;
  ring.classList.add('is-active');

  const animationDuration = getActivationRingDuration(ring);
  activationTimeoutId = window.setTimeout(() => {
    ring.classList.remove('is-active');
  }, animationDuration);
};

const showPopup = (message, type) => {
    if (message === lastNotification) return;
    lastNotification = message;
    if (typeof Swal === 'undefined') {
        console.warn('Swal no está disponible. Mensaje:', message);
        return;
    }

    Swal.fire({
        title: message,
        icon: type,
        showConfirmButton: false,
        timer: 3000,
        toast: true,
        position: 'top-end',
    });
};

const resetDeactivationTimer = () => {
    if (timeoutId) {
        clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        isActive = false;
        console.log('Sistema desactivado automáticamente por inactividad.');
        showPopup('Sistema desactivado por inactividad.', 'warning');
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
window.goToPageWithLoading = goToPageWithLoading;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (!prefersReducedMotion()) {
      ensureActivationRingElement();
    }
  }, { once: true });
} else if (!prefersReducedMotion()) {
  ensureActivationRingElement();
}

