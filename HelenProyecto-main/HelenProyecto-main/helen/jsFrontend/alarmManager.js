(function (global) {
  'use strict';

  if (!global) {
    return;
  }

  if (global.HelenAlarmManager && global.HelenAlarmManager.version >= 1) {
    return;
  }

  const documentRef = global.document || null;
  const pendingScripts = new Map();
  const state = {
    readyPromise: null,
    listenersAttached: false,
    busUnsubscribe: null,
    windowListener: null,
  };

  const resolveScriptUrl = () => {
    if (documentRef && documentRef.currentScript && documentRef.currentScript.src) {
      return documentRef.currentScript.src;
    }
    if (!documentRef) {
      return null;
    }
    const scripts = documentRef.getElementsByTagName('script');
    for (let index = scripts.length - 1; index >= 0; index -= 1) {
      const candidate = scripts[index];
      if (candidate && typeof candidate.src === 'string' && candidate.src.includes('alarmManager.js')) {
        return candidate.src;
      }
    }
    return null;
  };

  const baseScriptUrl = resolveScriptUrl();

  const resolveAsset = (relativePath) => {
    if (!relativePath) {
      return relativePath;
    }
    try {
      if (baseScriptUrl) {
        return new URL(relativePath, baseScriptUrl).toString();
      }
      if (global.location && global.location.href) {
        return new URL(relativePath, global.location.href).toString();
      }
    } catch (error) {
      // Fallback to returning the original path when URL construction fails.
    }
    return relativePath;
  };

  const urls = {
    notifications: resolveAsset('../ui/notifications/notifications.js'),
    alarmCore: resolveAsset('../pages/jsFrontend/alarm-core.js'),
  };

  const loadScript = (src) => {
    if (!src || !documentRef) {
      return Promise.resolve();
    }
    if (pendingScripts.has(src)) {
      return pendingScripts.get(src);
    }

    const promise = new Promise((resolve, reject) => {
      const script = documentRef.createElement('script');
      script.src = src;
      script.defer = true;
      script.addEventListener('load', () => resolve(), { once: true });
      script.addEventListener(
        'error',
        () => reject(new Error(`[HelenAlarmManager] No se pudo cargar ${src}`)),
        { once: true },
      );
      documentRef.head.appendChild(script);
    }).finally(() => {
      pendingScripts.delete(src);
    });

    pendingScripts.set(src, promise);
    return promise;
  };

  const ensureNotifications = () => {
    if (global.HelenNotifications && typeof global.HelenNotifications.ensure === 'function') {
      try {
        global.HelenNotifications.ensure();
      } catch (error) {
        console.warn('[HelenAlarmManager] No se pudo inicializar el popup global:', error);
      }
      return Promise.resolve(global.HelenNotifications);
    }
    return loadScript(urls.notifications)
      .then(() => {
        if (global.HelenNotifications && typeof global.HelenNotifications.ensure === 'function') {
          try {
            global.HelenNotifications.ensure();
          } catch (error) {
            console.warn('[HelenAlarmManager] No se pudo asegurar HelenNotifications:', error);
          }
          return global.HelenNotifications;
        }
        return null;
      })
      .catch((error) => {
        console.error('[HelenAlarmManager] No se pudo cargar notifications.js', error);
        return null;
      });
  };

  const ensureAlarmCore = () => {
    if (global.HelenScheduler && typeof global.HelenScheduler.ready === 'function') {
      return Promise.resolve(global.HelenScheduler);
    }
    return loadScript(urls.alarmCore)
      .then(() => (global.HelenScheduler && typeof global.HelenScheduler.ready === 'function'
        ? global.HelenScheduler
        : null))
      .catch((error) => {
        console.error('[HelenAlarmManager] No se pudo cargar alarm-core.js', error);
        return null;
      });
  };

  const ensureDependencies = () => ensureNotifications().then(() => ensureAlarmCore());

  const whenSchedulerReady = () => ensureAlarmCore().then((scheduler) => {
    if (scheduler && typeof scheduler.ready === 'function') {
      try {
        return Promise.resolve(scheduler.ready()).then(
          () => scheduler,
          () => scheduler,
        );
      } catch (error) {
        return Promise.resolve(scheduler);
      }
    }
    return scheduler || null;
  });

  const handleFiredEvent = (detail) => {
    if (global.HelenNotifications && typeof global.HelenNotifications.ensure === 'function') {
      try {
        global.HelenNotifications.ensure();
      } catch (error) {
        console.warn('[HelenAlarmManager] No se pudo refrescar el popup de alarmas:', error);
      }
    }
    if (detail && typeof detail === 'object') {
      try {
        global.dispatchEvent(new CustomEvent('helen:alarm:manager:observed', { detail }));
      } catch (error) {
        // Fallback a dispatchEvent opcional
      }
    }
  };

  const attachListeners = () => {
    if (state.listenersAttached) {
      return;
    }

    const windowHandler = (event) => {
      const payload = event && typeof event.detail !== 'undefined' ? event.detail : event;
      handleFiredEvent(payload);
    };

    if (global.HelenTimekeeperBus && typeof global.HelenTimekeeperBus.on === 'function') {
      try {
        state.busUnsubscribe = global.HelenTimekeeperBus.on('helen:timekeeper:fired', handleFiredEvent);
      } catch (error) {
        state.busUnsubscribe = null;
        console.warn('[HelenAlarmManager] No se pudo suscribir al bus global:', error);
      }
    }

    if (typeof global.addEventListener === 'function' && !state.windowListener) {
      global.addEventListener('helen:timekeeper:fired', windowHandler);
      state.windowListener = windowHandler;
    }

    state.listenersAttached = true;
  };

  const init = () => {
    if (state.readyPromise) {
      return state.readyPromise;
    }

    state.readyPromise = ensureDependencies()
      .then(() => whenSchedulerReady())
      .then((scheduler) => {
        attachListeners();
        return scheduler;
      })
      .catch((error) => {
        attachListeners();
        console.error('[HelenAlarmManager] Error durante la inicialización global:', error);
        return null;
      });

    return state.readyPromise;
  };

  const api = {
    version: 1,
    init,
    ready: () => init(),
    ensure: () => init(),
  };

  global.HelenAlarmManager = api;

  const bootstrap = () => {
    init().catch((error) => console.error('[HelenAlarmManager] Inicialización diferida fallida:', error));
  };

  if (documentRef) {
    if (documentRef.readyState === 'loading') {
      documentRef.addEventListener('DOMContentLoaded', bootstrap, { once: true });
    } else {
      bootstrap();
    }
  } else {
    bootstrap();
  }
})(typeof window !== 'undefined' ? window : globalThis);
