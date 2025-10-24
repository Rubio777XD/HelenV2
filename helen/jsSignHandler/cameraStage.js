(() => {
  const STAGE_SELECTOR = '[data-camera-stage]';
  const VIDEO_SELECTOR = '[data-camera-video]';
  const OVERLAY_SELECTOR = '[data-camera-overlay]';
  const RESOLUTION_SELECTOR = '[data-camera-resolution]';
  const META_SELECTOR = '[data-camera-meta]';
  const DEFAULT_WIDTH = 960;
  const DEFAULT_HEIGHT = 540;
  const POLL_INTERVAL_MS = 15000;

  let pollTimer = null;
  let inflight = false;

  const numberOr = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  };

  const locateStage = () => {
    if (typeof document === 'undefined') {
      return null;
    }
    return document.querySelector(STAGE_SELECTOR);
  };

  const collectElements = () => {
    const stage = locateStage();
    if (!stage) {
      return null;
    }
    return {
      stage,
      video: stage.querySelector(VIDEO_SELECTOR),
      overlay: stage.querySelector(OVERLAY_SELECTOR),
      resolutionLabel: stage.querySelector(RESOLUTION_SELECTOR),
      meta: stage.querySelector(META_SELECTOR),
    };
  };

  const formatResolution = (width, height, fps) => {
    const safeWidth = Math.round(width);
    const safeHeight = Math.round(height);
    const roundedFps = fps > 0 ? (fps >= 15 ? fps.toFixed(1) : fps.toFixed(2)) : '0';
    return `${safeWidth}×${safeHeight} @ ${roundedFps} fps`;
  };

  const applyDimensions = (width, height, fps) => {
    const elements = collectElements();
    if (!elements) {
      return;
    }

    const targetWidth = Math.max(1, Math.round(width || DEFAULT_WIDTH));
    const targetHeight = Math.max(1, Math.round(height || DEFAULT_HEIGHT));
    const targetFps = Math.max(0, Number(fps) || 0);
    const ratio = `${targetWidth} / ${targetHeight}`;

    elements.stage.style.setProperty('--camera-stage-aspect', ratio);
    elements.stage.dataset.cameraWidth = String(targetWidth);
    elements.stage.dataset.cameraHeight = String(targetHeight);
    elements.stage.dataset.cameraFps = String(targetFps);

    if (elements.video) {
      try {
        elements.video.width = targetWidth;
        elements.video.height = targetHeight;
      } catch (error) {
        console.warn('[Helen][camera] No se pudo asignar tamaño al elemento <video>:', error);
      }
      elements.video.style.setProperty('--camera-stage-aspect', ratio);
      elements.video.style.objectFit = 'contain';
    }

    if (elements.overlay) {
      try {
        elements.overlay.width = targetWidth;
        elements.overlay.height = targetHeight;
      } catch (error) {
        console.warn('[Helen][camera] No se pudo ajustar el canvas de overlay:', error);
      }
    }

    if (elements.resolutionLabel) {
      elements.resolutionLabel.textContent = formatResolution(targetWidth, targetHeight, targetFps);
    }

    if (typeof window !== 'undefined') {
      if (window.HelenRaspberryFit && typeof window.HelenRaspberryFit.refresh === 'function') {
        window.HelenRaspberryFit.refresh();
      }
      window.requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
      });
    }
  };

  const resolveDimensions = (snapshot) => {
    if (!snapshot || typeof snapshot !== 'object') {
      return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT, fps: 0 };
    }

    const pick = (...candidates) => {
      for (const candidate of candidates) {
        if (Array.isArray(candidate) && candidate.length >= 2) {
          const [w, h] = candidate;
          const width = numberOr(w, 0);
          const height = numberOr(h, 0);
          if (width > 0 && height > 0) {
            return { width, height };
          }
        }
      }
      return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
    };

    const { width, height } = pick(
      snapshot.presentation_resolution,
      snapshot.processing_resolution,
      snapshot.capture_resolution,
    );

    const fps = numberOr(
      snapshot.presentation_fps || snapshot.processing_fps || snapshot.capture_fps,
      0,
    );

    return { width, height, fps };
  };

  const applySnapshot = (snapshot) => {
    const { width, height, fps } = resolveDimensions(snapshot || {});
    applyDimensions(width, height, fps);
  };

  const schedulePoll = () => {
    if (pollTimer) {
      window.clearTimeout(pollTimer);
    }
    if (typeof window === 'undefined') {
      return;
    }
    pollTimer = window.setTimeout(() => {
      void fetchAndApply();
    }, POLL_INTERVAL_MS);
  };

  const fetchAndApply = async () => {
    if (inflight || typeof fetch !== 'function') {
      return;
    }
    const elements = collectElements();
    if (!elements) {
      return;
    }

    inflight = true;
    try {
      const response = await fetch('/health', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      if (payload && typeof payload === 'object') {
        applySnapshot(payload);
      }
    } catch (error) {
      console.warn('[Helen][camera] No se pudo obtener /health:', error);
    } finally {
      inflight = false;
      schedulePoll();
    }
  };

  const bootstrap = () => {
    applyDimensions(DEFAULT_WIDTH, DEFAULT_HEIGHT, 0);
    void fetchAndApply();
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', bootstrap, { once: true });
    } else {
      bootstrap();
    }
  }

  if (typeof window !== 'undefined') {
    window.HelenCameraStage = {
      refresh: () => void fetchAndApply(),
      applySnapshot,
      applyDimensions,
    };

    window.addEventListener('helen:display-mode', (event) => {
      const mode = event && event.detail ? event.detail.mode : undefined;
      if (mode === 'raspberry') {
        void fetchAndApply();
      }
    });
  }
})();
