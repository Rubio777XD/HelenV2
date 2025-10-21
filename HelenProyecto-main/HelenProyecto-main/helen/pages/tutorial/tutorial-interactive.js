(function () {
  'use strict';

  const trainers = Array.from(document.querySelectorAll('.gesture-trainer'));
  const motionButtons = Array.from(document.querySelectorAll('[data-action="toggle-motion"]'));
  const gestureKeys = ['gesture', 'character', 'label', 'command', 'action', 'key'];
  const trainerStates = new Map();
  const body = document.body;

  let userMotionPreference = null;

  const collapseGesture = (value) => {
    if (value == null) {
      return '';
    }

    let candidate = String(value);
    try {
      candidate = candidate.normalize('NFKC');
    } catch (error) {
      console.warn('No se pudo normalizar la seña recibida:', error);
    }

    return candidate
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/\s+/g, '')
      .toLowerCase();
  };

  const resolveGestureFromPayload = (payload = {}) => {
    for (const key of gestureKeys) {
      if (Object.prototype.hasOwnProperty.call(payload, key)) {
        const collapsed = collapseGesture(payload[key]);
        if (collapsed) {
          return collapsed;
        }
      }
    }

    if (payload.state) {
      const collapsed = collapseGesture(payload.state);
      if (collapsed) {
        return collapsed;
      }
    }

    return '';
  };

  const getStepTitle = (step) => {
    const titleEl = step.node.querySelector('.gesture-step__title');
    return titleEl ? titleEl.textContent.trim() : 'la seña';
  };

  const parseFeedbackMessages = (trainer) => {
    const map = {};
    const entries = Object.entries(trainer.dataset || {});
    for (const [key, value] of entries) {
      if (typeof value !== 'string') continue;
      if (!key || !key.startsWith('feedback')) continue;
      const stageKey = collapseGesture(key.replace(/^feedback/, ''));
      if (stageKey) {
        map[stageKey] = value.trim();
      }
    }
    return map;
  };

  const setModuleStage = (state, stage, options = {}) => {
    if (!state.modulePreview) {
      return;
    }

    const preview = state.modulePreview;
    const statusEl = state.moduleStatus;
    const stageKey = collapseGesture(stage) || 'idle';
    const transient = Boolean(options.transient);
    const duration = typeof options.duration === 'number' ? options.duration : 2000;

    if (!transient && state.previewTimer) {
      window.clearTimeout(state.previewTimer);
      state.previewTimer = null;
    }

    preview.classList.remove('is-animate');
    preview.classList.remove('is-primed', 'is-triggered', 'is-complete', 'is-reset');

    let message = state.feedbackMessages[stageKey];
    if (!message) {
      if (stageKey === 'start') {
        message = state.feedbackMessages.start || 'HELEN activada. Sigue las instrucciones en pantalla.';
      } else if (stageKey === state.moduleKey) {
        const label = state.moduleLabel || state.moduleKey;
        message = state.feedbackMessages[state.moduleKey] || `Seña ${label} detectada.`;
      } else if (stageKey === 'inicio') {
        message = state.feedbackMessages.inicio || 'Regresando al inicio.';
      } else if (stageKey === 'complete') {
        message = state.feedbackMessages.complete || state.successMessage;
      } else {
        message = state.moduleDefaultStatus || state.defaultMessage || '';
      }
    }

    switch (stageKey) {
      case 'start':
        preview.classList.add('is-primed');
        break;
      case state.moduleKey:
        preview.classList.add('is-triggered');
        break;
      case 'inicio':
        preview.classList.add('is-reset');
        break;
      case 'complete':
        preview.classList.add('is-complete');
        break;
      default:
        preview.classList.add('is-reset');
        break;
    }

    // Reactiva la animación
    void preview.offsetWidth;
    preview.classList.add('is-animate');
    state.moduleStage = stageKey;

    if (statusEl && typeof message === 'string') {
      statusEl.textContent = message;
    }

    if (transient) {
      if (state.previewTimer) {
        window.clearTimeout(state.previewTimer);
      }
      state.previewTimer = window.setTimeout(() => {
        state.previewTimer = null;
        setModuleStage(state, 'idle');
      }, Math.max(600, duration));
    }
  };

  const updateStatus = (state, message, positive = false) => {
    if (!state.statusEl) return;
    state.statusEl.textContent = message;
    state.statusEl.classList.toggle('is-positive', positive);
  };

  const highlightCurrentStep = (state) => {
    state.steps.forEach((step, index) => {
      step.node.classList.toggle('is-current', state.active && index === state.current);
    });
  };

  const updatePracticeButton = (state) => {
    if (!state.practiceButton) return;
    state.practiceButton.setAttribute('aria-pressed', state.active ? 'true' : 'false');
  };

  const resetSteps = (state) => {
    state.steps.forEach((step) => {
      step.node.classList.remove('is-completed', 'is-current');
      const status = step.node.querySelector('.gesture-step__status');
      if (status) {
        status.textContent = '';
      }
    });
  };

  const startPractice = (state) => {
    state.active = true;
    state.current = 0;
    state.trainer.classList.add('is-active');
    state.trainer.classList.remove('is-complete');
    resetSteps(state);
    highlightCurrentStep(state);
    updatePracticeButton(state);
    setModuleStage(state, 'start');

    const firstStep = state.steps[state.current];
    if (firstStep) {
      updateStatus(state, `Realiza “${getStepTitle(firstStep)}” frente a la cámara.`);
      firstStep.node.classList.add('is-current');
    } else {
      updateStatus(state, state.defaultMessage || 'Realiza las señas en orden.');
    }
  };

  const finishPractice = (state) => {
    state.active = false;
    state.trainer.classList.remove('is-active');
    state.trainer.classList.add('is-complete');
    updatePracticeButton(state);
    highlightCurrentStep(state);
    setModuleStage(state, 'complete');
    updateStatus(state, state.successMessage, true);
  };

  const completeCurrentStep = (state) => {
    const step = state.steps[state.current];
    if (!step) {
      return;
    }

    const stepId = step.node.dataset.stepId || '';
    setModuleStage(state, stepId);

    step.node.classList.remove('is-current');
    step.node.classList.add('is-completed');
    const status = step.node.querySelector('.gesture-step__status');
    if (status) {
      status.textContent = '✓';
    }

    const nextIndex = state.current + 1;
    if (nextIndex < state.steps.length) {
      state.current = nextIndex;
      const nextStep = state.steps[state.current];
      nextStep.node.classList.add('is-current');
      updateStatus(state, `¡Bien! Ahora realiza “${getStepTitle(nextStep)}”.`);
    } else {
      finishPractice(state);
    }
  };

  const attachTrainer = (trainer) => {
    const steps = Array.from(trainer.querySelectorAll('.gesture-step')).map((node, index) => {
      const matches = (node.dataset.match || '')
        .split('|')
        .map((token) => collapseGesture(token))
        .filter(Boolean);
      return { node, matches, index };
    });

    if (!steps.length) {
      return;
    }

    const practiceButton = trainer.querySelector('[data-action="toggle-practice"]');
    const statusEl = trainer.querySelector('.gesture-visual__status');
    const modulePreview = trainer.querySelector('[data-module-preview]');
    const moduleStatus = modulePreview ? modulePreview.querySelector('[data-module-status]') : null;
    const moduleDefaultStatus = moduleStatus ? moduleStatus.textContent.trim() : '';
    const moduleLabelEl = modulePreview ? modulePreview.querySelector('.module-preview__title') : null;
    const moduleLabel = moduleLabelEl ? moduleLabelEl.textContent.trim() : '';
    const moduleKey = collapseGesture(trainer.dataset.module || '');
    const feedbackMessages = parseFeedbackMessages(trainer);

    const state = {
      trainer,
      steps,
      practiceButton,
      statusEl,
      current: 0,
      active: false,
      successMessage: trainer.dataset.success || '¡Práctica completada! HELEN te entendió.',
      defaultMessage: statusEl ? (statusEl.dataset.default || statusEl.textContent.trim()) : '',
      modulePreview,
      moduleStatus,
      moduleDefaultStatus,
      moduleLabel,
      moduleKey,
      feedbackMessages,
      moduleStage: 'idle',
      previewTimer: null,
    };

    state.feedbackMessages.complete = state.successMessage;
    if (!state.feedbackMessages.idle && state.moduleDefaultStatus) {
      state.feedbackMessages.idle = state.moduleDefaultStatus;
    }

    trainerStates.set(trainer, state);
    setModuleStage(state, 'idle');

    if (practiceButton) {
      practiceButton.addEventListener('click', () => {
        startPractice(state);
      });
    }
  };

  const handleGesture = (data = {}) => {
    const collapsed = resolveGestureFromPayload(data);
    if (!collapsed) {
      return;
    }

    trainerStates.forEach((state) => {
      const isModuleGesture = state.moduleKey && collapsed === state.moduleKey;
      if (!state.active) {
        if (collapsed === 'start' || isModuleGesture || collapsed === 'inicio') {
          setModuleStage(state, collapsed, { transient: true });
        }
        return;
      }

      const step = state.steps[state.current];
      if (!step) {
        return;
      }

      if (step.matches.includes(collapsed)) {
        completeCurrentStep(state);
      }
    });
  };

  const setupMotionToggle = () => {
    if (!motionButtons.length) {
      return;
    }

    const prefersReduced = typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;

    const applyMotionState = (forced) => {
      const reduce = typeof forced === 'boolean'
        ? forced
        : (userMotionPreference !== null
          ? userMotionPreference
          : (prefersReduced ? prefersReduced.matches : false));

      body.classList.toggle('is-reduced-motion', reduce);
      motionButtons.forEach((button) => {
        button.setAttribute('aria-pressed', reduce ? 'true' : 'false');
        const stateLabel = button.querySelector('.motion-toggle__state');
        if (stateLabel) {
          stateLabel.textContent = reduce ? 'Activado' : 'Desactivado';
        }
      });
    };

    motionButtons.forEach((button) => {
      button.addEventListener('click', () => {
        userMotionPreference = !body.classList.contains('is-reduced-motion');
        applyMotionState(userMotionPreference);
      });
    });

    if (prefersReduced && typeof prefersReduced.addEventListener === 'function') {
      prefersReduced.addEventListener('change', (event) => {
        if (userMotionPreference === null) {
          applyMotionState(event.matches);
        }
      });
    }

    applyMotionState();
  };

  if (trainers.length) {
    trainers.forEach(attachTrainer);
  }

  setupMotionToggle();

  if (typeof window.socket === 'object' && typeof window.socket.on === 'function') {
    window.socket.on('message', handleGesture);
  } else {
    console.warn('[Helen] Tutorial interactivo: no hay conexión SSE para validar las señas.');
  }
})();
