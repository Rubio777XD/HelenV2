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
    updateStatus(state, state.successMessage, true);
  };

  const completeCurrentStep = (state) => {
    const step = state.steps[state.current];
    if (!step) {
      return;
    }

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
    const state = {
      trainer,
      steps,
      practiceButton,
      statusEl,
      current: 0,
      active: false,
      successMessage: trainer.dataset.success || '¡Práctica completada! HELEN te entendió.',
      defaultMessage: statusEl ? (statusEl.dataset.default || statusEl.textContent.trim()) : '',
    };

    trainerStates.set(trainer, state);

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
      if (!state.active) {
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
