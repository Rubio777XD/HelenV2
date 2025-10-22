(() => {
  'use strict';

  const modules = {
    weather: {
      title: 'Clima',
      subtitle: 'Secuencia guiada para consultar el clima.',
      icon: 'bi-cloud-sun',
      footnote: 'Secuencia oficial: H → C → H + I.',
      steps: [
        {
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Esta seña despierta al asistente antes de cualquier comando.',
          complete: 'Seña H reconocida. HELEN está lista para ayudarte.'
        },
        {
          text: 'Ahora haz la seña de la letra C para consultar el clima.',
          hint: 'Mantén la mano centrada frente a la cámara para una detección clara.',
          complete: '¡Perfecto! HELEN obtiene el clima actual.'
        },
        {
          text: 'HELEN te mostrará el clima actual. Para volver, haz H + I.',
          hint: 'Recuerda cerrar la sesión con la seña de Inicio.',
          complete: 'Tutorial de clima completado.'
        }
      ]
    },
    clock: {
      title: 'Hora',
      subtitle: 'Aprende a pedir la hora con confirmaciones guiadas.',
      icon: 'bi-clock-history',
      footnote: 'Secuencia oficial: H → R → H + I.',
      steps: [
        {
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Asegúrate de que HELEN esté atenta antes de continuar.',
          complete: 'Activación lista. HELEN está escuchando.'
        },
        {
          text: 'Luego haz la seña de la letra R para consultar la hora.',
          hint: 'Forma la letra R con claridad y sostenla un instante.',
          complete: 'Hora solicitada correctamente.'
        },
        {
          text: 'HELEN te dirá la hora actual. Para terminar, realiza H + I.',
          hint: 'Finaliza con Inicio para volver al panel principal.',
          complete: 'Tutorial de hora completado.'
        }
      ]
    },
    devices: {
      title: 'Dispositivos',
      subtitle: 'Simula cómo encender o apagar dispositivos conectados.',
      icon: 'bi-lightbulb',
      footnote: 'Secuencia oficial: H → Seña del dispositivo → H + I.',
      steps: [
        {
          text: 'Haz la seña de la letra H para activar a HELEN.',
          hint: 'Activa siempre a HELEN antes de controlar dispositivos.',
          complete: 'HELEN está lista para recibir la seña del dispositivo.'
        },
        {
          text: 'Realiza la seña de activación del dispositivo que necesites.',
          hint: 'Imagina la seña asignada al dispositivo para practicarla.',
          complete: 'Comando del dispositivo confirmado.'
        },
        {
          text: 'HELEN mostrará el estado ON/OFF. Para salir, haz H + I.',
          hint: 'Cierra con Inicio para regresar al menú principal.',
          complete: 'Tutorial de dispositivos completado.'
        }
      ]
    }
  };

  const root = document.querySelector('[data-tutorial]');
  if (!root) {
    return;
  }

  const menuView = root.querySelector('[data-view="menu"]');
  const flowView = root.querySelector('[data-view="flow"]');
  const triggers = Array.from(root.querySelectorAll('[data-module-trigger]'));
  const flowIcon = flowView.querySelector('[data-flow-icon]');
  const flowTitle = flowView.querySelector('[data-flow-title]');
  const flowSubtitle = flowView.querySelector('[data-flow-subtitle]');
  const footnote = flowView.querySelector('[data-flow-footnote]');
  const stepVisual = flowView.querySelector('[data-step-visual]');
  const stepNumber = flowView.querySelector('[data-step-number]');
  const stepDigit = flowView.querySelector('[data-step-digit]');
  const stepText = flowView.querySelector('[data-step-text]');
  const stepStatus = flowView.querySelector('[data-step-status]');
  const dotsContainer = flowView.querySelector('[data-step-dots]');
  const flowControls = flowView.querySelector('[data-flow-controls]');
  const nextButton = flowView.querySelector('[data-action="next-step"]');
  const closeButton = flowView.querySelector('[data-action="close-flow"]');
  const returnButton = flowView.querySelector('[data-action="return-menu"]');

  const reducedMotionQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;

  let activeKey = null;
  let activeStep = 0;
  let dots = [];
  let stepTimer = null;

  const getStepDelay = (stepData) => {
    if (stepData && typeof stepData.delay === 'number') {
      return Math.max(400, stepData.delay);
    }
    if (reducedMotionQuery && reducedMotionQuery.matches) {
      return 900;
    }
    return 2000;
  };

  const clearTimer = () => {
    if (stepTimer) {
      window.clearTimeout(stepTimer);
      stepTimer = null;
    }
  };

  const toggleFootnote = (text) => {
    if (!footnote) {
      return;
    }
    if (text) {
      footnote.textContent = text;
      footnote.classList.remove('is-hidden');
    } else {
      footnote.textContent = '';
      footnote.classList.add('is-hidden');
    }
  };

  const setView = (viewName) => {
    const isFlow = viewName === 'flow';
    menuView.classList.toggle('is-active', !isFlow);
    menuView.setAttribute('aria-hidden', isFlow ? 'true' : 'false');
    flowView.classList.toggle('is-active', isFlow);
    flowView.setAttribute('aria-hidden', isFlow ? 'false' : 'true');
    if (isFlow) {
      try {
        flowView.focus({ preventScroll: true });
      } catch (error) {
        flowView.focus();
      }
    }
  };

  const buildDots = (count) => {
    dotsContainer.innerHTML = '';
    dots = [];
    for (let index = 0; index < count; index += 1) {
      const dot = document.createElement('li');
      dot.className = 'step-dot';
      dot.dataset.stepIndex = String(index);
      dotsContainer.appendChild(dot);
      dots.push(dot);
    }
  };

  const updateDots = () => {
    dots.forEach((dot, index) => {
      dot.classList.toggle('is-active', index <= activeStep);
      dot.classList.toggle('is-current', index === activeStep);
    });
  };

  const animateCopy = () => {
    [stepText, stepStatus].forEach((node) => {
      if (!node) return;
      node.classList.remove('is-appear');
      void node.offsetWidth; // reflow para reiniciar animación
      node.classList.add('is-appear');
    });
  };

  const runStep = () => {
    const module = modules[activeKey];
    if (!module) {
      return;
    }

    const steps = module.steps;
    if (!Array.isArray(steps) || steps.length === 0) {
      return;
    }

    const total = steps.length;
    activeStep = Math.min(Math.max(activeStep, 0), total - 1);
    const stepData = steps[activeStep];
    const isLast = activeStep === total - 1;

    stepVisual.classList.remove('is-complete');
    stepNumber.textContent = `Paso ${activeStep + 1} de ${total}`;
    stepDigit.textContent = String(activeStep + 1);
    stepText.textContent = stepData.text;
    stepStatus.textContent = stepData.hint || 'Cuando HELEN confirme la seña verás esta marca verde.';
    animateCopy();

    flowControls.classList.toggle('is-hidden', isLast);
    nextButton.disabled = true;
    nextButton.classList.remove('is-visible');
    nextButton.setAttribute('aria-hidden', isLast ? 'true' : 'false');

    returnButton.classList.remove('is-visible');
    returnButton.setAttribute('aria-hidden', 'true');

    updateDots();
    clearTimer();

    const delay = getStepDelay(stepData);
    stepTimer = window.setTimeout(() => {
      stepVisual.classList.add('is-complete');
      stepStatus.textContent = stepData.complete || '¡Paso completado!';

      if (isLast) {
        returnButton.classList.add('is-visible');
        returnButton.setAttribute('aria-hidden', 'false');
        try {
          returnButton.focus({ preventScroll: true });
        } catch (error) {
          returnButton.focus();
        }
      } else {
        nextButton.disabled = false;
        nextButton.classList.add('is-visible');
        try {
          nextButton.focus({ preventScroll: true });
        } catch (error) {
          nextButton.focus();
        }
      }
    }, Math.max(600, delay));
  };

  const startModule = (key) => {
    const module = modules[key];
    if (!module) {
      return;
    }

    activeKey = key;
    activeStep = 0;

    flowIcon.className = `flow-heading__icon bi ${module.icon}`;
    flowTitle.textContent = module.title;
    flowSubtitle.textContent = module.subtitle;
    toggleFootnote(module.footnote);

    buildDots(module.steps.length);
    setView('flow');
    runStep();
  };

  const exitFlow = () => {
    clearTimer();
    activeKey = null;
    activeStep = 0;
    flowControls.classList.remove('is-hidden');
    nextButton.disabled = true;
    nextButton.classList.remove('is-visible');
    nextButton.setAttribute('aria-hidden', 'false');
    returnButton.classList.remove('is-visible');
    returnButton.setAttribute('aria-hidden', 'true');
    toggleFootnote('');
    setView('menu');
  };

  triggers.forEach((trigger) => {
    trigger.addEventListener('click', () => {
      const key = trigger.dataset.moduleTrigger;
      startModule(key);
    });
  });

  nextButton.addEventListener('click', () => {
    const module = modules[activeKey];
    if (!module) {
      return;
    }

    if (activeStep < module.steps.length - 1) {
      activeStep += 1;
      runStep();
    }
  });

  const leaveFlow = () => {
    exitFlow();
  };

  closeButton.addEventListener('click', leaveFlow);
  returnButton.addEventListener('click', leaveFlow);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && flowView.classList.contains('is-active')) {
      exitFlow();
    }
  });

  if (reducedMotionQuery && typeof reducedMotionQuery.addEventListener === 'function') {
    reducedMotionQuery.addEventListener('change', () => {
      if (!flowView.classList.contains('is-active')) {
        return;
      }
      clearTimer();
      runStep();
    });
  }

  setView('menu');
})();
