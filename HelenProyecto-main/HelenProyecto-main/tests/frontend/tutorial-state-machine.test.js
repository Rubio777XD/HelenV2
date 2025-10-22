const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  TutorialStateMachine,
  modules,
} = require('../../helen/pages/tutorial/tutorial-interactive.js');

test('tutorial state machine completes weather module with confirmations', () => {
  const events = { started: [], passed: [] };
  const machine = new TutorialStateMachine({
    steps: modules.weather.steps,
    confirmationsRequired: 2,
    stepTimeoutMs: 5000,
    onStepStarted: ({ step, index }) => events.started.push({ id: step.id, index }),
    onStepPassed: ({ step, index }) => events.passed.push({ id: step.id, index }),
  });

  assert.equal(machine.start(), true, 'machine should start');
  assert.equal(events.started[0].id, 'activate');

  // Step 1 - Start gesture twice
  assert.equal(machine.handleGesture({ gesture: 'Start' }), true);
  assert.equal(machine.handleGesture({ gesture: 'Start' }), true);
  assert.equal(events.passed.length, 1);
  assert.equal(events.passed[0].id, 'activate');

  assert.equal(machine.advance(), true);
  assert.equal(events.started[1].id, 'ask-weather');

  // Step 2 - Clima gesture twice
  machine.handleGesture({ gesture: 'Clima' });
  machine.handleGesture({ gesture: 'Clima' });
  assert.equal(events.passed.length, 2);
  assert.equal(events.passed[1].id, 'ask-weather');

  assert.equal(machine.advance(), true);
  assert.equal(events.started[2].id, 'return-home');

  // Step 3 - Inicio gesture twice
  machine.handleGesture({ gesture: 'Inicio' });
  machine.handleGesture({ gesture: 'Inicio' });
  assert.equal(events.passed.length, 3);
  assert.equal(events.passed[2].id, 'return-home');
});

test('timeout forces retry before completing step', async () => {
  let progressSnapshots = [];
  let resolveTimeout;
  const timeoutPromise = new Promise((resolve) => { resolveTimeout = resolve; });

  const machine = new TutorialStateMachine({
    steps: modules.weather.steps,
    confirmationsRequired: 2,
    stepTimeoutMs: 80,
    onStepTimeout: () => resolveTimeout(),
    onProgress: ({ confirmations, required }) => {
      progressSnapshots.push({ confirmations, required });
    },
  });

  machine.start();
  await timeoutPromise;

  machine.handleGesture({ gesture: 'Start' });
  machine.handleGesture({ gesture: 'Start' });
  assert.equal(progressSnapshots.at(-1).confirmations, 2);
});

test('incorrect gesture resets confirmation count', () => {
  const confirmations = [];
  const machine = new TutorialStateMachine({
    steps: modules.weather.steps,
    confirmationsRequired: 2,
    stepTimeoutMs: 5000,
    onProgress: ({ confirmations: current, required }) => {
      confirmations.push({ current, required });
    },
  });

  machine.start();
  machine.handleGesture({ gesture: 'Clima' }); // wrong command first
  assert.equal(confirmations.length, 0, 'no progress on mismatched gesture');

  machine.handleGesture({ gesture: 'Start' });
  machine.handleGesture({ gesture: 'Clima' }); // resets progress
  machine.handleGesture({ gesture: 'Start' });
  machine.handleGesture({ gesture: 'Start' });

  const lastProgress = confirmations.at(-1);
  assert.equal(lastProgress.current, 2);
  assert.equal(lastProgress.required, 2);
});
