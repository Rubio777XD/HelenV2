const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Worker } = require('node:worker_threads');
const path = require('node:path');

const WORKER_PATH = path.resolve(__dirname, '../../helen/pages/jsFrontend/time-worker.js');

const waitForMessage = (worker, predicate, timeout = 1000) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => {
    worker.off('message', handler);
    reject(new Error('timeout waiting for worker message'));
  }, timeout);

  const handler = (data) => {
    if (predicate(data)) {
      clearTimeout(timer);
      worker.off('message', handler);
      resolve(data);
    }
  };

  worker.on('message', handler);
});

const createWorker = async () => {
  const worker = new Worker(WORKER_PATH);
  await waitForMessage(worker, (data) => data.type === 'ready', 500);
  return worker;
};

const syncWithWorker = (worker) => {
  worker.postMessage({ type: 'ping' });
  return waitForMessage(worker, (data) => data.type === 'pong');
};

test('worker fast-forward triggers timer firing within tolerance', async () => {
  const worker = await createWorker();
  try {
    const start = Date.now();
    worker.postMessage({
      type: 'set-items',
      items: [
        {
          id: 'timer-1',
          type: 'timer',
          state: 'running',
          targetEpochMs: start + 60000,
          remainingMs: 60000,
        },
      ],
    });

    await syncWithWorker(worker);
    const tickPromise = waitForMessage(worker, (data) => data.type === 'tick');
    worker.postMessage({ type: 'debug-fast-forward', milliseconds: 59000 });

    const tick = await tickPromise;
    const remaining = tick.payload.find((entry) => entry.id === 'timer-1').remainingMs;
    assert.ok(Math.abs(remaining - 1000) <= 120, `remaining ${remaining}ms should be close to 1000ms`);

    const firedPromise = waitForMessage(worker, (data) => data.type === 'fired');
    worker.postMessage({ type: 'debug-fast-forward', milliseconds: 2000 });
    const fired = await firedPromise;
    assert.equal(fired.payload[0].id, 'timer-1');
  } finally {
    worker.terminate();
  }
});

test('worker fires multiple alarms in order', async () => {
  const worker = await createWorker();
  try {
    const now = Date.now();
    worker.postMessage({
      type: 'set-items',
      items: [
        { id: 'a1', type: 'alarm', state: 'running', targetEpochMs: now + 1000 },
        { id: 'a2', type: 'alarm', state: 'running', targetEpochMs: now + 2000 },
        { id: 'a3', type: 'alarm', state: 'running', targetEpochMs: now + 3000 },
      ],
    });

    const firedIds = [];

    await syncWithWorker(worker);
    let firedPromise = waitForMessage(worker, (data) => data.type === 'fired');
    worker.postMessage({ type: 'debug-fast-forward', milliseconds: 1000 });
    let fired = await firedPromise;
    fired.payload.forEach((entry) => firedIds.push(entry.id));

    await syncWithWorker(worker);
    firedPromise = waitForMessage(worker, (data) => data.type === 'fired');
    worker.postMessage({ type: 'debug-fast-forward', milliseconds: 1000 });
    fired = await firedPromise;
    fired.payload.forEach((entry) => firedIds.push(entry.id));

    await syncWithWorker(worker);
    firedPromise = waitForMessage(worker, (data) => data.type === 'fired');
    worker.postMessage({ type: 'debug-fast-forward', milliseconds: 1000 });
    fired = await firedPromise;
    fired.payload.forEach((entry) => firedIds.push(entry.id));

    assert.deepEqual(firedIds, ['a1', 'a2', 'a3']);
  } finally {
    worker.terminate();
  }
});
