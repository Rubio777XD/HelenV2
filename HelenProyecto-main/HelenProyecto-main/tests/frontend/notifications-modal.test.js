const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const SCHEDULER_PATH = path.resolve(__dirname, '../../helen/pages/jsFrontend/alarm-core.js');
const NOTIFICATIONS_PATH = path.resolve(__dirname, '../../helen/ui/notifications/notifications.js');
const PENDING_KEY = 'helen:timekeeper:pendingQueue:v1';

const createTimerControls = () => {
  const trackedTimeouts = new Set();
  const trackedIntervals = new Set();

  const originalSetTimeout = global.setTimeout;
  const originalClearTimeout = global.clearTimeout;
  const originalSetInterval = global.setInterval;
  const originalClearInterval = global.clearInterval;

  const wrapSetTimeout = function (fn, delay, ...args) {
    if (typeof fn !== 'function') {
      return originalSetTimeout(fn, delay, ...args);
    }
    let timerId;
    const wrapped = (...callbackArgs) => {
      trackedTimeouts.delete(timerId);
      fn(...callbackArgs);
    };
    timerId = originalSetTimeout(wrapped, delay, ...args);
    trackedTimeouts.add(timerId);
    return timerId;
  };

  const wrapClearTimeout = function (timerId) {
    trackedTimeouts.delete(timerId);
    return originalClearTimeout(timerId);
  };

  const wrapSetInterval = function (fn, delay, ...args) {
    if (typeof fn !== 'function') {
      return originalSetInterval(fn, delay, ...args);
    }
    const timerId = originalSetInterval(fn, delay, ...args);
    trackedIntervals.add(timerId);
    return timerId;
  };

  const wrapClearInterval = function (timerId) {
    trackedIntervals.delete(timerId);
    return originalClearInterval(timerId);
  };

  global.setTimeout = wrapSetTimeout;
  global.clearTimeout = wrapClearTimeout;
  global.setInterval = wrapSetInterval;
  global.clearInterval = wrapClearInterval;

  return {
    clearAll() {
      trackedTimeouts.forEach((timerId) => originalClearTimeout(timerId));
      trackedIntervals.forEach((timerId) => originalClearInterval(timerId));
      trackedTimeouts.clear();
      trackedIntervals.clear();
    },
    restore() {
      global.setTimeout = originalSetTimeout;
      global.clearTimeout = originalClearTimeout;
      global.setInterval = originalSetInterval;
      global.clearInterval = originalClearInterval;
    },
  };
};

class ClassList {
  constructor(element) {
    this.element = element;
    this._set = new Set();
  }
  _sync() {
    this.element._className = Array.from(this._set).join(' ');
  }
  add(...names) {
    names.filter(Boolean).forEach((name) => {
      if (!this._set.has(name)) {
        this._set.add(name);
        this._sync();
      }
    });
  }
  remove(...names) {
    names.filter(Boolean).forEach((name) => {
      if (this._set.delete(name)) {
        this._sync();
      }
    });
  }
  contains(name) {
    return this._set.has(name);
  }
  toggle(name, force) {
    if (force === undefined) {
      if (this._set.has(name)) {
        this._set.delete(name);
        this._sync();
        return false;
      }
      this._set.add(name);
      this._sync();
      return true;
    }
    if (force) {
      this.add(name);
      return true;
    }
    this.remove(name);
    return false;
  }
  setFromString(value) {
    this._set = new Set(String(value || '').split(/\s+/).filter(Boolean));
    this._sync();
  }
  toString() {
    return this.element._className;
  }
}

class MockElement {
  constructor(tagName, document) {
    this.tagName = tagName.toUpperCase();
    this._document = document;
    this.parentNode = null;
    this.children = [];
    this.attributes = new Map();
    this._datasetStore = {};
    const element = this;
    const toDataAttr = (prop) => {
      const name = String(prop);
      return `data-${name.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`)}`;
    };
    this.dataset = new Proxy(this._datasetStore, {
      set(target, prop, value) {
        const stringValue = String(value);
        target[prop] = stringValue;
        element.attributes.set(toDataAttr(prop), stringValue);
        return true;
      },
      get(target, prop) {
        return target[prop];
      },
      has(target, prop) {
        return prop in target;
      },
      deleteProperty(target, prop) {
        if (prop in target) {
          delete target[prop];
          element.attributes.delete(toDataAttr(prop));
        }
        return true;
      },
      ownKeys(target) {
        return Reflect.ownKeys(target);
      },
      getOwnPropertyDescriptor(target, prop) {
        if (!(prop in target)) {
          return undefined;
        }
        return {
          value: target[prop],
          writable: true,
          enumerable: true,
          configurable: true,
        };
      },
    });
    this.style = {};
    this.eventListeners = {};
    this.hidden = false;
    this._text = '';
    this._className = '';
    this.classList = new ClassList(this);
    this.id = '';
    Object.defineProperty(this, 'className', {
      get: () => this._className,
      set: (value) => this.classList.setFromString(value),
    });
  }
  get textContent() {
    return this._text;
  }
  set textContent(value) {
    this._text = String(value ?? '');
    this.children = [];
  }
  get innerHTML() {
    return this._text;
  }
  set innerHTML(value) {
    this._text = String(value ?? '');
    this.children = [];
  }
  appendChild(child) {
    if (!child) return child;
    if (child.parentNode) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  append(...nodes) {
    nodes.forEach((node) => this.appendChild(node));
  }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) {
      this.children.splice(index, 1);
      child.parentNode = null;
    }
    return child;
  }
  remove() {
    if (this.parentNode) {
      this.parentNode.removeChild(this);
    }
  }
  setAttribute(name, value) {
    const stringValue = String(value);
    this.attributes.set(name, stringValue);
    if (name === 'id') {
      this.id = stringValue;
    }
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
      this._datasetStore[key] = stringValue;
    }
  }
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  addEventListener(type, listener, options = {}) {
    if (!this.eventListeners[type]) {
      this.eventListeners[type] = [];
    }
    this.eventListeners[type].push({ listener, once: Boolean(options.once) });
  }
  removeEventListener(type, listener) {
    const list = this.eventListeners[type];
    if (!list) return;
    this.eventListeners[type] = list.filter((entry) => entry.listener !== listener);
  }
  dispatchEvent(event) {
    const list = this.eventListeners[event.type] || [];
    list.slice().forEach((entry) => {
      entry.listener.call(this, event);
      if (entry.once) {
        this.removeEventListener(event.type, entry.listener);
      }
    });
    return true;
  }
  click() {
    this.dispatchEvent({ type: 'click', target: this, currentTarget: this, preventDefault() {}, stopPropagation() {} });
  }
  focus() {
    if (this._document) {
      this._document.activeElement = this;
    }
  }
  contains(node) {
    if (node === this) return true;
    return this.children.some((child) => child.contains && child.contains(node));
  }
  _matches(selector) {
    if (!selector) return false;
    if (selector.startsWith('.')) {
      return this.classList.contains(selector.slice(1));
    }
    if (selector.startsWith('#')) {
      return this.id === selector.slice(1);
    }
    if (selector.startsWith('[') && selector.endsWith(']')) {
      const content = selector.slice(1, -1);
      const [attr, rawValue] = content.split('=');
      const attrName = attr.trim();
      if (rawValue) {
        const expected = rawValue.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        return this.getAttribute(attrName) === expected;
      }
      return this.getAttribute(attrName) != null;
    }
    return this.tagName.toLowerCase() === selector.toLowerCase();
  }
  querySelector(selector) {
    if (this._matches(selector)) {
      return this;
    }
    for (const child of this.children) {
      if (child instanceof MockElement) {
        const found = child.querySelector(selector);
        if (found) return found;
      }
    }
    return null;
  }
  querySelectorAll(selector, acc = []) {
    if (this._matches(selector)) {
      acc.push(this);
    }
    for (const child of this.children) {
      if (child instanceof MockElement) {
        child.querySelectorAll(selector, acc);
      }
    }
    return acc;
  }
}

class MockLocalStorage {
  constructor() {
    this._map = new Map();
  }
  getItem(key) {
    const value = this._map.get(String(key));
    return value === undefined ? null : value;
  }
  setItem(key, value) {
    this._map.set(String(key), String(value));
  }
  removeItem(key) {
    this._map.delete(String(key));
  }
  clear() {
    this._map.clear();
  }
}

class MockDocument {
  constructor() {
    this.readyState = 'complete';
    this._listeners = {};
    this.activeElement = null;
    this.documentElement = new MockElement('html', this);
    this.body = new MockElement('body', this);
    this.documentElement.appendChild(this.body);
    this.body.focus();
    this.currentScript = { src: 'https://example.com/pages/jsFrontend/alarm-core.js' };
  }
  createElement(tag) {
    return new MockElement(tag, this);
  }
  querySelector(selector) {
    return this.documentElement.querySelector(selector);
  }
  querySelectorAll(selector) {
    return this.documentElement.querySelectorAll(selector, []);
  }
  getElementById(id) {
    return this.documentElement.querySelector(`#${id}`);
  }
  addEventListener(type, listener) {
    if (!this._listeners[type]) {
      this._listeners[type] = [];
    }
    this._listeners[type].push(listener);
  }
  dispatchEvent(event) {
    const list = this._listeners[event.type] || [];
    list.slice().forEach((listener) => listener.call(this, event));
    return true;
  }
  contains(node) {
    return this.documentElement.contains(node);
  }
}

const createMockDom = () => {
  const document = new MockDocument();
  const localStorage = new MockLocalStorage();
  const windowListeners = {};

  const window = {
    document,
    location: { href: 'https://example.com/index.html' },
    navigator: {},
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    requestAnimationFrame: (cb) => window.setTimeout(cb, 0),
    cancelAnimationFrame: (id) => window.clearTimeout(id),
    addEventListener(type, listener) {
      if (!windowListeners[type]) windowListeners[type] = [];
      windowListeners[type].push(listener);
    },
    removeEventListener(type, listener) {
      const list = windowListeners[type];
      if (!list) return;
      windowListeners[type] = list.filter((fn) => fn !== listener);
    },
    dispatchEvent(event) {
      const list = windowListeners[event.type] || [];
      list.slice().forEach((listener) => listener.call(window, event));
      return true;
    },
    localStorage,
    performance: { now: () => Date.now() },
    crypto: { randomUUID: () => `mock-${Math.random().toString(16).slice(2)}` },
  };

  document.defaultView = window;
  document.body.focus();
  return { window, document, localStorage };
};

const setupEnvironment = async () => {
  delete require.cache[SCHEDULER_PATH];
  delete require.cache[NOTIFICATIONS_PATH];

  const timerControls = createTimerControls();
  const { window, document, localStorage } = createMockDom();
  const original = {
    window: global.window,
    document: global.document,
    navigator: global.navigator,
    localStorage: global.localStorage,
    requestAnimationFrame: global.requestAnimationFrame,
    cancelAnimationFrame: global.cancelAnimationFrame,
    fetch: global.fetch,
    Audio: global.Audio,
    SharedWorker: global.SharedWorker,
    performance: global.performance,
  };

  global.window = window;
  global.document = document;
  global.navigator = window.navigator;
  global.localStorage = localStorage;
  global.requestAnimationFrame = window.requestAnimationFrame;
  global.cancelAnimationFrame = window.cancelAnimationFrame;
  global.performance = window.performance;

  class FakeAudio {
    constructor(src = '') {
      this.src = src;
      this.loop = false;
      this.preload = 'auto';
      this.paused = true;
      this.currentTime = 0;
    }
    play() {
      this.paused = false;
      return Promise.resolve();
    }
    pause() {
      this.paused = true;
    }
    cloneNode() {
      const clone = new FakeAudio(this.src);
      clone.loop = this.loop;
      return clone;
    }
  }

  const ports = [];
  class MockSharedWorker {
    constructor() {
      const port = {
        onmessage: null,
        start() {},
        postMessage() {},
      };
      ports.push(port);
      setTimeout(() => {
        if (port.onmessage) {
          port.onmessage({ data: { type: 'ready' } });
        }
      }, 0);
      this.port = port;
    }
  }

  global.fetch = async () => { throw new Error('offline'); };
  global.Audio = FakeAudio;
  window.Audio = FakeAudio;
  global.SharedWorker = MockSharedWorker;
  window.SharedWorker = MockSharedWorker;
  window.Worker = undefined;
  window.AudioContext = undefined;
  window.webkitAudioContext = undefined;

  require(NOTIFICATIONS_PATH);
  require(SCHEDULER_PATH);

  const scheduler = window.HelenScheduler;
  assert.ok(scheduler, 'HelenScheduler should be available');
  await scheduler.ready();

  const cleanup = async () => {
    timerControls.clearAll();
    timerControls.restore();
    delete require.cache[SCHEDULER_PATH];
    delete require.cache[NOTIFICATIONS_PATH];
    global.window = original.window;
    global.document = original.document;
    global.navigator = original.navigator;
    global.localStorage = original.localStorage;
    global.requestAnimationFrame = original.requestAnimationFrame;
    global.cancelAnimationFrame = original.cancelAnimationFrame;
    global.fetch = original.fetch;
    global.Audio = original.Audio;
    global.SharedWorker = original.SharedWorker;
    global.performance = original.performance;
  };

  return { window, scheduler, ports, cleanup };
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 40));

test('due timer displays modal with accessible controls', async () => {
  const { window, scheduler, ports, cleanup } = await setupEnvironment();
  try {
    scheduler.createTimer({ id: 'timer-1', label: 'Focus 25', durationMs: 60000 });
    await flush();

    const port = ports[0];
    port.onmessage({ data: { type: 'fired', payload: [{ id: 'timer-1' }] } });

    await flush();

    const modal = window.document.querySelector('.helen-global-modal');
    assert.ok(modal, 'modal should exist');
    assert.ok(modal.classList.contains('is-visible'), 'modal should be visible');
    assert.equal(window.document.body.classList.contains('helen-global-modal-open'), true);
    assert.equal(modal.getAttribute('aria-live'), 'assertive');

    const title = modal.querySelector('.helen-global-modal__title');
    assert.equal(title.textContent, 'Temporizador finalizado');

    const label = modal.querySelector('.helen-global-modal__label');
    assert.equal(label.textContent, 'Focus 25');

    const actionBtn = modal.querySelector('.helen-global-modal__primary');
    assert.ok(actionBtn, 'primary button should exist');
    assert.equal(actionBtn.textContent, 'Detener');
    assert.equal(window.document.activeElement, actionBtn, 'primary button receives focus');

    const status = modal.querySelector('.helen-global-modal__status');
    assert.ok(status, 'status element should exist');
    assert.equal(status.hidden, true, 'status hidden by default');

    actionBtn.click();
    await flush();

    assert.equal(modal.classList.contains('is-visible'), false);
    assert.equal(window.document.body.classList.contains('helen-global-modal-open'), false);
  } finally {
    await cleanup();
  }
});

test('notifications queue next item after dismiss and persist state', async () => {
  const { window, scheduler, ports, cleanup } = await setupEnvironment();
  try {
    scheduler.createTimer({ id: 'timer-1', label: 'Tarea Uno', durationMs: 60000 });
    scheduler.createTimer({ id: 'timer-2', label: 'Tarea Dos', durationMs: 60000 });
    await flush();

    const port = ports[0];
    port.onmessage({ data: { type: 'fired', payload: [{ id: 'timer-1' }] } });
    port.onmessage({ data: { type: 'fired', payload: [{ id: 'timer-2' }] } });

    await flush();

    const modal = window.document.querySelector('.helen-global-modal');
    const label = modal.querySelector('.helen-global-modal__label');
    assert.equal(label.textContent, 'Tarea Uno');

    const actionBtn = modal.querySelector('.helen-global-modal__primary');
    assert.ok(actionBtn, 'primary button available');

    const badge = modal.querySelector('.helen-global-modal__badge');
    assert.equal(badge.hidden, false, 'badge visible while there are pending notifications');
    assert.equal(badge.textContent, '+1');

    actionBtn.click();
    await flush();

    assert.equal(label.textContent, 'Tarea Dos');
    assert.equal(badge.hidden, true, 'badge hidden when queue is empty');

    actionBtn.click();
    await flush();

    assert.equal(modal.classList.contains('is-visible'), false);
    assert.equal(window.document.body.classList.contains('helen-global-modal-open'), false);

    const stored = window.localStorage.getItem(PENDING_KEY);
    assert.equal(stored, '[]', 'pending queue cleared after dismissing all notifications');
  } finally {
    await cleanup();
  }
});

test('global timekeeper bus emits fired events', async () => {
  const { window, scheduler, ports, cleanup } = await setupEnvironment();
  const domEvents = [];
  const domLatencies = [];
  const busEvents = [];
  let removeBusListener = null;
  let fireStart = 0;
  const handleDomEvent = (event) => {
    domEvents.push(event.detail);
    if (fireStart) {
      domLatencies.push(window.performance.now() - fireStart);
    }
  };
  try {
    window.addEventListener('helen:timekeeper:fired', handleDomEvent);
    if (window.HelenTimekeeperBus && typeof window.HelenTimekeeperBus.on === 'function') {
      removeBusListener = window.HelenTimekeeperBus.on('helen:timekeeper:fired', (detail) => {
        busEvents.push(detail);
      });
    }

    scheduler.createTimer({ id: 'timer-bus', label: 'Bus Timer', durationMs: 60000 });
    await flush();

    const port = ports[0];
    fireStart = window.performance.now();
    port.onmessage({ data: { type: 'fired', payload: [{ id: 'timer-bus' }] } });
    await flush();

    assert.equal(domEvents.length, 1, 'DOM event should fire once');
    assert.equal(domEvents[0].id, 'timer-bus');
    assert.equal(domEvents[0].tone, 'timer');
    assert.equal(domEvents[0].type, 'timer');
    if (domLatencies.length) {
      assert.ok(domLatencies[0] < 50, `latencia DOM esperada <50ms, recibida ${domLatencies[0]}`);
    }

    if (busEvents.length) {
      assert.equal(busEvents.length, 1, 'bus listener should receive event once');
      assert.equal(busEvents[0].eventId, domEvents[0].eventId);
    }
  } finally {
    if (removeBusListener) {
      try { removeBusListener(); } catch (error) {}
    }
    window.removeEventListener('helen:timekeeper:fired', handleDomEvent);
    await cleanup();
  }
});
