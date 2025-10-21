(function () {
  'use strict';

  const scheduler = window.HelenScheduler;
  if (!scheduler) {
    console.warn('[Alarmas] HelenScheduler no está disponible.');
    return;
  }

  const alarmList = document.querySelector('.alarm-list, .alarms-list');
  const loadingElement = document.querySelector('.alarm-loading');
  const noAlarmsMessage = document.querySelector('.no-alarms-message');
  const newAlarmBtn = document.getElementById('new-alarm-btn');
  const modal = document.getElementById('newAlarmModal');
  const modalClose = modal ? modal.querySelector('.close-modal') : null;
  const modalCancel = modal ? modal.querySelector('.cancel-btn') : null;
  const modalAction = modal ? modal.querySelector('.create-btn') : null;
  const modalTitle = modal ? modal.querySelector('.modal-header h2') : null;
  const timeInput = document.getElementById('alarm-time-input');
  const labelInput = document.getElementById('alarm-label-input');
  const daysContainer = document.getElementById('modal-days');
  const deleteModal = document.getElementById('deleteConfirmModal');
  const deleteCancel = deleteModal ? deleteModal.querySelector('[data-cancel-delete]') : null;
  const deleteClose = deleteModal ? deleteModal.querySelector('[data-close-delete]') : null;
  const deleteConfirm = document.getElementById('confirmDeleteBtn');
  const toast = document.getElementById('helenToast');

  let editingAlarmId = null;
  let pendingDeleteId = null;
  let alarms = [];

  const pad2 = (value) => String(value).padStart(2, '0');

  const showToast = (type, message) => {
    if (!toast) return;
    toast.className = 'helen-toast';
    toast.textContent = message || '';
    if (type === 'error') {
      toast.classList.add('error');
    } else {
      toast.classList.add('success');
    }
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 3000);
  };

  const hideLoading = () => {
    if (loadingElement) loadingElement.style.display = 'none';
  };

  const showNoAlarms = (visible) => {
    if (noAlarmsMessage) {
      noAlarmsMessage.style.display = visible ? 'flex' : 'none';
    }
  };

  const getSelectedDays = () => {
    if (!daysContainer) return [];
    return Array.from(daysContainer.querySelectorAll('.day.active'))
      .map((day) => Number(day.dataset.day))
      .filter((day) => Number.isFinite(day));
  };

  const setSelectedDays = (days) => {
    if (!daysContainer) return;
    const set = new Set(days || []);
    daysContainer.querySelectorAll('.day').forEach((day) => {
      const value = Number(day.dataset.day);
      day.classList.toggle('active', set.has(value));
    });
  };

  const formatDisplayTime = (alarm) => {
    const meta = alarm.metadata || {};
    const hours24 = Number(meta.hours24);
    const minutes = Number(meta.minutes);
    const date = new Date();
    if (Number.isFinite(hours24)) {
      date.setHours(hours24);
    }
    if (Number.isFinite(minutes)) {
      date.setMinutes(minutes);
    }
    const hours = Number.isFinite(hours24) ? hours24 : date.getHours();
    const mins = Number.isFinite(minutes) ? minutes : date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hours12 = ((hours % 12) || 12);
    return `${pad2(hours12)}:${pad2(mins)} ${ampm}`;
  };

  const formatRepeatDays = (alarm) => {
    const meta = alarm.metadata || {};
    const days = Array.isArray(meta.repeatDays) ? meta.repeatDays : [];
    return days;
  };

  const renderDaysChips = (container, daysArray) => {
    if (!container) return;
    container.innerHTML = '';
    if (!Array.isArray(daysArray) || daysArray.length === 0) return;
    const order = [1, 2, 3, 4, 5, 6, 0];
    const letter = { 0: 'D', 1: 'L', 2: 'M', 3: 'X', 4: 'J', 5: 'V', 6: 'S' };
    order.forEach((day) => {
      if (daysArray.includes(day)) {
        const span = document.createElement('span');
        span.textContent = letter[day];
        container.appendChild(span);
      }
    });
  };

  const buildAlarmElement = (alarm) => {
    const template = document.getElementById('alarm-template');
    if (!template) return null;
    const fragment = template.content.cloneNode(true);
    const alarmItem = fragment.querySelector('.alarm-item');
    const timeEl = fragment.querySelector('.alarm-time');
    const nameEl = fragment.querySelector('.alarm-name');
    const daysEl = fragment.querySelector('.alarm-days');
    const toggleInput = fragment.querySelector('.switch input');
    const editBtn = fragment.querySelector('.edit-btn');
    const deleteBtn = fragment.querySelector('.delete-alarm');

    alarmItem.dataset.id = alarm.id;
    timeEl.textContent = formatDisplayTime(alarm);
    nameEl.textContent = alarm.label || 'Alarma';
    renderDaysChips(daysEl, formatRepeatDays(alarm));

    const isActive = Boolean(alarm.metadata && alarm.metadata.active);
    toggleInput.checked = isActive;
    alarmItem.classList.toggle('inactive', !isActive);

    toggleInput.addEventListener('change', (event) => {
      scheduler.toggleAlarm(alarm.id, event.target.checked);
      showToast(event.target.checked ? 'success' : 'success', event.target.checked ? 'Alarma activada' : 'Alarma desactivada');
    });

    if (editBtn) {
      editBtn.addEventListener('click', () => openAlarmModal('edit', alarm));
    }
    if (deleteBtn) {
      deleteBtn.addEventListener('click', () => openDeleteModal(alarm.id));
    }

    return fragment;
  };

  const renderAlarms = () => {
    if (!alarmList) return;
    alarmList.querySelectorAll('.alarm-item').forEach((item) => item.remove());
    hideLoading();
    if (!alarms.length) {
      showNoAlarms(true);
      return;
    }
    showNoAlarms(false);
    const sorted = alarms.slice().sort((a, b) => {
      const aTime = a.targetEpochMs || a.createdAt || 0;
      const bTime = b.targetEpochMs || b.createdAt || 0;
      return aTime - bTime;
    });
    sorted.forEach((alarm) => {
      const fragment = buildAlarmElement(alarm);
      if (fragment) {
        alarmList.appendChild(fragment);
      }
    });
  };

  const openAlarmModal = (mode, alarm) => {
    if (!modal) return;
    editingAlarmId = null;
    if (mode === 'edit' && alarm) {
      editingAlarmId = alarm.id;
      if (modalTitle) modalTitle.textContent = 'Editar alarma';
      if (timeInput) {
        const hours24 = Number(alarm.metadata && alarm.metadata.hours24);
        const minutes = Number(alarm.metadata && alarm.metadata.minutes);
        const hoursValue = Number.isFinite(hours24) ? pad2(hours24) : '07';
        const minutesValue = Number.isFinite(minutes) ? pad2(minutes) : '00';
        timeInput.value = `${hoursValue}:${minutesValue}`;
      }
      if (labelInput) {
        labelInput.value = alarm.label || '';
      }
      setSelectedDays(formatRepeatDays(alarm));
      if (modalAction) modalAction.textContent = 'Guardar';
    } else {
      if (modalTitle) modalTitle.textContent = 'Nueva alarma';
      if (timeInput) timeInput.value = '07:00';
      if (labelInput) labelInput.value = '';
      setSelectedDays([]);
      if (modalAction) modalAction.textContent = 'Crear';
    }
    modal.classList.add('open');
  };

  const closeAlarmModal = () => {
    if (!modal) return;
    modal.classList.remove('open');
    editingAlarmId = null;
  };

  const openDeleteModal = (id) => {
    if (!deleteModal) return;
    pendingDeleteId = id;
    deleteModal.classList.add('open');
  };

  const closeDeleteModal = () => {
    if (!deleteModal) return;
    pendingDeleteId = null;
    deleteModal.classList.remove('open');
  };

  const getAlarmPayloadFromModal = () => {
    if (!timeInput) return null;
    const timeValue = timeInput.value || '07:00';
    const [hoursStr, minutesStr] = timeValue.split(':');
    const hours24 = Number(hoursStr);
    const minutes = Number(minutesStr);
    const ampm = hours24 >= 12 ? 'PM' : 'AM';
    let hours12 = hours24 % 12;
    if (hours12 === 0) hours12 = 12;
    return {
      hours: pad2(hours12),
      minutes: pad2(minutes),
      ampm,
      repeatDays: getSelectedDays(),
      label: labelInput ? labelInput.value.trim() : '',
    };
  };

  const createAlarmFromModal = () => {
    const payload = getAlarmPayloadFromModal();
    if (!payload) return;
    scheduler.createAlarm({
      hours: payload.hours,
      minutes: payload.minutes,
      ampm: payload.ampm,
      repeatDays: payload.repeatDays,
      label: payload.label,
      active: true,
    });
    showToast('success', 'Alarma creada');
    closeAlarmModal();
  };

  const saveEditAlarm = () => {
    if (!editingAlarmId) return;
    const payload = getAlarmPayloadFromModal();
    if (!payload) return;
    scheduler.updateAlarm(editingAlarmId, {
      hours: payload.hours,
      minutes: payload.minutes,
      ampm: payload.ampm,
      repeatDays: payload.repeatDays,
      label: payload.label,
    });
    showToast('success', 'Alarma actualizada');
    closeAlarmModal();
  };

  const deleteAlarm = (id) => {
    scheduler.deleteAlarm(id);
    showToast('success', 'Alarma eliminada');
  };

  const updateClock = () => {
    const nowDate = new Date();
    const timeElement = document.querySelector('.time');
    const dateElement = document.querySelector('.date-section');

    if (timeElement) {
      let hours = nowDate.getHours();
      const minutes = nowDate.getMinutes().toString().padStart(2, '0');
      hours = hours % 12;
      hours = hours ? hours : 12;
      timeElement.textContent = `${hours.toString().padStart(2, '0')}:${minutes}`;
    }

    if (dateElement) {
      const options = { weekday: 'short', day: 'numeric', month: 'short' };
      dateElement.textContent = nowDate.toLocaleDateString('es-ES', options);
    }
  };

  const refreshAlarms = () => {
    alarms = scheduler.list('alarm') || [];
    renderAlarms();
  };

  scheduler.on('update', refreshAlarms);

  document.addEventListener('DOMContentLoaded', () => {
    scheduler.ready().then(refreshAlarms);
    updateClock();
    setInterval(updateClock, 1000);

    if (newAlarmBtn) newAlarmBtn.addEventListener('click', () => openAlarmModal('create'));
    if (modalClose) modalClose.addEventListener('click', closeAlarmModal);
    if (modalCancel) modalCancel.addEventListener('click', closeAlarmModal);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) closeAlarmModal();
      });
    }
    if (modalAction) {
      modalAction.addEventListener('click', () => {
        if (editingAlarmId) {
          saveEditAlarm();
        } else {
          createAlarmFromModal();
        }
      });
    }

    if (daysContainer) {
      daysContainer.querySelectorAll('.day').forEach((day) => {
        day.addEventListener('click', () => {
          day.classList.toggle('active');
        });
      });
    }

    if (deleteCancel) deleteCancel.addEventListener('click', closeDeleteModal);
    if (deleteClose) deleteClose.addEventListener('click', closeDeleteModal);
    if (deleteModal) {
      deleteModal.addEventListener('click', (event) => {
        if (event.target === deleteModal) {
          closeDeleteModal();
        }
      });
    }
    if (deleteConfirm) {
      deleteConfirm.addEventListener('click', () => {
        if (pendingDeleteId) {
          deleteAlarm(pendingDeleteId);
        }
        closeDeleteModal();
      });
    }
  });
})();
