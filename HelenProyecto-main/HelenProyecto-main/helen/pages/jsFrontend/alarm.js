// =========================
// Alarmas – UI nueva + compat
// =========================

let editingAlarmId = null;   // null => creando, string => editando esa alarma
let pendingDeleteId = null;  // id en espera de confirmación de borrado

document.addEventListener('DOMContentLoaded', function () {
    // ===== Layout viejo (protegido) =====
    const header = document.querySelector('.header');
    const alarmContainer = document.querySelector('.alarm-container');
    const currentTime = document.querySelector('.current-time');
    if (header) {
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
    }
    if (alarmContainer) alarmContainer.style.paddingTop = '0';
    if (currentTime && currentTime.remove) currentTime.remove();

    // ===== Inicialización =====
    updateClock();
    setInterval(updateClock, 1000);
    loadAlarms();

    // ===== Modal Crear/Editar =====
    const newAlarmBtn = document.getElementById('new-alarm-btn');
    const modal = document.getElementById('newAlarmModal');
    const modalClose = document.querySelector('#newAlarmModal .close-modal');
    const modalCancel = document.querySelector('#newAlarmModal .cancel-btn');
    const modalAction = document.querySelector('#newAlarmModal .create-btn'); // Crear/Guardar dinámico
    const modalTitle = document.querySelector('#newAlarmModal .modal-header h2');

    if (newAlarmBtn) newAlarmBtn.addEventListener('click', () => openAlarmModal('create'));
    if (modalClose) modalClose.addEventListener('click', () => modal.classList.remove('open'));
    if (modalCancel) modalCancel.addEventListener('click', () => modal.classList.remove('open'));
    if (modal) {
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('open'); });
    }

    // Acción principal del modal (Crear o Guardar)
    if (modalAction) {
        modalAction.addEventListener('click', () => {
            if (editingAlarmId) saveEditAlarm();
            else addAlarmFromModal();
        });
    }

    // Toggle de días en el modal
    document.querySelectorAll('.day-selectors .day').forEach(day => {
        day.addEventListener('click', function () {
            this.classList.toggle('active');
        });
    });

    // ===== Modal Confirmación de Borrado =====
    const deleteModal = document.getElementById('deleteConfirmModal');
    const deleteClose = document.querySelector('[data-close-delete]');
    const deleteCancel = document.querySelector('[data-cancel-delete]');
    const confirmDelete = document.getElementById('confirmDeleteBtn');

    if (deleteClose) deleteClose.addEventListener('click', () => { pendingDeleteId = null; deleteModal.classList.remove('open'); });
    if (deleteCancel) deleteCancel.addEventListener('click', () => { pendingDeleteId = null; deleteModal.classList.remove('open'); });
    if (deleteModal) deleteModal.addEventListener('click', (e) => { if (e.target === deleteModal) { pendingDeleteId = null; deleteModal.classList.remove('open'); } });
    if (confirmDelete) confirmDelete.addEventListener('click', () => {
        if (!pendingDeleteId) return;
        // Borrado real
        deleteAlarm(pendingDeleteId);
        pendingDeleteId = null;
        deleteModal.classList.remove('open');
        showToast('success', 'Alarma eliminada con éxito');
    });

    // ===== Compat: eliminar listeners del layout viejo =====
    const addBtnOld = document.getElementById('add-alarm');
    if (addBtnOld) addBtnOld.removeEventListener('click', addAlarm);

    const hourInputOld = document.getElementById('hours');
    const minuteInputOld = document.getElementById('minutes');
    if (hourInputOld) hourInputOld.replaceWith(hourInputOld.cloneNode(true));
    if (minuteInputOld) minuteInputOld.replaceWith(minuteInputOld.cloneNode(true));
});

// ===================== Reloj (seguro si no existe en DOM) =====================
function updateClock() {
    const now = new Date();
    const timeElement = document.querySelector('.time');
    const dateElement = document.querySelector('.date-section');

    if (timeElement) {
        let hours = now.getHours();
        const minutes = now.getMinutes().toString().padStart(2, '0');
        hours = hours % 12;
        hours = hours ? hours : 12; // 0 -> 12
        hours = hours.toString().padStart(2, '0');
        timeElement.textContent = `${hours}:${minutes}`;
    }

    if (dateElement) {
        const options = { weekday: 'short', day: 'numeric', month: 'short' };
        const formattedDate = now.toLocaleDateString('es-ES', options);
        dateElement.textContent = formattedDate;
    }

    checkAlarms(now);
}

// ===================== Storage helpers =====================
function saveAlarms(alarms) {
    localStorage.setItem('alarms', JSON.stringify(alarms));
}
function getAlarms() {
    return JSON.parse(localStorage.getItem('alarms')) || [];
}

// ===================== Cargar lista =====================
function loadAlarms() {
    const alarmList = document.querySelector('.alarm-list, .alarms-list');
    const loadingElement = document.querySelector('.alarm-loading');
    const noAlarmsMessage = document.querySelector('.no-alarms-message');
    if (!alarmList) return;

    setTimeout(() => {
        if (loadingElement) loadingElement.style.display = 'none';

        const alarms = getAlarms();

        // Limpiar elementos previos
        alarmList.querySelectorAll('.alarm-item').forEach(a => a.remove());

        if (alarms.length === 0) {
            if (noAlarmsMessage) noAlarmsMessage.style.display = 'flex';
            return;
        }
        if (noAlarmsMessage) noAlarmsMessage.style.display = 'none';

        alarms.forEach(alarm => {
            const el = createAlarmElement(alarm);
            alarmList.appendChild(el);
        });
    }, 300);
}

// ===================== Render de tarjeta =====================
function createAlarmElement(alarm) {
    const template = document.getElementById('alarm-template');
    const fragment = template.content.cloneNode(true);

    const alarmItem = fragment.querySelector('.alarm-item');
    alarmItem.dataset.id = alarm.id;

    // Hora (UI sin AM/PM)
    fragment.querySelector('.alarm-time').textContent = `${alarm.hours}:${alarm.minutes}`;

    // Etiqueta
    fragment.querySelector('.alarm-name').textContent = alarm.name || 'Alarma';

    // Días
    renderDaysChips(fragment.querySelector('.alarm-days'), alarm.days);

    // Estado
    const toggleInput = fragment.querySelector('.switch input');
    toggleInput.checked = !!alarm.active;
    if (!alarm.active) alarmItem.classList.add('inactive');
    toggleInput.addEventListener('change', function () {
        toggleAlarm(alarm.id, this.checked);
        const card = document.querySelector(`.alarm-item[data-id="${alarm.id}"]`);
        if (card) card.classList.toggle('inactive', !this.checked);
    });

    // Editar -> abre modal precargado
    const editBtn = fragment.querySelector('.edit-btn');
    if (editBtn) {
        editBtn.addEventListener('click', () => openAlarmModal('edit', alarm));
    }

    // Borrar -> abre modal propio de confirmación
    const delBtn = fragment.querySelector('.delete-alarm');
    if (delBtn) {
        delBtn.addEventListener('click', () => {
            pendingDeleteId = alarm.id;
            const deleteModal = document.getElementById('deleteConfirmModal');
            if (deleteModal) deleteModal.classList.add('open');
        });
    }

    return fragment;
}

function renderDaysChips(container, daysArray) {
    if (!container) return;
    container.innerHTML = '';
    if (!Array.isArray(daysArray) || daysArray.length === 0) return;
    // Orden visual: L M X J V S D  => 1,2,3,4,5,6,0
    const order = [1, 2, 3, 4, 5, 6, 0];
    const letter = { 0: 'D', 1: 'L', 2: 'M', 3: 'X', 4: 'J', 5: 'V', 6: 'S' };
    order.forEach(d => {
        if (daysArray.includes(d)) {
            const span = document.createElement('span');
            span.textContent = letter[d];
            container.appendChild(span);
        }
    });
}

// ===================== Crear desde MODAL =====================
function addAlarmFromModal() {
    const timeInput = document.getElementById('alarm-time-input');
    const labelInput = document.getElementById('alarm-label-input');

    const timeValue = (timeInput?.value || '').trim(); // "HH:MM" 24h
    if (!timeValue) { showToast('error', 'Por favor elige una hora válida'); return; }

    const [h24Str, mStr] = timeValue.split(':');
    let h24 = parseInt(h24Str, 10);
    const minutes = (mStr || '00').padStart(2, '0');

    const ampm = h24 >= 12 ? 'PM' : 'AM';
    let h12 = h24 % 12; if (h12 === 0) h12 = 12;
    const hours = String(h12).padStart(2, '0');

    const activeDays = [];
    document.querySelectorAll('.day-selectors .day.active').forEach(day => {
        activeDays.push(parseInt(day.dataset.day, 10));
    });

    const name = (labelInput?.value || '').trim();

    const alarmId = Date.now().toString();
    const newAlarm = { id: alarmId, hours, minutes, ampm, days: activeDays, name, active: true };

    const alarms = getAlarms();
    alarms.push(newAlarm);
    saveAlarms(alarms);

    resetAlarmModal();
    document.getElementById('newAlarmModal').classList.remove('open');
    showToast('success', 'Alarma agregada con éxito');
    loadAlarms();
}

// ===================== Editar =====================
function openAlarmModal(mode, alarm) {
    const modal = document.getElementById('newAlarmModal');
    const title = document.querySelector('#newAlarmModal .modal-header h2');
    const actionBtn = document.querySelector('#newAlarmModal .create-btn');
    const timeInput = document.getElementById('alarm-time-input');
    const labelInput = document.getElementById('alarm-label-input');
    const dayButtons = document.querySelectorAll('.day-selectors .day');

    // Reset selección de días
    dayButtons.forEach(d => d.classList.remove('active'));

    if (mode === 'edit' && alarm) {
        editingAlarmId = alarm.id;
        if (title) title.textContent = 'Editar Alarma';
        if (actionBtn) actionBtn.textContent = 'Guardar';

        // 12h -> 24h para <input type="time">
        let h = parseInt(alarm.hours, 10);
        if (alarm.ampm === 'PM' && h < 12) h += 12;
        if (alarm.ampm === 'AM' && h === 12) h = 0;
        const h24 = String(h).padStart(2, '0');
        const mm = String(alarm.minutes).padStart(2, '0');
        if (timeInput) timeInput.value = `${h24}:${mm}`;
        if (labelInput) labelInput.value = alarm.name || '';

        if (Array.isArray(alarm.days)) {
            alarm.days.forEach(d => {
                const btn = document.querySelector(`.day-selectors .day[data-day="${d}"]`);
                if (btn) btn.classList.add('active');
            });
        }
    } else {
        editingAlarmId = null;
        if (title) title.textContent = 'Nueva Alarma';
        if (actionBtn) actionBtn.textContent = 'Crear';
        if (timeInput) timeInput.value = '';
        if (labelInput) labelInput.value = '';
    }

    modal.classList.add('open');
}

function saveEditAlarm() {
    const timeInput = document.getElementById('alarm-time-input');
    const labelInput = document.getElementById('alarm-label-input');
    const timeValue = (timeInput?.value || '').trim();
    if (!timeValue) { showToast('error', 'Elige una hora válida'); return; }

    const [h24Str, mStr] = timeValue.split(':');
    let h24 = parseInt(h24Str, 10);
    const minutes = (mStr || '00').padStart(2, '0');

    const ampm = h24 >= 12 ? 'PM' : 'AM';
    let h12 = h24 % 12; if (h12 === 0) h12 = 12;
    const hours = String(h12).padStart(2, '0');

    const activeDays = [];
    document.querySelectorAll('.day-selectors .day.active').forEach(day => {
        activeDays.push(parseInt(day.dataset.day, 10));
    });

    const alarms = getAlarms();
    const idx = alarms.findIndex(a => a.id === editingAlarmId);
    if (idx !== -1) {
        alarms[idx].hours = hours;
        alarms[idx].minutes = minutes;
        alarms[idx].ampm = ampm;
        alarms[idx].days = activeDays;
        alarms[idx].name = (labelInput?.value || '').trim();
        saveAlarms(alarms);
    }

    document.getElementById('newAlarmModal').classList.remove('open');
    showToast('success', 'Alarma actualizada');
    loadAlarms();
    editingAlarmId = null;
}

function resetAlarmModal() {
    const labelInput = document.getElementById('alarm-label-input');
    const timeInput = document.getElementById('alarm-time-input');
    if (labelInput) labelInput.value = '';
    if (timeInput) timeInput.value = '';
    document.querySelectorAll('.day-selectors .day.active').forEach(d => d.classList.remove('active'));
}

// ===================== ON/OFF =====================
function toggleAlarm(alarmId, active) {
    const alarms = getAlarms();
    const i = alarms.findIndex(a => a.id === alarmId);
    if (i !== -1) {
        alarms[i].active = active;
        saveAlarms(alarms);
    }
}

// ===================== Borrar (sin SweetAlert, con modal propio) =====================
function deleteAlarm(alarmId) {
    const alarms = getAlarms();
    const filtered = alarms.filter(a => a.id !== alarmId);
    saveAlarms(filtered);

    const el = document.querySelector(`.alarm-item[data-id="${alarmId}"]`);
    if (el) el.remove();

    const noAlarmsMessage = document.querySelector('.no-alarms-message');
    if (filtered.length === 0 && noAlarmsMessage) {
        noAlarmsMessage.style.display = 'flex';
    }
}

// ===================== Disparo de alarmas =====================
function checkAlarms(currentTime) {
    const alarms = getAlarms();

    alarms.forEach(alarm => {
        if (!alarm.active) return;

        let hours = parseInt(alarm.hours, 10);
        const minutes = parseInt(alarm.minutes, 10);

        // 12h -> 24h
        if (alarm.ampm === 'PM' && hours < 12) {
            hours += 12;
        } else if (alarm.ampm === 'AM' && hours === 12) {
            hours = 0;
        }

        const now = currentTime;
        const alarmDay = now.getDay(); // 0..6

        if (
            (alarm.days.length === 0 || alarm.days.includes(alarmDay)) &&
            now.getHours() === hours &&
            now.getMinutes() === minutes &&
            now.getSeconds() === 0
        ) {
            ringAlarm(alarm);
        }
    });
}

function ringAlarm(alarm) {
    const audio = new Audio('https://cdn.freesound.org/previews/219/219244_4082826-lq.mp3');
    audio.loop = true;

    const timeStr = `${alarm.hours}:${alarm.minutes} ${alarm.ampm}`;
    const iconSVG = `
    <svg viewBox="0 0 24 24"><path d="M12 22a2.5 2.5 0 0 0 2.45-2h-4.9A2.5 2.5 0 0 0 12 22Zm7-6V11a7 7 0 0 0-5-6.71V3a2 2 0 1 0-4 0v1.29A7 7 0 0 0 5 11v5l-2 2v1h18v-1l-2-2Z"/></svg>
  `;

    Swal.fire({
        customClass: { popup: 'swal-helen' },
        showConfirmButton: false,
        html: `
      <div class="helen-alarm-card">
        <div class="helen-alarm-icon">${iconSVG}</div>
        <div>
          <div class="helen-alarm-title">${alarm.name || 'Alarma'}</div>
          <div class="helen-alarm-time">${timeStr}</div>
        </div>
        <div class="helen-alarm-actions">
          <button class="helen-stop-btn" id="helenStopSwal">Detener</button>
        </div>
      </div>
    `,
        allowOutsideClick: false,
        allowEscapeKey: false,
        background: 'transparent',
        backdrop: 'rgba(0,0,0,.45)',
        didOpen: () => {
            audio.play().catch(() => { });
            const btn = document.getElementById('helenStopSwal');
            if (btn) btn.addEventListener('click', () => Swal.close());
        },
        willClose: () => { try { audio.pause(); } catch (e) { } }
    });
}



// ===================== Toast propio (éxito / error) =====================
function showToast(type, message) {
    const t = document.getElementById('helenToast');
    if (!t) return;
    t.className = 'helen-toast';
    t.textContent = message || '';
    if (type === 'error') t.classList.add('error');
    else t.classList.add('success');
    t.style.display = 'block';
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => { t.style.display = 'none'; }, 3000);
}

// (Opcional) Si aún quieres usar los toasts de SweetAlert para otros mensajes:
function showAlert(type, message) {
    Swal.fire({
        toast: true,
        icon: type,
        title: message,
        position: 'top-end',
        showConfirmButton: false,
        timer: 3000,
        timerProgressBar: true
    });
}
