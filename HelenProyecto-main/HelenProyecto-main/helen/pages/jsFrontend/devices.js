// ================== Estado global ==================
let selectedIcon = '';
let selectedAddIcon = '';
let deviceToDeleteId = null;
let deviceToEditId = null;
let nextDeviceId = 1;
let selectedRoom = '';
let selectedType = '';

const qs  = s => document.querySelector(s);
const qsa = s => document.querySelectorAll(s);

// ================== Init ==================
document.addEventListener('DOMContentLoaded', () => {
  ensureEmptyStateNode();
  updateUIState();
});

// ================== Helpers UI ==================
function ensureEmptyStateNode() {
  const list = qs('#deviceList');
  if (!list) return;

  // Crea contenedor de vacío si no existe
  if (!qs('#emptyState')) {
    const empty = document.createElement('section');
    empty.id = 'emptyState';
    empty.className = 'empty-state';
    empty.style.display = 'none';
    empty.innerHTML = `
      <div class="empty-hero">
        <div class="empty-icon">🏠</div>
        <h3 class="empty-title">Sin dispositivos configurados</h3>
        <p class="empty-sub">Conecta tus dispositivos inteligentes para controlarlos desde HELEN</p>
        <div class="empty-perks">
          <div class="perk"><i class="bi bi-wifi"></i><span>Conexión Automática</span></div>
          <div class="perk"><i class="bi bi-houses"></i><span>Control Centralizado</span></div>
          <div class="perk"><i class="bi bi-plus-circle"></i><span>Fácil Configuración</span></div>
        </div>
        <button class="add-button" onclick="showAddModal()">+ Agregar dispositivo</button>
      </div>
    `;
    // Inserta el vacío justo antes de la lista para mantener layout
    list.parentNode.insertBefore(empty, list);
  }
}

function updateUIState() {
  const list     = qs('#deviceList');
  const empty    = qs('#emptyState');
  const subtitle = qs('#devices-subtitle');

  if (!list) return;

  const count = list.querySelectorAll('.device-card').length;

  if (count > 0) {
    list.style.display = 'grid';
    if (empty) empty.style.display = 'none';
    if (subtitle) subtitle.textContent = 'Tus dispositivos conectados';
  } else {
    list.style.display = 'none';
    if (empty) empty.style.display = 'block';
    if (subtitle) subtitle.textContent = 'No hay dispositivos conectados aún';
  }
}

// ================== Eliminar ==================
function showDeleteModal(deviceId) {
  deviceToDeleteId = deviceId;
  const m = qs('#deleteModal');
  if (m) m.style.display = 'grid';
}
function hideDeleteModal() {
  const m = qs('#deleteModal');
  if (m) m.style.display = 'none';
  deviceToDeleteId = null;
}
function confirmDelete() {
  if (!deviceToDeleteId) return;
  const el = qs(`#device-${deviceToDeleteId}`);
  if (el) el.remove();
  hideDeleteModal();
  updateUIState();
}

// ================== Editar ==================
function showEditModal(deviceId) {
  deviceToEditId = deviceId;

  const card = qs(`#device-${deviceId}`);
  if (!card) return;

  const currentName = card.querySelector('.device-name')?.textContent || '';
  const currentIcon = card.querySelector('.device-icon')?.textContent || '';

  // set selección inicial por nombre "Sala - Luz"
  const [room, type] = currentName.split(' - ');
  if (room) {
    const r = qs(`#editModal .room-option[data-room="${room}"]`);
    r && selectRoom(r, room);
  }
  if (type) {
    const t = qs(`#editModal .type-option[data-type="${type}"]`);
    t && selectDeviceType(t, type);
  }

  // icono actual
  qsa('#editModal .icon-option').forEach(opt => {
    const match = opt.textContent.trim() === currentIcon.trim();
    opt.classList.toggle('selected', match);
    if (match) selectedIcon = currentIcon;
  });

  const m = qs('#editModal');
  if (m) m.style.display = 'grid';
  updateDeviceNamePreview('edit');
}

function hideEditModal() {
  const m = qs('#editModal');
  if (m) m.style.display = 'none';
  deviceToEditId = null;
  selectedRoom = '';
  selectedType = '';
  selectedIcon = '';
}

function selectIcon(el, icon) {
  qsa('#editModal .icon-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedIcon = icon;
  suggestDeviceType(icon, 'edit');
}

function selectRoom(el, room) {
  qsa('#editModal .room-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedRoom = room;
  updateDeviceNamePreview('edit');
}

function selectDeviceType(el, type) {
  qsa('#editModal .type-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedType = type;
  updateDeviceNamePreview('edit');
}

function updateDeviceNamePreview(mode) {
  const span = qs(`#${mode === 'edit' ? 'deviceNamePreview' : 'newDeviceNamePreview'}`);
  if (!span) return;

  if (selectedRoom && selectedType) {
    span.textContent = `${selectedRoom} - ${selectedType}`;
    span.parentElement.classList.remove('name-preview-empty');
  } else if (selectedRoom) {
    span.textContent = `${selectedRoom} - ...`;
    span.parentElement.classList.remove('name-preview-empty');
  } else if (selectedType) {
    span.textContent = `... - ${selectedType}`;
    span.parentElement.classList.remove('name-preview-empty');
  } else {
    span.textContent = 'Selecciona ubicación y tipo';
    span.parentElement.classList.add('name-preview-empty');
  }
}

function suggestDeviceType(icon, mode) {
  let suggested = '';
  switch (icon) {
    case '💡': suggested = 'Luz'; break;
    case '🔌': suggested = 'Enchufe'; break;
    case '🌡️': suggested = 'Termostato'; break;
    case '📺': suggested = 'TV'; break;
    case '🔊': suggested = 'Altavoz'; break;
    case '🎮': suggested = 'Consola'; break;
    case '🚪': suggested = 'Puerta'; break;
    case '🔒': suggested = 'Cerradura'; break;
    case '💧': suggested = 'Riego'; break;
    case '☕': suggested = 'Cafetera'; break;
    case '💻': suggested = 'Ordenador'; break;
    case '⏰': suggested = 'Alarma'; break;
    case '📱': suggested = 'Móvil'; break;
    case '🎛️': suggested = 'Control'; break;
    case '🖨️': suggested = 'Impresora'; break;
    default: return;
  }
  const scope = mode === 'edit' ? '#editModal' : '#addModal';
  const el = qs(`${scope} .type-option[data-type="${suggested}"]`);
  if (!el) return;
  mode === 'edit' ? selectDeviceType(el, suggested) : selectAddDeviceType(el, suggested);
}

function saveDeviceChanges() {
  if (!selectedRoom || !selectedType) { alert('Selecciona ubicación y tipo'); return; }
  if (!selectedIcon) { alert('Selecciona un ícono'); return; }

  const card = qs(`#device-${deviceToEditId}`);
  if (!card) return;

  card.querySelector('.device-name').textContent = `${selectedRoom} - ${selectedType}`;
  card.querySelector('.device-icon').textContent = selectedIcon;

  hideEditModal();
}

// ================== Agregar ==================
function showAddModal() {
  selectedAddIcon = '';
  selectedRoom = '';
  selectedType = '';

  qsa('#addModal .icon-option').forEach(o => o.classList.remove('selected'));
  qsa('#addModal .room-option').forEach(o => o.classList.remove('selected'));
  qsa('#addModal .type-option').forEach(o => o.classList.remove('selected'));

  updateDeviceNamePreview('add');
  const m = qs('#addModal');
  if (m) m.style.display = 'grid';
}
function hideAddModal() {
  const m = qs('#addModal');
  if (m) m.style.display = 'none';
  selectedRoom = '';
  selectedType = '';
  selectedAddIcon = '';
}

function selectAddIcon(el, icon) {
  qsa('#addModal .icon-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedAddIcon = icon;
  suggestDeviceType(icon, 'add');
}
function selectAddRoom(el, room) {
  qsa('#addModal .room-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedRoom = room;
  updateDeviceNamePreview('add');
}
function selectAddDeviceType(el, type) {
  qsa('#addModal .type-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  selectedType = type;
  updateDeviceNamePreview('add');
}

function addNewDevice() {
  if (!selectedRoom || !selectedType) { alert('Por favor selecciona una ubicación y un tipo para el dispositivo'); return; }
  if (!selectedAddIcon) { alert('Por favor selecciona un ícono para el dispositivo'); return; }

  const deviceName = `${selectedRoom} - ${selectedType}`;

  const list = qs('#deviceList');
  if (!list) return;

  const id = nextDeviceId++;

  const card = document.createElement('div');
  card.className = 'device-card is-off'; // inicia apagado
  card.id = `device-${id}`;
  card.innerHTML = `
    <div class="device-header">
      <div class="device-icon">${selectedAddIcon}</div>
      <div class="device-name">${deviceName}</div>
    </div>
    <div class="device-status">
      <label class="switch">
        <input type="checkbox" onchange="toggleDeviceStatus(this, ${id})">
        <span class="slider"></span>
      </label>
      <span class="device-status-text">Apagado</span>
      <span class="status-badge status-off">OFF</span>
    </div>
    <div class="device-actions">
      <button class="action-button" onclick="showEditModal(${id})" aria-label="Editar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <button class="action-button" onclick="showDeleteModal(${id})" aria-label="Eliminar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  `;
  list.appendChild(card);

  hideAddModal();
  updateUIState();
}

// ================== Encendido/Apagado ==================
function toggleDeviceStatus(checkbox, deviceId) {
  const card       = qs(`#device-${deviceId}`);
  if (!card) return;

  const statusText  = card.querySelector('.device-status-text');
  const statusBadge = card.querySelector('.status-badge');
  const isOn = checkbox.checked;

  if (statusText)  statusText.textContent  = isOn ? 'Encendido' : 'Apagado';
  if (statusBadge) {
    statusBadge.textContent = isOn ? 'ON' : 'OFF';
    statusBadge.className   = 'status-badge ' + (isOn ? 'status-on' : 'status-off');
  }
  card.classList.toggle('is-off', !isOn);

  // Hook opcional para backend
  if (typeof window.onDeviceToggle === 'function') {
    try { window.onDeviceToggle(deviceId, isOn); } catch (_) {}
  }
}

// ================== Vacío (compatibilidad antigua) ==================
function checkEmptyState() {
  // Compat: algunas vistas llamaban a esta función
  ensureEmptyStateNode();
  updateUIState();
}

// ========== Exponer para los onclick del HTML ==========
window.showDeleteModal = showDeleteModal;
window.hideDeleteModal = hideDeleteModal;
window.confirmDelete   = confirmDelete;

window.showEditModal   = showEditModal;
window.hideEditModal   = hideEditModal;
window.selectIcon      = selectIcon;
window.selectRoom      = selectRoom;
window.selectDeviceType= selectDeviceType;
window.saveDeviceChanges = saveDeviceChanges;

window.showAddModal    = showAddModal;
window.hideAddModal    = hideAddModal;
window.selectAddIcon   = selectAddIcon;
window.selectAddRoom   = selectAddRoom;
window.selectAddDeviceType = selectAddDeviceType;
window.addNewDevice    = addNewDevice;

window.toggleDeviceStatus = toggleDeviceStatus;
window.checkEmptyState    = checkEmptyState;
