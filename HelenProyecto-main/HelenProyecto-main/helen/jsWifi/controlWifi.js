/* =========================================================
   HELEN – WIFI (sin modal, compatible backend)
   Requiere: window.wifiAPI.{scanNetworks,getCurrentConnections,connect,forget}
   IDs usados: refresh-wifi, scan-label, networks-empty, wifi-list,
               password-container, wifi-password, toggle-password,
               password-toggle-icon, connect-wifi-button,
               connect-loader, connect-text, connecting-text
   ========================================================= */

// --------- Estado ----------
let selectedNetwork = null;
let currentConnection = null;
let networks = [];

// --------- Elementos ----------
const refreshWifiButton  = document.getElementById('refresh-wifi');
const scanLabel          = document.getElementById('scan-label');

const networksEmpty      = document.getElementById('networks-empty');
const networksEmptyMessage = networksEmpty ? networksEmpty.querySelector('p') : null;
const wifiList           = document.getElementById('wifi-list');

const wifiButton         = document.getElementById('wifi-button');
const wifiModal          = document.getElementById('wifi-modal');
const closeWifiModalBtn  = document.getElementById('close-wifi-modal');
const wifiStatusIcon     = document.getElementById('wifi-status-icon');

const passwordContainer  = document.getElementById('password-container');
const passwordInput      = document.getElementById('wifi-password');
const togglePassword     = document.getElementById('toggle-password');
const passwordToggleIcon = document.getElementById('password-toggle-icon');

const connectWifiButton  = document.getElementById('connect-wifi-button');
const connectLoader      = document.getElementById('connect-loader');
const connectText        = document.getElementById('connect-text');
const connectingText     = document.getElementById('connecting-text');

const hasElectron = !!window.electronAPI;
const hasWifiAPI  = !!(window.wifiAPI && window.wifiAPI.scanNetworks);

// --------- Utilidades UI ----------
function setScanningUI(isScanning) {
  if (!refreshWifiButton || !scanLabel) return;
  refreshWifiButton.classList.toggle('scanning', !!isScanning);
  scanLabel.textContent = isScanning ? 'Buscando...' : 'Buscar Redes';
}

function setNetworksEmptyState(visible, message) {
  if (!networksEmpty) return;
  networksEmpty.style.display = visible ? 'flex' : 'none';
  if (message && networksEmptyMessage) {
    networksEmptyMessage.textContent = message;
  }
}

function showPasswordPanel(show) {
  if (!passwordContainer) return;
  passwordContainer.style.display = show ? 'grid' : 'none';
}

function normalizeQuality(q) {
  const n = Number(q);
  if (Number.isFinite(n)) return Math.max(0, Math.min(100, Math.round(n)));
  return 0;
}

function isSecure(network) {
  // El backend puede devolver: '', null, 'wpa2', 'WPA', 'WPA2, WPA3', etc.
  if (!network) return false;
  const s = (network.security || '').toString().toLowerCase().trim();
  return !!(s && s !== 'open' && s !== 'none' && s !== 'abierta');
}

function barsIconFromQuality(q) {
  const pct = normalizeQuality(q);
  // Bootstrap Icons: bi-reception-0 .. -4
  if (pct >= 80) return 'bi-reception-4';
  if (pct >= 60) return 'bi-reception-3';
  if (pct >= 40) return 'bi-reception-2';
  if (pct >= 20) return 'bi-reception-1';
  return 'bi-reception-0';
}

// --------- Password toggle ----------
if (togglePassword) {
  togglePassword.addEventListener('click', () => {
    const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    passwordToggleIcon.className = (type === 'password') ? 'bi bi-eye-slash' : 'bi bi-eye';
  });
}

// --------- Conexión actual ----------
async function checkCurrentConnection() {
  try {
    if (!hasWifiAPI) {
      updateWifiStatusIcon(false);
      return false;
    }
    const current = await window.wifiAPI.getCurrentConnections();
    if (current && current.length) {
      currentConnection = current[0];
      localStorage.setItem('wifiConnected', 'true');
      localStorage.setItem('currentSSID', currentConnection.ssid);
      updateWifiStatusIcon(true);
      return true;
    }
    currentConnection = null;
    localStorage.setItem('wifiConnected', 'false');
    localStorage.removeItem('currentSSID');
    updateWifiStatusIcon(false);
    return false;
  } catch (e) {
    console.error('[WiFi] checkCurrentConnection:', e);
    currentConnection = null;
    updateWifiStatusIcon(false);
    return false;
  }
}

function updateWifiStatusIcon(isConnected) {
  if (!wifiStatusIcon) return;
  const stateClass = isConnected ? 'bi-wifi' : 'bi-wifi-off';
  wifiStatusIcon.className = `bi ${stateClass} wifi-icon`;
  const label = isConnected ? 'WiFi conectado' : 'WiFi desconectado';
  wifiStatusIcon.setAttribute('aria-label', label);
  wifiStatusIcon.setAttribute('title', label);
}

// --------- Render de red ----------
function renderNetworkItem(network, isActive) {
  const li = document.createElement('div');
  li.className = 'wifi-item' + (isActive ? ' active' : '');
  li.setAttribute('role', 'option');
  li.tabIndex = 0;
  const quality = normalizeQuality(network.quality);
  const bars = barsIconFromQuality(quality);
  const secure = isSecure(network);

  li.innerHTML = `
    <div class="wifi-signal"><i class="bi bi-wifi"></i></div>
    <div class="wifi-info">
      <div class="wifi-row">
        <div class="wifi-name">${network.ssid || '(SSID desconocido)'}</div>
        ${isActive ? '<span class="badge">Conectado</span>' : ''}
      </div>
      <div class="meta">
        <span class="lock">${secure ? '<i class="bi bi-lock"></i> Segura' : '<i class="bi bi-unlock"></i> Abierta'}</span>
        <span class="signal">${quality}%</span>
        <span class="bars"><i class="bi ${bars}"></i></span>
      </div>
    </div>
  `;

  const selectNetwork = () => {
    // selección visual
    document.querySelectorAll('.wifi-item.selected').forEach(el => el.classList.remove('selected'));
    li.classList.add('selected');

    // estado
    selectedNetwork = {
      ssid: network.ssid,
      quality,
      security: network.security
    };

    connectWifiButton.disabled = false;
    showPasswordPanel(secure);

    const isCurrent = !!(currentConnection && currentConnection.ssid === network.ssid);
    connectText.textContent = isCurrent ? 'Reconectar' : 'Conectar';
  };

  li.addEventListener('click', selectNetwork);
  li.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectNetwork();
    }
  });

  return li;
}

// --------- Escanear redes ----------
async function scanNetworks() {
  try {
    if (!wifiList) return;
    setScanningUI(true);
    selectedNetwork = null;
    connectWifiButton.disabled = true;
    showPasswordPanel(false);
    passwordInput.value = '';

    wifiList.innerHTML = '';
    setNetworksEmptyState(true, 'Buscando redes disponibles...');

    if (!hasWifiAPI) {
      // DEMO (cuando se abre en navegador sin backend)
      await new Promise(r => setTimeout(r, 800));
      networks = [
        { ssid: 'HELEN_Network', quality: 92, security: 'wpa2' },
        { ssid: 'Cafeteria',     quality: 70, security: 'wpa'  },
        { ssid: 'Invitados',     quality: 38, security: ''     }
      ];
    } else {
      networks = await window.wifiAPI.scanNetworks(); // ← backend real
    }

    // Ordenar por conectada y luego por calidad
    networks = (networks || []).map(n => ({
      ssid: n.ssid,
      quality: normalizeQuality(n.quality),
      security: n.security
    }));

    if (currentConnection && currentConnection.ssid) {
      const currentSsid = currentConnection.ssid;
      networks.sort((a, b) => (a.ssid === currentSsid ? -1 : b.ssid === currentSsid ? 1 : 0));
    }
    networks.sort((a, b) => b.quality - a.quality);

    wifiList.innerHTML = '';
    if (!networks.length) {
      setNetworksEmptyState(true, 'No se encontraron redes');
      return;
    }

    setNetworksEmptyState(false);
    networks.forEach(net => {
      const isActive = !!(currentConnection && currentConnection.ssid === net.ssid);
      wifiList.appendChild(renderNetworkItem(net, isActive));
    });
  } catch (err) {
    console.error('[WiFi] scanNetworks:', err);
    if (wifiList) {
      wifiList.innerHTML = '';
    }
    setNetworksEmptyState(true, 'Error al buscar redes');
  } finally {
    setScanningUI(false);
  }
}

// --------- Conectar ----------
async function connectToNetwork() {
  if (!selectedNetwork || !hasWifiAPI) return;

  // Loading UI
  if (connectLoader) connectLoader.style.display = 'inline-block';
  if (connectText) connectText.style.display = 'none';
  if (connectingText) connectingText.style.display = 'inline';
  if (connectWifiButton) connectWifiButton.disabled = true;

  try {
    const secure = isSecure(selectedNetwork);
    const password = secure ? (passwordInput.value || '') : undefined;

    // ¿Se cambia de red?
    const current = await window.wifiAPI.getCurrentConnections();
    const isChanging = current && current.length && current[0].ssid !== selectedNetwork.ssid;
    if (isChanging) {
      try { await window.wifiAPI.forget(current[0].ssid); } catch { /* ignore */ }
    }

    await window.wifiAPI.connect({
      ssid: selectedNetwork.ssid,
      password
    });

    // espera breve para que el sistema actualice
    await new Promise(r => setTimeout(r, 4000));
    const ok = await checkCurrentConnection();

    if (ok) {
      Swal.fire({
        icon: 'success',
        title: 'Conectado',
        text: `Conectado a ${selectedNetwork.ssid}`,
        timer: 2500,
        showConfirmButton: false,
        background: 'var(--card-bg)',
        color: 'var(--text)'
      });
      updateWifiStatusIcon(true);
      scanNetworks();
    } else {
      throw new Error('No se pudo establecer conexión');
    }
  } catch (e) {
    console.error('[WiFi] connectToNetwork:', e);
    const selectedSsid = selectedNetwork && selectedNetwork.ssid ? selectedNetwork.ssid : 'la red';
    Swal.fire({
      icon: 'error',
      title: 'Error de conexión',
      text: `No se pudo conectar a ${selectedSsid}. Verifica la contraseña e intenta de nuevo.`,
      confirmButtonText: 'OK',
      background: 'var(--card-bg)',
      color: 'var(--text)',
      confirmButtonColor: 'var(--primary)'
    });
  } finally {
    // Restablecer UI del botón
    if (connectLoader) connectLoader.style.display = 'none';
    if (connectText) connectText.style.display = 'inline';
    if (connectingText) connectingText.style.display = 'none';
    if (connectWifiButton) connectWifiButton.disabled = false;
  }
}

// --------- Eventos backend (opcional) ----------
if (hasElectron && window.electronAPI && typeof window.electronAPI.onWifiStatusChange === 'function') {
  window.electronAPI.onWifiStatusChange((event, data) => {
    currentConnection = data && data.currentConnection ? data.currentConnection : null;
    // Si quieres refrescar automáticamente al cambiar estado del SO:
    // scanNetworks();
  });
}

// --------- Arranque ----------
document.addEventListener('DOMContentLoaded', async () => {
  await checkCurrentConnection();
  await scanNetworks();
});

// --------- Listeners ----------
if (refreshWifiButton) {
  refreshWifiButton.addEventListener('click', scanNetworks);
}

if (connectWifiButton) {
  connectWifiButton.addEventListener('click', connectToNetwork);
}

if (passwordInput) {
  passwordInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') connectToNetwork();
  });
}

const isModalVisible = () => {
  if (!wifiModal) return false;
  return getComputedStyle(wifiModal).display !== 'none';
};

function openWifiModal() {
  if (!wifiModal) return;
  wifiModal.style.display = 'grid';
  if (document.body) {
    document.body.classList.add('wifi-modal-open');
  }
  scanNetworks();
}

function closeWifiModal() {
  if (!wifiModal) return;
  wifiModal.style.display = 'none';
  if (document.body) {
    document.body.classList.remove('wifi-modal-open');
  }
  selectedNetwork = null;
  if (wifiList) {
    wifiList.querySelectorAll('.wifi-item.selected').forEach(el => el.classList.remove('selected'));
  }
  showPasswordPanel(false);
  if (passwordInput) {
    passwordInput.value = '';
  }
  if (connectWifiButton) {
    connectWifiButton.disabled = true;
  }
  const hasNetworks = Array.isArray(networks) && networks.length > 0;
  if (!hasNetworks) {
    setNetworksEmptyState(true, 'Buscando redes disponibles...');
  }
}

if (wifiButton) {
  wifiButton.addEventListener('click', openWifiModal);
}

if (closeWifiModalBtn) {
  closeWifiModalBtn.addEventListener('click', closeWifiModal);
}

if (wifiModal) {
  wifiModal.addEventListener('click', (event) => {
    if (event.target === wifiModal) {
      closeWifiModal();
    }
  });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && isModalVisible()) {
    closeWifiModal();
  }
});
