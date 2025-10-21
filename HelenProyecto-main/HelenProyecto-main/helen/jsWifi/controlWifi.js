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
const wifiList           = document.getElementById('wifi-list');

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
togglePassword?.addEventListener('click', () => {
  const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
  passwordInput.setAttribute('type', type);
  passwordToggleIcon.className = (type === 'password') ? 'bi bi-eye-slash' : 'bi bi-eye';
});

// --------- Conexión actual ----------
async function checkCurrentConnection() {
  try {
    if (!hasWifiAPI) return false;
    const current = await window.wifiAPI.getCurrentConnections();
    if (current && current.length) {
      currentConnection = current[0];
      localStorage.setItem('wifiConnected', 'true');
      localStorage.setItem('currentSSID', currentConnection.ssid);
      return true;
    }
    currentConnection = null;
    localStorage.setItem('wifiConnected', 'false');
    localStorage.removeItem('currentSSID');
    return false;
  } catch (e) {
    console.error('[WiFi] checkCurrentConnection:', e);
    currentConnection = null;
    return false;
  }
}

// --------- Render de red ----------
function renderNetworkItem(network, isActive) {
  const li = document.createElement('div');
  li.className = 'wifi-item' + (isActive ? ' active' : '');
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

  li.addEventListener('click', () => {
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
  });

  return li;
}

// --------- Escanear redes ----------
async function scanNetworks() {
  try {
    setScanningUI(true);
    selectedNetwork = null;
    connectWifiButton.disabled = true;
    showPasswordPanel(false);
    passwordInput.value = '';

    wifiList.innerHTML = '';
    networksEmpty.style.display = 'flex';

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

    if (currentConnection?.ssid) {
      networks.sort((a, b) => (a.ssid === currentConnection.ssid ? -1 : b.ssid === currentConnection.ssid ? 1 : 0));
    }
    networks.sort((a, b) => b.quality - a.quality);

    wifiList.innerHTML = '';
    if (!networks.length) {
      networksEmpty.style.display = 'flex';
      networksEmpty.querySelector('p').textContent = 'No se encontraron redes';
      return;
    }

    networksEmpty.style.display = 'none';
    networks.forEach(net => {
      const isActive = !!(currentConnection && currentConnection.ssid === net.ssid);
      wifiList.appendChild(renderNetworkItem(net, isActive));
    });
  } catch (err) {
    console.error('[WiFi] scanNetworks:', err);
    wifiList.innerHTML = '';
    networksEmpty.style.display = 'flex';
    networksEmpty.querySelector('p').textContent = 'Error al buscar redes';
  } finally {
    setScanningUI(false);
  }
}

// --------- Conectar ----------
async function connectToNetwork() {
  if (!selectedNetwork || !hasWifiAPI) return;

  // Loading UI
  connectLoader.style.display = 'inline-block';
  connectText.style.display = 'none';
  connectingText.style.display = 'inline';
  connectWifiButton.disabled = true;

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
      scanNetworks();
    } else {
      throw new Error('No se pudo establecer conexión');
    }
  } catch (e) {
    console.error('[WiFi] connectToNetwork:', e);
    Swal.fire({
      icon: 'error',
      title: 'Error de conexión',
      text: `No se pudo conectar a ${selectedNetwork?.ssid || 'la red'}. Verifica la contraseña e intenta de nuevo.`,
      confirmButtonText: 'OK',
      background: 'var(--card-bg)',
      color: 'var(--text)',
      confirmButtonColor: 'var(--primary)'
    });
  } finally {
    // Restablecer UI del botón
    connectLoader.style.display = 'none';
    connectText.style.display = 'inline';
    connectingText.style.display = 'none';
    connectWifiButton.disabled = false;
  }
}

// --------- Eventos backend (opcional) ----------
if (hasElectron && window.electronAPI?.onWifiStatusChange) {
  window.electronAPI.onWifiStatusChange((event, data) => {
    currentConnection = data?.currentConnection || null;
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
refreshWifiButton?.addEventListener('click', scanNetworks);
connectWifiButton?.addEventListener('click', connectToNetwork);
passwordInput?.addEventListener('keyup', (e) => {
  if (e.key === 'Enter') connectToNetwork();
});
