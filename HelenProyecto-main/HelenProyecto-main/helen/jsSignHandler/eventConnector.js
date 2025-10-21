// eventConnector.js - Versión modificada
// Conexión al socket y mejoras en la navegación

// Inicializar la conexión al socket
const socket = io('http://127.0.0.1:5000');

let isActive = false;
let timeoutId;
let lastNotification = "";

const DEACTIVATION_DELAY = 3000;

// Función para determinar la ruta base según el contexto
const getBasePath = () => {
  const path = window.location.pathname;
  if (path.includes('/pages/')) {
    return '../';
  }
  return '../';
};

const showPopup = (message, type) => {
    if (message === lastNotification) return;
    lastNotification = message;
    Swal.fire({
        title: message,
        icon: type,
        showConfirmButton: false,
        timer: 3000,
        toast: true,
        position: 'top-end',
    });
};

const resetDeactivationTimer = () => {
    if (timeoutId) {
        clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
        isActive = false;
        console.log('Sistema desactivado automáticamente por inactividad.');
        showPopup('Sistema desactivado por inactividad.', 'warning');
    }, DEACTIVATION_DELAY);
};
const goToPageWithLoading = (targetUrl, pageName) => {
    const currentUrl = window.location.href;
    if (currentUrl.includes(targetUrl)) {
        console.log(`Ya estás en ${targetUrl}, no se necesita redirección.`);
        return;
    }
    
    const basePath = getBasePath();
    const fullTargetUrl = targetUrl.startsWith('/') ? targetUrl : basePath + targetUrl;
    
    if (window.loadingScreen) {
        console.log(`Mostrando pantalla de carga para: ${pageName}`);
        
        const navigationFunction = async () => {

            await new Promise(resolve => setTimeout(resolve, 800));
            console.log(`Navegando a: ${fullTargetUrl}`);
            
            if (window.myAPI && window.myAPI.navigate) {
                window.myAPI.navigate(fullTargetUrl);
            } else {
                window.location.href = fullTargetUrl;
            }
        };
        
        window.loadingScreen.showAndExecute(
            navigationFunction,
            `Cargando ${pageName}...`
        );
    } else {
        console.log(`Navegando a: ${fullTargetUrl}`);
        if (window.myAPI && window.myAPI.navigate) {
            window.myAPI.navigate(fullTargetUrl);
        } else {
            window.location.href = fullTargetUrl;
        }
    }
};

const enhancedGoToClock = () => goToPageWithLoading("pages/clock/clock.html", "Reloj");
const enhancedGoToWeather = () => goToPageWithLoading("pages/weather/weather.html", "Clima");
const enhancedGoToDevices = () => goToPageWithLoading("pages/devices/devices.html", "Dispositivos");
const enhancedGoToHome = () => goToPageWithLoading("index.html", "Inicio");
const enhancedGoToAlarm = () => goToPageWithLoading("pages/clock/alarm.html", "Alarma");
const enhancedGoToSettings = () => goToPageWithLoading("pages/settings/settings.html", "Ajustes");

window.goToClock = enhancedGoToClock;
window.goToWeather = enhancedGoToWeather;
window.goToDevices = enhancedGoToDevices;
window.goToHome = enhancedGoToHome;
window.goToAlarm = enhancedGoToAlarm;
window.goToSettings = enhancedGoToSettings;
