/* 
 Archivo genérico para control de pestañas mediante señas
 NO EDITAR a menos que se agregue una pestaña nueva
 En caso de necesitar crear un control específico para x pestaña:

 Crea una copia de este archivo en la carpeta jsControl y elimina la declaración de actions.js en el html
 Declara la ruta de la copia siguiendo el siguiente formato: 

 <script src="../jsControl/(NombreDePestaña)Control.js"></script> 
 
 --- IMPORTANTE ---

 La declaración debe estar debajo de SocketIO y eventConnector, EJEMPLO:

  <script src="../../jsSignHandler/SocketIO.js"></script>
  <script src="../../jsSignHandler/eventConnector.js"></script>
  <script src="../jsControl/devicesControl.js"></script>

 Si no se tiene SocketIO y eventConnector las señas no funcionarán.

--- ----------- ---

 Finalmente edita -- UNICAMENTE -- el switch a corde a las señas requeridas por la pestaña 
 NO ELIMINES las declaraciones de pestaña, ya que no podrás cambiar a las pestañas eliminadas
 desde la pestaña actual. Por ejemplo, si se elimina 'Clima' desde la pestaña actual no podrás
 acceder a clima.*/

const ACTIVATION_ALIASES = ['start', 'activar', 'heyhelen', 'holahelen', 'oyehelen', 'wake'];
const GESTURE_KEYS = ['character', 'gesture', 'key', 'label', 'command', 'action'];
const NUMERIC_GESTURE_PATTERN = /^\d+$/;

const sanitizeGesture = (value) => {
    if (typeof value === 'string') {
        return value.trim();
    }
    if (value == null) {
        return '';
    }
    return String(value).trim();
};

const normalizeGesture = (gesture) => {
    if (!gesture) return '';
    try {
        return gesture.normalize('NFKC');
    } catch (error) {
        console.warn('No se pudo normalizar la seña recibida:', error);
        return gesture;
    }
};

const collapseGesture = (gesture) => {
    return gesture
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/\s+/g, '')
        .toLowerCase();
};

const isNumericGesture = (gesture) => {
    return NUMERIC_GESTURE_PATTERN.test(gesture);
};

const resolveGestureFromPayload = (payload = {}) => {
    for (const key of GESTURE_KEYS) {
        if (Object.prototype.hasOwnProperty.call(payload, key)) {
            const gesture = sanitizeGesture(payload[key]);
            if (gesture) {
                return gesture;
            }
        }
    }
    return '';
};

const gestureActions = {
    alarma: () => goToAlarm(),
    alarmas: () => goToAlarm(),
    clima: () => goToWeather(),
    weather: () => goToWeather(),
    inicio: () => goToHome(),
    home: () => goToHome(),
    reloj: () => goToClock(),
    hora: () => goToClock(),
    ajustes: () => goToSettings(),
    configuracion: () => goToSettings(),
    dispositivos: () => goToDevices(),
    foco: () => goToDevices(),
    devices: () => goToDevices()
};

const triggerActivationRing = (options = {}) => {
    if (typeof window.triggerActivationAnimation === 'function') {
        window.triggerActivationAnimation(options);
    }
};

const triggerRingErrorSignal = () => {
    if (typeof window.triggerRingError === 'function') {
        window.triggerRingError();
    }
};

const resetRingToIdle = () => {
    if (typeof window.setActivationRingState === 'function') {
        window.setActivationRingState('idle');
    }
};

const shouldInterceptNavigation = (collapsedGesture, normalizedGesture) => {
    if (window.helenTutorial && typeof window.helenTutorial.interceptNavigation === 'function') {
        try {
            return Boolean(window.helenTutorial.interceptNavigation(collapsedGesture, normalizedGesture));
        } catch (error) {
            console.warn('No se pudo interceptar la navegación en modo tutorial:', error);
        }
    }
    return false;
};

socket.on('message', (data = {}) => {
    if (data && data.active === false) {
        isActive = false;
        resetRingToIdle();
        return;
    }

    const rawGesture = resolveGestureFromPayload(data);
    const normalizedGesture = normalizeGesture(rawGesture);
    const collapsedGesture = collapseGesture(normalizedGesture);

    const collapsedState = data.state ? collapseGesture(normalizeGesture(sanitizeGesture(data.state))) : '';

    const isActivation = data.active === true
        || (collapsedState && ACTIVATION_ALIASES.some(alias => collapsedState === alias.replace(/\s+/g, '')))
        || ACTIVATION_ALIASES.some(alias => collapsedGesture === alias.replace(/\s+/g, ''));

    if (isActivation) {
        isActive = true;
        triggerActivationRing({ persist: true });
        resetDeactivationTimer();
        return;
    }

    if (!isActive) {
        resetRingToIdle();
        return;
    }

    if (!normalizedGesture) {
        console.warn('Se recibió un mensaje sin comando de seña identificable.');
        return;
    }

    if (isNumericGesture(collapsedGesture)) {
        console.log(`Comando numérico detectado: ${normalizedGesture}`);
        triggerActivationRing();
        resetDeactivationTimer();
        return;
    }

    resetDeactivationTimer();

    const action = gestureActions[collapsedGesture];
    if (typeof action === 'function') {
        console.log(`Ejecutando acción para la seña: ${normalizedGesture}`);
        triggerActivationRing();
        if (shouldInterceptNavigation(collapsedGesture, normalizedGesture)) {
            return;
        }
        action();
        return;
    }

    console.warn('Seña no reconocida:', normalizedGesture);
    triggerRingErrorSignal();
});
