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

socket.on('message', (data) => {
    console.log('Mensaje recibido del servidor:', data);

    if (data.character === 'Start') {
        isActive = true;
        console.log('Sistema activado. Ahora puedes realizar acciones.');
        showPopup('¡Sistema activado! Puedes realizar acciones ahora.', 'success');
        resetDeactivationTimer();
        return;
    }

    if (!isActive) {
        console.log('Sistema inactivo. Usa la seña "Start" para activarlo.');
        showPopup('Sistema inactivo. Usa la seña "Start" para activarlo.', 'info');
        return;
    }

    resetDeactivationTimer();

    switch (data.character) {
        case 'Alarma':
            console.log('Navegando a la alarma...');
            goToAlarm();
            break;

        case 'Clima':
            console.log('Navegando al clima...');
            goToWeather();
            break;

        case 'Inicio':
            console.log('Navegando al inicio...');
            goToHome();
            break;

        case 'Reloj':
            console.log('Navegando al reloj...');
            goToClock();
            break;

        case 'Ajustes':
            console.log('Navegando a ajustes...');
            goToSettings();
            break;
            
        case 'Dispositivos':
            console.log('Navegando a los dispositivos...');
            goToDevices();
            break;
    }
});