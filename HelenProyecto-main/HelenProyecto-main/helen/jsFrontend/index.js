$(function () {
  // --- Selectores principales (ya existentes) ---
  const clock = $('.clock');          // <div class="clock">
  const dateSection = $('.date-section');   // <div class="date-section">
  const locationSection = $('.location');       // <div class="location">
  const weatherIcon = $('.weather-icon');   // <div class="weather-icon"> (usamos clases de Bootstrap Icons)
  const temperature = $('.temperature');    // <div class="temperature">
  const weatherDescription = $('.status');       // <div class="status">
  const activeSensor = $('.active-sensor');

  // --- NUEVOS selectores para métricas extra en la tarjeta ---
  const feelsLike = $('#feels-like-value');  // <span id="feels-like-value">
  const humidityValue = $('#humidity-value');    // <span id="humidity-value">
  const windValue = $('#wind-value');        // <span id="wind-value">
  const pressureValue = $('#pressure-value');     // ✅ nuevo
  const uvValue = $('#uv-value');          // <span id="uv-value">

  // =================== HORA/FECHA ===================
  const getTime = () => {
    const date = new Date();
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();
    const day = date.getDate();
    const month = date.getMonth();
    const year = date.getFullYear();
    const dayOfWeek = date.getDay();

    const days = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
    const months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];

    // HH:MM:SS (24h, con cero a la izquierda)
    const hh = String(hours).padStart(2, '0');
    const mm = String(minutes).padStart(2, '0');
    const ss = String(seconds).padStart(2, '0');
    clock.html(`${hh}:${mm}`);
    dateSection.html(`${days[dayOfWeek]}, ${day} de ${months[month]} de ${year}`);

    setTimeout(getTime, 1000);
  };

  // =================== UBICACIÓN (FIJA DE MOMENTO) ===================
  const getLocation = () => {
    const city = 'Tijuana';
    const country = 'MX';
    locationSection.html(`${city}, ${country}`);
  };

  // =================== CLIMA (OpenWeather) ===================
  const getWeather = async () => {
    try {
      const key = '6266f75957014a7de4ae0ded34d1e7cc';
      const lat = 32.43347;
      const lon = -116.67447;
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${key}&units=metric&lang=es`;

      const response = await axios.get(url);
      const data = response.data;

      const weatherInfo = {
        icon: data.weather[0]?.icon || '',
        temperature: Math.round(data.main?.temp ?? 0),
        description: (data.weather[0]?.description || '').toString()
      };

      // Map a Bootstrap Icons
      const mappingIcons = {
        '01d': 'bi-brightness-high', '01n': 'bi-moon',
        '02d': 'bi-cloud-sun', '02n': 'bi-cloud-moon',
        '03d': 'bi-cloud', '03n': 'bi-cloud',
        '04d': 'bi-clouds', '04n': 'bi-clouds',
        '09d': 'bi-cloud-drizzle', '09n': 'bi-cloud-drizzle',
        '10d': 'bi-cloud-rain', '10n': 'bi-cloud-rain',
        '11d': 'bi-cloud-lightning', '11n': 'bi-cloud-lightning',
        '13d': 'bi-cloud-snow', '13n': 'bi-cloud-snow',
        '50d': 'bi-cloud-haze', '50n': 'bi-cloud-haze'
      };

      // Evitar acumular clases en el icono
      const biClass = mappingIcons[weatherInfo.icon] || '';
      weatherIcon.attr('class', `weather-icon bi ${biClass}`);

      // Pintar estado y temperatura
      weatherDescription.html(weatherInfo.description.charAt(0).toUpperCase() + weatherInfo.description.slice(1));
      temperature.html(`${weatherInfo.temperature}°C`);

      // --- Métricas extra (si existen los elementos en el DOM) ---
      if (feelsLike.length) feelsLike.text(`${Math.round(data.main?.feels_like ?? weatherInfo.temperature)}°`);
      if (humidityValue.length) humidityValue.text(`${data.main?.humidity ?? '--'}%`);
      if (windValue.length) windValue.text(`${Math.round((data.wind?.speed ?? 0))} km/h`);
      if (uvValue.length) uvValue.text('—'); // UV lo dejamos fijo por ahora
      if (feelsLike.length)
        feelsLike.text(`${Math.round(data.main?.feels_like ?? weatherInfo.temperature)}°`);

      if (humidityValue.length)
        humidityValue.text(`${data.main?.humidity ?? '--'}%`);

      if (windValue.length) {
        // OpenWeather: velocidad del viento en m/s → km/h
        const windMs = data.wind?.speed ?? 0;
        windValue.text(`${Math.round(windMs * 3.6)} km/h`);
      }

      // ✅ Reemplazo: Presión atmosférica en hPa (en lugar de UV)
      const pressureValue = $('#pressure-value');
      if (pressureValue.length)
        pressureValue.text(`${Math.round(data.main?.pressure ?? 0)} hPa`);
    } catch (err) {
      console.error('[getWeather] error:', err);
      weatherDescription.html('No disponible');
      // Evita dejar valores “sucios”
      if (temperature.length) temperature.html('--°C');
    }

    // refresco cada 10 minutos
    setTimeout(getWeather, 1000 * 60 * 10);
  };

  // =================== OTROS ===================
  activeSensor.on('click', function () {
    alert('Sensor activado');
  });

  // Lanzar iniciales
  getTime();
  getLocation();
  getWeather();
});
