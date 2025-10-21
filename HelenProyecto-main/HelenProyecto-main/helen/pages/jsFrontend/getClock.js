// Zonas horarias de municipios de Baja California
const ZONAS_HORARIAS = {
  mexicali: "America/Tijuana",
  ensenada: "America/Tijuana",
  tecate: "America/Tijuana",
  rosarito: "America/Tijuana",
  tijuana: "America/Tijuana",
  chiapas: "America/Mexico_City",
};

// Configuración global
const CONFIG = {
  STORAGE_KEY: "selectedCity",
  DEFAULT_CITY: "tijuana",
  TIMEZONEDB_API_KEY: "SJABR4Q4XL7D", 
};

let ultimaHoraObtenida = null;
let ultimaFechaObtenida = null;
let formato24Horas = true; // Por defecto, formato 24h


/**
 * Obtiene la hora local desde la API de TimeZoneDB con reintentos en caso de error.
 * @param {string} timezone - Zona horaria.
 * @param {number} retries - Número de reintentos.
 * @param {number} delay - Tiempo de espera entre reintentos en milisegundos.
 */
async function obtenerHoraLocalTimeZoneDB(timezone, retries = 5, delay = 2000) {
  try {
    const response = await fetch(
      `https://api.timezonedb.com/v2.1/get-time-zone?key=${CONFIG.TIMEZONEDB_API_KEY}&format=json&by=zone&zone=${timezone}`
    );

    if (!response.ok) {
      throw new Error(`Error en la solicitud: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      horaLocal: data.formatted, // Hora en formato ISO (ejemplo: "2024-05-09 22:12:00")
    };
  } catch (error) {
    if (retries > 0) {
      console.warn(`Error al obtener la hora. Reintentando en ${delay / 1000} segundos...`);
      await new Promise((resolve) => setTimeout(resolve, delay)); // Espera antes de reintentar
      return obtenerHoraLocalTimeZoneDB(timezone, retries - 1, delay); // Reintenta
    } else {
      throw new Error(`Error después de varios intentos: ${error.message}`);
    }
  }
}

/**
 * Formatea la fecha en formato "día de Mes, Año" (ejemplo: "9 de Mayo, 2024")
 * @param {string} fecha - Fecha en formato ISO (ejemplo: "2024-05-09 22:12:00")
 */
function formatearFecha(fecha) {
  const meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
  ];
  const [fechaStr, horaStr] = fecha.split(" ");
  const [anio, mes, dia] = fechaStr.split("-");
  return `${dia} de ${meses[parseInt(mes) - 1]}, ${anio}`;
}

/**
 * Extrae la hora en formato "HH:MM:SS" desde una fecha ISO
 */
function extraerHora(fecha) {
  const [_, horaStr] = fecha.split(" ");
  const [hora, minuto, segundo] = horaStr.split(":");

  if (formato24Horas) {
    return `${hora}:${minuto}:${segundo}`;
  } else {
    let h = parseInt(hora);
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12; // convierte 0 a 12
    return `${h.toString().padStart(2, '0')}:${minuto}:${segundo} ${ampm}`;
  }
}


/**
 * Actualiza las etiquetas específicas con la hora y la fecha
 * @param {string} hora - Hora en formato "HH:MM:SS"
 * @param {string} fecha - Fecha en formato "día de Mes, Año"
 */
function actualizarEtiquetas(hora, fecha) {
  const clockItem = document.querySelector(".clock-item");
  const dateItem = document.querySelector(".date-item");

  if (clockItem && dateItem) {
    let horaMostrada = hora;

    if (!formato24Horas) {
      const [h, m, s] = hora.split(":").map(Number);
      const ampm = h >= 12 ? "PM" : "AM";
      const hora12 = h % 12 === 0 ? 12 : h % 12;
      horaMostrada = `${hora12.toString().padStart(2, "0")}:${m
        .toString()
        .padStart(2, "0")}:${s.toString().padStart(2, "0")} ${ampm}`;
    }

    clockItem.textContent = horaMostrada;
    dateItem.textContent = fecha;
  }
}


/**
 * Simula el avance del tiempo
 */
function simularAvanceTiempo() {
  if (!ultimaHoraObtenida || !ultimaFechaObtenida) return;

  const [hora, minuto, segundo] = ultimaHoraObtenida.split(":");
  let segundosTotales = parseInt(hora) * 3600 + parseInt(minuto) * 60 + parseInt(segundo);

  setInterval(() => {
    segundosTotales++;
    const nuevaHora = new Date(segundosTotales * 1000).toISOString().substr(11, 8);
    actualizarEtiquetas(nuevaHora, ultimaFechaObtenida);
  }, 1000);
}

/**
 * Obtiene la ubicación desde localStorage y actualiza la hora local
 */
async function actualizarHoraLocal() {
  const municipioKey = localStorage.getItem(CONFIG.STORAGE_KEY) || CONFIG.DEFAULT_CITY;


  if (!municipioKey || !ZONAS_HORARIAS[municipioKey]) {
    console.error("Municipio no encontrado en localStorage o zona horaria.");
    return;
  }

  const timezone = ZONAS_HORARIAS[municipioKey];

  try {
    // Obtiene la hora local desde la API de TimeZoneDB con reintentos
    const resultado = await obtenerHoraLocalTimeZoneDB(timezone);

    // Extrae la hora y la fecha de la respuesta
    ultimaHoraObtenida = extraerHora(resultado.horaLocal);
    ultimaFechaObtenida = formatearFecha(resultado.horaLocal);

    // Actualiza la fecha y hora para que se aprecie visualmente
    actualizarEtiquetas(ultimaHoraObtenida, ultimaFechaObtenida);

    // Inicia la simulación del avance del tiempo
    simularAvanceTiempo();
  } catch (error) {
    console.error(error.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const selector = document.getElementById("citySelector");

  // Si hay un valor guardado, lo selecciona
  const saved = localStorage.getItem(CONFIG.STORAGE_KEY) || CONFIG.DEFAULT_CITY;
  if (selector && saved) selector.value = saved;

  // Escuchar cambios en el selector
  if (selector) {
    selector.addEventListener("change", () => {
      const selected = selector.value;
      if (ZONAS_HORARIAS[selected]) {
        localStorage.setItem(CONFIG.STORAGE_KEY, selected);
        location.reload(); // Recargar para aplicar la nueva zona horaria
      }
    });
  }
});

function alternarFormatoHora() {
  formato24Horas = !formato24Horas;

  const btn = document.querySelector(".toggle-format-btn");
  btn.textContent = formato24Horas ? "Cambiar a 12 horas" : "Cambiar a 24 horas";

  // Corregido: ya tienes la hora lista, no la vuelvas a extraer
  if (ultimaHoraObtenida && ultimaFechaObtenida) {
    actualizarEtiquetas(ultimaHoraObtenida, ultimaFechaObtenida);
  }
}




// Actualiza la hora al cargar la página
actualizarHoraLocal();

// Actualiza la hora cada hora (3600000 ms = 1 hora)
setInterval(actualizarHoraLocal, 3600000);