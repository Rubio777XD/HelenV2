#!/bin/bash
#/usr/local/bin/forget-wifi.sh

SSID="$1"

if [ -z "$SSID" ]; then
  echo "Uso: $0 \"SSID\""
  exit 1
fi

CONF="/etc/wpa_supplicant/wpa_supplicant.conf"

# Crear copia de seguridad
cp "$CONF" "$CONF.bak"

# Usar awk para eliminar el bloque network={...} que contiene el SSID
awk -v ssid="$SSID" '
BEGIN { skip = 0 }
/^network=\{/ { block = ""; skip = 0 }
/^network=\{/ { block = $0; in_block = 1; next }
/^\}/ {
  block = block "\n" $0
  if (block ~ "ssid=\"" ssid "\"") {
    skip = 1
  }
  if (!skip) {
    print block
  }
  in_block = 0
  next
}
{
  if (in_block) {
    block = block "\n" $0
    next
  }
  print $0
}
' "$CONF.bak" > "$CONF"

# Reconfigurar WiFi
wpa_cli -i wlan0 reconfigure

echo "Red \"$SSID\" eliminada correctamente del archivo y WiFi reconfigurado."
