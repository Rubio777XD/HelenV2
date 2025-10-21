#!/bin/bash
#/usr/local/bin/connect-wifi.sh

SSID="$1"
PASSWORD="$2"

if [ -z "$SSID" ] || [ -z "$PASSWORD" ]; then
  echo "Uso: $0 \"SSID\" \"PASSWORD\""
  exit 1
fi

CONF="/etc/wpa_supplicant/wpa_supplicant.conf"
BACKUP="$CONF.bak"

# Crear copia de seguridad
cp "$CONF" "$BACKUP"

# Procesar con awk: actualizar si existe, agregar si no
awk -v ssid="$SSID" -v psk="$PASSWORD" '
BEGIN { in_block = 0; updated = 0 }
/^network=\{/ {
  block = $0
  in_block = 1
  next
}
/^\}/ {
  block = block "\n" $0
  if (block ~ "ssid=\"" ssid "\"") {
    # Actualizar el bloque con la nueva contraseña y asegurar disabled=0
    gsub(/psk="[^"]*"/, "psk=\"" psk "\"", block)
    if (block ~ /disabled=[01]/) {
      gsub(/disabled=[01]/, "disabled=0", block)
    } else {
      sub(/\}/, "  disabled=0\n}", block)
    }
    print block
    updated = 1
  } else {
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
END {
  if (!updated) {
    print ""
    print "network={"
    print "  ssid=\"" ssid "\""
    print "  psk=\"" psk "\""
    print "  key_mgmt=WPA-PSK"
    print "  disabled=0"
    print "}"
  }
}
' "$BACKUP" > "$CONF"

# Reconfigurar WiFi
wpa_cli -i wlan0 reconfigure

echo "Red \"$SSID\" actualizada o agregada correctamente y reconfigurado wpa_supplicant."
