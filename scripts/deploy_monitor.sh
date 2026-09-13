#!/usr/bin/env bash
# Despliega la imagen con el Monitor encendido, por el camino de D-A6.4
# (docker save | ssh docker load). Deja la imagen que estaba servida etiquetada para
# rollback ANTES de subir nada.
#
# El host va por argumento y no aquí dentro: este archivo se commitea y la IP del
# servidor no entra al repo (sesión 13). La invocación con la IP vive en
# docs/MESA_JUICIO.md, que está gitignored.
#
#   scripts/deploy_monitor.sh root@<IP>
#   ALTUR_MONITOR_TOKEN=... scripts/deploy_monitor.sh root@<IP>      # token fijo
#
# Necesita hotspot: la WiFi del Tec bloquea SSH (D-A6.5). Tarda ~4 min por la subida
# de ~73 MB, y a los 120 s Claude Code lo manda a segundo plano — por eso lo corre
# Joaquín con `!`, no el agente.
set -euo pipefail

HOST=${1:-}
IMG=${IMG:-altur-detect:c2-mon}
ROLLBACK_TAG=${ROLLBACK_TAG:-altur-detect:c2-rollback}
NAME=${NAME:-altur-detect}

if [ -z "$HOST" ]; then
    echo "uso: $0 usuario@host   (p. ej. root@<IP del servidor>)" >&2
    exit 2
fi

# Token del monitor. Sin él las rutas /monitor responden 404 aunque el monitor esté
# encendido: el endpoint es público y está en la misma IP:puerto que /detect.
TOKEN=${ALTUR_MONITOR_TOKEN:-$(openssl rand -hex 8)}

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "1/6  imagen local"
LOCAL_ID=$(sg docker -c "docker image inspect $IMG --format '{{.Id}}'")
echo "  $IMG = $LOCAL_ID"

# El rollback se asegura ANTES de subir: si el ssh no funciona o el etiquetado falla,
# se descubre ahora y no después de cuatro minutos de subida y con el contenedor caído.
step "2/6  red de seguridad: etiquetar lo que está servido como $ROLLBACK_TAG"
ssh -o ConnectTimeout=15 "$HOST" "set -e
docker tag \"\$(docker inspect $NAME --format '{{.Image}}')\" $ROLLBACK_TAG
docker image inspect $ROLLBACK_TAG --format '  rollback listo: {{.Id}}'"

step "3/6  subiendo la imagen (~73 MB; por hotspot son ~4 min)"
sg docker -c "docker save $IMG" | gzip -1 | ssh -o ConnectTimeout=15 "$HOST" 'gunzip | docker load'

step "4/6  comprobando que la imagen remota es la misma"
REMOTE_ID=$(ssh "$HOST" "docker image inspect $IMG --format '{{.Id}}'")
echo "  remoto = $REMOTE_ID"
if [ "$LOCAL_ID" != "$REMOTE_ID" ]; then
    echo "  ID distinto: se aborta SIN tocar el contenedor. Sigue sirviendo el de antes." >&2
    exit 1
fi

step "5/6  cambiando el contenedor"
# Mismos flags de endurecimiento que D-A6.4, con el tmpfs acotado de docs/MESA_JUICIO.md:
# el JSONL del monitor vive ahí y el resto del filesystem es de solo lectura.
ssh "$HOST" "set -e
docker rm -f $NAME
docker run -d --name $NAME --restart unless-stopped \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges \
  -e WORKERS=4 -e ALTUR_MONITOR=1 -e ALTUR_MONITOR_TOKEN=$TOKEN \
  -p 8000:8000 $IMG
for i in \$(seq 60); do curl -fsS localhost:8000/health/ready >/dev/null 2>&1 && break; sleep 1; done
echo -n '  ready:   '; curl -s localhost:8000/health/ready; echo
echo -n '  version: '; curl -s localhost:8000/version | head -c 200; echo
echo -n '  monitor: '; curl -s -o /dev/null -w 'con token %{http_code}' \"localhost:8000/monitor?k=$TOKEN\"
curl -s -o /dev/null -w ', sin token %{http_code}\n' localhost:8000/monitor
echo -n '  reglas DROP de egress: '; iptables -S DOCKER-USER | grep -c DROP"

step "6/6  smoke oficial contra el servidor"
# Es el paso que decide si esto se queda o se revierte. 9/9 o rollback.
"$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/smoke.py" --url "http://${HOST#*@}:8000"

cat <<EOF

──────────────────────────────────────────────────────────────────────
  MONITOR:  http://${HOST#*@}:8000/monitor?k=$TOKEN
  Apúntalo en docs/MESA_JUICIO.md. Sin el token, /monitor responde 404.

  ROLLBACK (si algo sale mal; ~15 s, no sube nada):
  ssh $HOST 'docker rm -f $NAME && docker run -d --name $NAME \\
    --restart unless-stopped --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \\
    --cap-drop ALL --security-opt no-new-privileges -e WORKERS=4 \\
    -p 8000:8000 $ROLLBACK_TAG'
──────────────────────────────────────────────────────────────────────
EOF
