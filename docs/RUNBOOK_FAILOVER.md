# Runbook de failover

**FACT:** este procedimiento prepara el respaldo de `altur-detect` sin depender de quien lo
opere. **UNK:** el operador final sera Bini o Joaquin.

**OBS (A4, 2026-09-12):** el bundle y la imagen YA existen y las cuatro simulaciones locales
**se ejecutaron**. Los numeros de este documento estan medidos, no estimados. Lo que sigue
bloqueado esta marcado como tal y no se ha fingido: Vultr, URL/TLS, operador y la prueba desde
otra red.

Reproducir las simulaciones:

```bash
python scripts/build_bundle.py --out models/acoustic_ch0_v1
cp -a models/acoustic_ch0_v1 models/current          # seleccionar el candidato es un acto explicito
python scripts/build_bundle.py --out /tmp/altur-prev  # un build anterior real para el rollback
python scripts/failover_sim.py --bundle models/current --previous /tmp/altur-prev
```

**FACT:** el bundle es reproducible. Dos builds del mismo commit, mismos datos y misma semilla
producen un `model.json` **byte a byte identico**; lo unico que cambia es `created_utc`, y por
eso el SHA-256 del manifest identifica un *build*, no una especificacion. La identidad de la
especificacion son `run_id` y `code_digest`, que si son estables.

## Variables obligatorias

Sustituir todos los valores entre `<...>` antes del ensayo. No continuar con placeholders.

```bash
export PRIMARY_URL='<PRIMARY_URL>'
export BACKUP_URL='<BACKUP_URL>'
export IMAGE_REF='<REGISTRY_HOST>/<REPOSITORY>@sha256:<IMAGE_DIGEST>'
export EXPECTED_IMAGE_DIGEST='sha256:<IMAGE_DIGEST>'
export EXPECTED_BUNDLE_SHA256='<BUNDLE_SHA256>'
export EXPECTED_COMMIT='<FULL_GIT_COMMIT>'
export EXPECTED_CONCURRENCY='<CONCURRENT_REQUESTS>'
export OPERATOR='<BINI_OR_JOAQUIN>'
export REGISTRY_ACCESS='<REGISTRY_ACCESS_METHOD_OR_SECRET_REFERENCE>'
export REGISTRY_HOST='<REGISTRY_HOST>'
export REGISTRY_USER='<REGISTRY_USER>'
export REGISTRY_TOKEN='<REGISTRY_TOKEN_FROM_APPROVED_SECRET_STORE>'
export BACKUP_LISTEN_PORT='<BACKUP_HOST_PORT>'
export VALID_FIXTURE='<PATH_TO_GENERATED_NON_DATASET_STEREO_8KHZ_WAV>'
export PREVIOUS_IMAGE_REF='<REGISTRY_HOST>/<REPOSITORY>@sha256:<PREVIOUS_IMAGE_DIGEST>'
export PORT_BLOCKER_IMAGE='<REGISTRY_HOST>/<UTILITY_REPOSITORY>@sha256:<UTILITY_IMAGE_DIGEST>'
export OPEN_BACKUP_ROUTE_CMD='<NON_INTERACTIVE_COMMAND_TO_OPEN_BACKUP_ROUTE>'
export CLOSE_BACKUP_ROUTE_CMD='<NON_INTERACTIVE_COMMAND_TO_CLOSE_BACKUP_ROUTE>'
```

`REGISTRY_ACCESS` documenta el mecanismo autorizado; no es una credencial. No guardar tokens en
este repo, shell history, logs ni capturas.

## Preflight del host de respaldo

Ejecutar en el host de respaldo y guardar la salida en el registro privado del ensayo.

```bash
printf 'operator=%s utc=%s\n' "$OPERATOR" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nproc
free -h
df -h / /var/lib/docker
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}'
ss -ltn "sport = :$BACKUP_LISTEN_PORT"
timedatectl show -p NTPSynchronized -p TimeUSec --value
```

Salida esperada:

- `operator` no contiene placeholders y el reloj esta en UTC.
- CPU, RAM y disco cumplen los minimos medidos en A4.3: **2 vCPU, 1 GB RAM libre y 300 MB de
  disco**. Medido: imagen de **70 MB**, residente **137 MB** con 2 workers en reposo.
- Docker reporta versiones de cliente y servidor sin error de permisos.
- `ss` no devuelve un proceso escuchando en el puerto elegido.
- `NTPSynchronized` es `yes`. Si no, parar: los tiempos y certificados no son confiables.

## Activacion del respaldo

### 1. Autenticar y cargar la imagen por digest

```bash
printf '%s' "$REGISTRY_TOKEN" | docker login "$REGISTRY_HOST" \
  --username "$REGISTRY_USER" --password-stdin
docker pull "$IMAGE_REF"
docker image inspect "$IMAGE_REF" --format '{{join .RepoDigests "\n"}}'
```

Salida esperada: `Login Succeeded`, `Status: Downloaded newer image` o `Image is up to date`, y
una linea que termina exactamente en `$EXPECTED_IMAGE_DIGEST`. Una etiqueta flotante como
`:latest` no es aceptable.

### 2. Verificar el SHA-256 del bundle

El manifest del bundle enumera el SHA-256 de cada artefacto. En este runbook, el digest agregado
del bundle es el SHA-256 de `manifest.json`: al comprometer la lista de archivos y sus hashes,
compromete el contenido completo que verifica el cargador.

```bash
actual_bundle_sha256="$(docker run --rm --network none --entrypoint python "$IMAGE_REF" -c \
  'from altur.bundle import sha256_file,verify; p="/app/models/current"; problems=verify(p); assert not problems, problems; print(sha256_file(p + "/manifest.json"))')"
test "$actual_bundle_sha256" = "$EXPECTED_BUNDLE_SHA256"
printf 'bundle_files_sha256=ok\nbundle_sha256=%s\n' "$actual_bundle_sha256"
```

Salida esperada tras A4.1:

```text
bundle_files_sha256=ok
bundle_sha256=<BUNDLE_SHA256>
```

El comando termina con codigo distinto de cero si cualquier archivo falla su hash o si el digest
del manifest no coincide con `$EXPECTED_BUNDLE_SHA256`.

### 3. Arrancar sin egress y con reinicio automatico

```bash
docker network inspect altur-closed >/dev/null 2>&1 || \
  docker network create --internal altur-closed
docker rm -f altur-detect-backup >/dev/null 2>&1 || true
docker run -d --name altur-detect-backup \
  --restart unless-stopped \
  --network altur-closed \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges \
  -p "${BACKUP_LISTEN_PORT}:8000" \
  "$IMAGE_REF"
docker inspect altur-detect-backup --format \
  'status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}} network={{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}'
```

Salida esperada: id de contenedor y
`status=running restart=unless-stopped network=altur-closed`.

### 🔴 CORRECCION MEDIDA EN A4.3 — no usar `--internal` para el contenedor que sirve

**La INF anterior queda REFUTADA.** Se midio: con `docker network create --internal` **el puerto
publicado no funciona**. `docker inspect` devuelve `ports=map[8000/tcp:[]]` — el mapeo esta
vacio. El contenedor responde `200` en `/health/ready` **por dentro** y es **inalcanzable desde
el host**.

En un failover eso es el peor fallo posible: `docker inspect` reporta `running`, el healthcheck
interno pasa, y nadie puede consultarlo. El comando de arriba, tal como estaba escrito, producia
un respaldo muerto que parecia sano.

**Lo que si funciona (medido):** red bridge normal, con el resto del endurecimiento intacto.

```bash
docker run -d --name altur-detect-backup \
  --restart unless-stopped \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges \
  -p "${BACKUP_LISTEN_PORT}:8000" \
  "$IMAGE_REF"
```

**OBS:** en bridge el contenedor **si tiene egress** (se comprobo abriendo una conexion a
`1.1.1.1:80` desde dentro, y funciona). Por lo tanto **el "sin egress" hay que imponerlo en el
firewall del host, no en Docker.** Lo que si esta demostrado a nivel de aplicacion: el camino de
inferencia **no abre ningun socket** — se corrio `predict()` con `socket.socket` parcheado para
lanzar excepcion, y pasa. Y la imagen no trae sklearn, scipy, webrtcvad, pandas, matplotlib,
torch, librosa ni parselmouth: la unica dependencia numerica es NumPy.

**Arranque medido:** de `docker run` a `/health/ready` 200 en **1.52 s**, warm-up incluido
(29 ms). El warm-up corre ANTES de que readiness responda 200, que es lo unico que lo hace util.

### 4. Comprobar localmente y abrir la ruta

```bash
curl --fail --silent --show-error \
  "http://127.0.0.1:${BACKUP_LISTEN_PORT}/health/ready"
sh -ceu "$OPEN_BACKUP_ROUTE_CMD"
curl --fail --silent --show-error "$BACKUP_URL/health/ready"
```

Salida esperada: `{"status":"ready",...}` antes y despues de abrir la ruta. El comando de
ruta debe quedar congelado para el proveedor real (firewall, balanceador o DNS) antes del ensayo;
no se permite una consola interactiva durante el failover.

## Verificacion obligatoria desde otra red

Ejecutar estos pasos desde una red distinta a la primaria y a la del host de respaldo, por
ejemplo datos moviles. `VALID_FIXTURE` debe ser audio sintetico generado para pruebas, nunca una
llamada del dataset.

```bash
curl --fail --silent --show-error "$BACKUP_URL/health"
curl --fail --silent --show-error "$BACKUP_URL/health/ready"
curl --fail --silent --show-error "$BACKUP_URL/version" | tee /tmp/altur-version.json
curl --fail --silent --show-error \
  -H 'Content-Type: audio/wav' --data-binary "@$VALID_FIXTURE" \
  -D /tmp/altur-valid.headers -o /tmp/altur-valid.json \
  -w 'http=%{http_code} total_s=%{time_total}\n' "$BACKUP_URL/detect"
curl --silent --show-error \
  -H 'Content-Type: application/json' --data '{"audio":"!!!!"}' \
  -o /tmp/altur-invalid.json \
  -w 'http=%{http_code} total_s=%{time_total}\n' "$BACKUP_URL/detect"
```

Salida esperada:

- `/health` devuelve HTTP 200 y `status=alive`.
- `/health/ready` devuelve HTTP 200 y `status=ready`.
- `/version` devuelve `git_commit` o `bundle_commit=$EXPECTED_COMMIT`, el identificador del
  bundle correspondiente a `$EXPECTED_BUNDLE_SHA256` y ninguna ruta o secreto.
- El fixture valido devuelve HTTP 200 con solo `is_synthetic` booleano y `confidence` en `[0,1]`.
- El fixture invalido devuelve HTTP 400 y un `error` estable, sin stack trace.
- La respuesta valida incluye `X-Inference-Ms`; `curl` registra tambien tiempo total de red.

### Rafaga concurrente

```bash
rm -rf /tmp/altur-burst && mkdir -p /tmp/altur-burst
seq "$EXPECTED_CONCURRENCY" | xargs -P "$EXPECTED_CONCURRENCY" -I '{}' sh -ceu '
  code=$(curl --silent --show-error -H "Content-Type: audio/wav" \
    --data-binary "@$1" -o "/tmp/altur-burst/{}.json" \
    -w "%{http_code} %{time_total}" "$2/detect")
  printf "%s %s\n" "{}" "$code" > "/tmp/altur-burst/{}.result"
' sh "$VALID_FIXTURE" "$BACKUP_URL"
sort -n /tmp/altur-burst/*.result
```

Salida esperada: exactamente `$EXPECTED_CONCURRENCY` lineas, todas con HTTP `200`; cada JSON
cumple el contrato y todos los tiempos quedan registrados. Confirmar en el acta privada:

```text
operator=<...>
utc_start=<...>
primary_url=<...>
backup_url=<...>
image_digest=sha256:<...>
bundle_sha256=<...>
commit=<...>
valid_response=<redacted_schema_and_values>
invalid_response=<error_code_only>
single_request_total_s=<...>
single_request_inference_ms=<...>
concurrency=<...>
burst_successes=<...>/<...>
recovery_seconds=<...>
```

## Simulaciones locales — EJECUTADAS el 2026-09-12 (A4.4)

`scripts/failover_sim.py` corre los cuatro escenarios sobre el proceso y mide desde el incidente
hasta readiness. Resultado: **4/4 pasan.**

| escenario | resultado | tiempo |
|---|---|---|
| proceso caido | ✅ readiness cae y vuelve | **0.31 s** de relanzado |
| bundle corrupto | ✅ readiness 503, liveness 200, sin fuga de rutas | **0.30 s** hasta 503 |
| puerto ocupado | ✅ el arranque falla legible (`returncode=3`, `[Errno 98] address already in use`) | inmediato |
| rollback al bundle anterior | ✅ `/version` declara el build anterior | **0.31 s** |

**Estos numeros miden la recuperacion de la APLICACION, no la del contenedor ni la de la red.**
En el host, el relanzado lo hace `--restart unless-stopped` y hay que sumarle el arranque del
contenedor (**1.52 s** medido) y la propagacion de la ruta, que no esta medida.

### 🔴 Un smoke en verde NO prueba que el modelo este cuerdo

**OBS (A4.3, medido):** sobre cuatro entradas que no son voz —ruido blanco, silencio digital,
un tono puro de 440 Hz y ruido rosa— el bundle responde `is_synthetic=false` con **confianza
entre 0.9997 y 1.000000**. Una logistica sobre features estandarizadas extrapola en linea recta,
asi que fuera de distribucion la sigmoide satura. **El detector no tiene un "no se".**

Consecuencia para este runbook: `VALID_FIXTURE` es audio sintetico, asi que un smoke en verde
prueba **liveness y contrato**, no correccion del modelo. No se debe usar como criterio de
"el respaldo esta sirviendo bien". Es la misma patologia que D-A3.8 por otro eje: ahi el modelo
era mas confiado cuanto menos audio tenia; aqui lo es cuanto menos se parece la entrada a una
llamada.

Los comandos originales con Docker siguen abajo y son validos con la correccion de red de A4.3.

### Proceso caido

```bash
docker kill altur-detect-backup
docker inspect altur-detect-backup --format '{{.RestartCount}} {{.State.Status}}'
curl --retry 20 --retry-all-errors --retry-delay 1 --fail --silent --show-error \
  "$BACKUP_URL/health/ready"
```

Esperado: `RestartCount` aumenta, el estado vuelve a `running` y readiness se recupera por la
politica `unless-stopped`.

### Bundle corrupto

Con una copia local desechable del bundle, cambiar un byte de un artefacto declarado y montarla
solo lectura en un contenedor de prueba:

```bash
cp -a '<VALID_BUNDLE_DIRECTORY>' /tmp/altur-bundle-corrupt
printf 'x' >> '/tmp/altur-bundle-corrupt/<DECLARED_ARTIFACT>'
docker run --rm --network none \
  -v /tmp/altur-bundle-corrupt:/app/models/current:ro \
  --entrypoint python "$IMAGE_REF" -c \
  'from altur.bundle import verify; p=verify("/app/models/current"); assert p; print("rejected", p)'
```

Esperado: verificacion rechazada por `hash distinto`; nunca se activa silenciosamente una
prediccion constante.

**OBS (A4.2, medido):** con un byte cambiado el servicio arranca y queda asi en **0.30 s**:

| endpoint | codigo | cuerpo |
|---|---|---|
| `/health` | 200 | `status=alive`, `detector_loaded=false` |
| `/health/ready` | **503** | `status=not_ready`, `reason=["hash distinto en model.json"]` |
| `/version` | 200 | `bundle_ok=false`, `bundle_error=["hash distinto en model.json"]` |
| `/detect` | **503** | no responde ninguna prediccion |

El proceso queda **vivo a proposito**: un proceso muerto no diagnostica. Y se verifico que
`/version` no filtra ninguna ruta del host.

`ConstantDetector` solo se activa con `ALTUR_EMERGENCY_CONSTANT=1`, que es una decision
explicita del operador y queda en el acta. Nunca por defecto.

### Puerto ocupado

```bash
docker run -d --rm --name altur-port-blocker \
  -p "${BACKUP_LISTEN_PORT}:80" "$PORT_BLOCKER_IMAGE"
docker run --rm -p "${BACKUP_LISTEN_PORT}:8000" "$IMAGE_REF"
docker rm -f altur-port-blocker
```

Esperado: Docker rechaza el segundo arranque con `port is already allocated`; el preflight debe
detectar el conflicto antes de abrir la ruta. La imagen utilitaria debe precargarse por digest para
conservar el ensayo sin egress.

### Rollback al bundle anterior

```bash
docker pull "$PREVIOUS_IMAGE_REF"
docker rm -f altur-detect-backup
docker run -d --name altur-detect-backup --restart unless-stopped \
  --network altur-closed -p "${BACKUP_LISTEN_PORT}:8000" "$PREVIOUS_IMAGE_REF"
curl --retry 20 --retry-all-errors --retry-delay 1 --fail --silent --show-error \
  "$BACKUP_URL/version"
```

Esperado: `/version` reporta exactamente el commit y digest anteriores congelados, y el smoke
valido/invalido vuelve a pasar. No hacer rollback a una etiqueta flotante.

## Cierre y criterio de aceptacion

1. Cerrar la ruta con `sh -ceu "$CLOSE_BACKUP_ROUTE_CMD"` cuando termine el ensayo.
2. Conservar solo el acta agregada y hashes; borrar fixtures y respuestas temporales.
3. No registrar tokens, audio del dataset, IDs ni rutas privadas.
4. Marcar cualquier placeholder restante como bloqueo, no como exito.

**FACT:** una simulacion local valida comandos y recuperacion del proceso. **FACT:** no valida
DNS/TLS, firewall externo, carrier, NAT, conectividad del juez ni operacion humana real. Por eso la
simulacion local **NO sustituye** el ensayo fisico de `/health`, `/version` y `/detect` desde otra
red.
