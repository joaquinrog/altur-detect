# Primer ensayo del Monitor con dos laptops

Fecha de preparación: 2026-09-12

## Ruta automatizada para Joaquín

La guía interactiva para una sesión nueva está en `PROMPT_DEMO_BINI_ALTUR.md`. La automatización
local que usa esa guía es:

```bash
./.venv/bin/python scripts/demo_operator.py start
./.venv/bin/python scripts/demo_operator.py watch --duration 900
./.venv/bin/python scripts/demo_operator.py status
./.venv/bin/python scripts/demo_operator.py stop
```

`start` valida imagen y puerto, genera un token efímero, levanta el contenedor endurecido, espera
readiness, descubre la IP LAN y abre el Monitor. `watch` reporta llamadas nuevas, errores, caídas y
latencias mayores a 30 s. `stop` apaga solo `altur-demo` y elimina el token de `/tmp`.

## Estado verificado antes del ensayo

- UI implementada en `src/altur/monitor_ui.py` y revisada por dos builders.
- Suite completa: 408 tests pasan, 1 skip; Ruff limpio.
- Imagen local construida: `altur-detect:c2-mon-ui-s15`, ID `84b47840bd9e`.
- Smoke local de la imagen: readiness correcto y detector `spectral_factory_lfcc_logreg@1`.
- Dos llamadas reales de `train`: seed 4 clasificada sintética y seed 5 clasificada humana, ambas contestadas sin error.
- Capturas revisadas a 1440×1000 y 390×844: sin bloqueantes ni recortes de la información crítica.
- Esta imagen todavía no está desplegada en Vultr. Este documento cubre primero el ensayo local entre las dos laptops.

## Objetivo

Reproducir el flujo del turno de Altur:

- La laptop de Joaquín sirve `POST /detect` y muestra el Monitor.
- La laptop de Bini actúa como el cliente de Altur y envía llamadas por la red.
- Bini evalúa si la pantalla comunica, sin explicación previa, estas tres respuestas:
  1. Si la última llamada fue clasificada como humana o sintética.
  2. La confianza calibrada en ese veredicto.
  3. La latencia de esa llamada frente al límite de 30 segundos.

Este ensayo prueba el Monitor y la conexión entre laptops. No sustituye el e2e completo ni valida los claims del pitch.

## Qué no se debe confundir

- `confidence` no es accuracy. Es la probabilidad calibrada de la clase reportada. Si el resultado es humano y muestra 96 %, significa 96 % de confianza en `humana`.
- El tiempo del Monitor cubre subida al servidor, decodificación e inferencia. No incluye handshake ni viaje de vuelta, por lo que queda ligeramente debajo del tiempo del cliente (~2 RTT).
- El bundle servido actualmente es `spectral_factory_lfcc_logreg@1`: 120 features LFCC del caller. No es la late fusion LFCC + YIN descrita en el bloque 7 del guion 3.0.
- Los “182 ms end-to-end” del bloque 9 no deben compararse verbalmente con una llamada enviada por WiFi: la red y el tamaño del cuerpo cambian el número.
- Este ensayo usa solamente `train`. Nunca usar `val`, `all` ni `--use-val`.

## Preparación

- Ambas laptops conectadas al mismo WiFi.
- Laptop de Joaquín conectada a corriente y sin suspensión automática.
- Puerto `8000` libre en la laptop de Joaquín.
- Imagen candidata `altur-detect:c2-mon-ui-s15` disponible en la laptop de Joaquín. Verificar antes de empezar:

```bash
sg docker -c "docker image inspect altur-detect:c2-mon-ui-s15 --format '{{.Id}}'"
```

Debe comenzar con `sha256:84b47840bd9e` mientras no se reconstruya la imagen.
- Bini tiene una copia local autorizada de `altur-detect` con `data/manifest.csv`, `data/audio/` y `data/official/check_endpoint.py`.
- Cerrar notificaciones y aplicaciones que puedan tapar el Monitor.

## 1. Laptop de Joaquín: obtener IP

Ejecutar justo antes del ensayo; la IP cambia por DHCP:

```bash
ip -4 -br addr show wlp0s20f3
```

Al preparar este documento era `10.22.219.231/20`. No usarla de memoria.

## 2. Laptop de Joaquín: levantar el endpoint

Desde `~/Desktop/altur-detect`:

```bash
export ALTUR_MONITOR_TOKEN=$(openssl rand -hex 8)
```

```bash
sg docker -c "docker run -d --rm \
  --name altur-mesa \
  -p 0.0.0.0:8000:8000 \
  -e ALTUR_MONITOR=1 \
  -e ALTUR_MONITOR_TOKEN=$ALTUR_MONITOR_TOKEN \
  altur-detect:c2-mon-ui-s15"
```

Esperar readiness:

```bash
curl http://127.0.0.1:8000/health/ready
```

Debe responder `status: ready` y detector `spectral_factory_lfcc_logreg@1`.

Confirmar también que el Monitor exige token y que `/detect` conserva su contrato:

```bash
curl -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/monitor
```

Debe responder `404`. La URL con `?k=<TOKEN>` debe responder `200`.

## 3. Laptop de Joaquín: abrir el Monitor

Abrir en el navegador, sustituyendo `<TOKEN>` por el valor generado:

```text
http://127.0.0.1:8000/monitor?k=<TOKEN>
```

Poner el navegador en pantalla completa. Bini no necesita el token: solo enviará tráfico a `/detect`.

## 4. Laptop de Bini: comprobar la red

Sustituir `<IP-JOAQUIN>` por la IP obtenida ese día:

```bash
curl http://<IP-JOAQUIN>:8000/health/ready
```

No continuar si no responde. Revisar que ambas laptops estén en la misma red, que la IP sea actual y que el AP no aísle clientes.

## 5. Ensayo controlado de transiciones

Desde la raíz de `altur-detect` en la laptop de Bini, ejecutar una llamada a la vez. Esperar a que termine la transición visual antes de correr la siguiente.

```bash
python data/official/check_endpoint.py \
  --url http://<IP-JOAQUIN>:8000/detect \
  --manifest data/manifest.csv \
  --audio-dir data/audio \
  --split train --n 1 --seed 4 --timeout 30
```

```bash
python data/official/check_endpoint.py \
  --url http://<IP-JOAQUIN>:8000/detect \
  --manifest data/manifest.csv \
  --audio-dir data/audio \
  --split train --n 1 --seed 5 --timeout 30
```

```bash
python data/official/check_endpoint.py \
  --url http://<IP-JOAQUIN>:8000/detect \
  --manifest data/manifest.csv \
  --audio-dir data/audio \
  --split train --n 1 --seed 1 --timeout 30
```

Con el estado actual del dataset, los primeros resultados de esos seeds alternan sintética, humana y sintética. Si el dataset cambia, verificarlo antes de usar esta secuencia.

Observar:

- El veredicto cambia una sola vez por llamada y toma el color de la clase.
- La confianza cambia en naranja chorizo y sigue ligada al veredicto.
- La latencia y el chorizo suben o bajan juntos.
- La barra conserva exactamente diez eslabones.
- Un poll sin una llamada nueva no vuelve a disparar animaciones.
- No hay saltos de layout al cambiar el número de dígitos.

## 6. Corrida continua como Altur

En la laptop de Bini:

```bash
python data/official/check_endpoint.py \
  --url http://<IP-JOAQUIN>:8000/detect \
  --manifest data/manifest.csv \
  --audio-dir data/audio \
  --split train \
  --n 10 \
  --seed 1 \
  --timeout 30 \
  --out /tmp/altur-demo-bini.json
```

El cliente debe terminar con 10 llamadas contestadas y 0 errores. El Monitor puede saltar visualmente llamadas intermedias si varias llegan dentro del poll de un segundo; nunca se ralentiza `/detect` para favorecer la animación.

## 7. Prueba de comprensión

Bini mira la pantalla sin explicación previa y responde:

1. ¿Qué determinó en la última llamada?
2. ¿Con qué confianza?
3. ¿Cuánto tardó?
4. ¿Qué viste primero?
5. ¿Qué parte tuviste que descifrar?

No explicar el significado antes de obtener sus primeras respuestas.

## Criterios de éxito

- Las primeras tres respuestas son correctas en menos de cinco segundos.
- Bini no confunde confianza con accuracy.
- El naranja se interpreta como confianza y el rojo exclusivamente como sintética.
- La relación entre latencia y límite de 30 segundos se entiende sin explicación.
- Las transiciones ayudan a localizar el cambio y no distraen.
- La corrida termina con 10/10 respuestas y 0 errores.
- El Monitor no muestra ids completos, audio, labels reales ni el token.

## Si algo falla

### El Monitor responde 404

- Confirmar `ALTUR_MONITOR=1`.
- Confirmar que el token de la URL sea el mismo que recibió el contenedor.
- Reiniciar entre ensayos si cambió una variable: las variables se leen al arrancar.

### Bini no alcanza el endpoint

- Volver a leer la IP de Joaquín.
- Confirmar la misma red.
- Probar `/health` antes de `/detect`.
- Si el AP aísla clientes, cambiar a otra red local o usar cable directo. No usar hotspot para la corrida de latencia.

### El Monitor no anima todas las llamadas

- Verificar si sí aparecen en el historial.
- El polling ocurre cada segundo; una corrida rápida puede incorporar varias llamadas entre actualizaciones.
- Para revisar movimiento usar las tres llamadas individuales del paso 5.
- No añadir pausas al endpoint ni al registro.

## Cierre

Conservar `/tmp/altur-demo-bini.json` para comparar lo que midió la laptop cliente.

En la laptop de Joaquín:

```bash
sg docker -c "docker rm -f altur-mesa"
```

Registrar solamente:

- Qué entendió Bini sin ayuda.
- Qué elemento confundió.
- Qué quitaría.
- Qué conservaría.
- Si las animaciones facilitaron o entorpecieron la lectura.

No desplegar, etiquetar ni hacer push como consecuencia automática del ensayo. Primero revisar los resultados y volver a correr las verificaciones.
