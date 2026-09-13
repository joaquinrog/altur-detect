# Guion de demo — altur-detect (C2)

Turno frente a Altur. Todo lo que aparece aquí está **medido**, no estimado; la fecha de cada
medición va junto al número. Si algo no se midió, dice que no se midió.

- **Endpoint:** `http://64.177.80.133:8000` · `POST /detect`
- **Bundle servido:** `spectral_factory_lfcc_logreg@1` (120 LFCC del caller, umbral 0.5351)
- **Fallback:** la laptop sirviendo el mismo bundle en la LAN
- **Duración objetivo:** 4 min de guion + preguntas

---

## 0. Checklist previo (correr 10 min antes del turno)

Marcar cada línea. Si una falla, se aplica el plan de recuperación de la §7 **antes** de empezar.

```bash
cd ~/Desktop/altur-detect

# 1. URL viva y readiness
curl -sS -m 10 http://64.177.80.133:8000/health
curl -sS -m 10 http://64.177.80.133:8000/health/ready

# 2. El bundle servido es C2 y coincide con models/current
curl -sS -m 10 http://64.177.80.133:8000/version | python3 -m json.tool | head -20
./.venv/bin/python -c "import json;m=json.load(open('models/current/manifest.json'));print(m['detector_name'],m['run_id'],m['threshold'])"

# 3. Una llamada real, cronometrada
./.venv/bin/python data/official/check_endpoint.py \
  --url http://64.177.80.133:8000/detect --split train --n 3 \
  --manifest data/manifest.csv --audio-dir data/audio

# 4. El fallback local arranca y da el mismo veredicto
ALTUR_BUNDLE_DIR=models/current ./.venv/bin/uvicorn altur.api:app --host 0.0.0.0 --port 8000 --workers 2 &
curl -sS -m 5 http://127.0.0.1:8000/health/ready      # debe decir spectral_factory_lfcc_logreg@1
ip -4 addr show | grep -oP 'inet \K[\d.]+' | grep -v 127.0.0.1   # IP de la laptop en la LAN
```

| # | Comprobación | Criterio |
|---|---|---|
| 1 | `/health` y `/health/ready` | HTTP 200, `status=ready`, detector `spectral_factory_lfcc_logreg@1` |
| 2 | `/version` vs `models/current` | mismo `run_id` y mismo `threshold` |
| 3 | Audio de prueba | 3/3 respondidas, 0 errores, todas muy por debajo de 30 s |
| 4 | Fallback local | readiness en ~2 s y detector **no** `constant@1` |
| 5 | Conexión | hotspot listo como segunda red; SSH al servidor **no** funciona desde la WiFi del Tec |
| 6 | Tiempo | reloj a la vista; el guion son 4 min |
| 7 | Batería | laptop a corriente, suspensión desactivada |

🔴 **Si `/health/ready` responde `constant@1`, parar.** Significa que el proceso arrancó sin
`ALTUR_BUNDLE_DIR` y está contestando una constante que se lee como un modelo. No se demuestra
nada con eso.

---

## 1. Apertura (20 s)

> "Somos Chorizos Circuits. Contestamos una sola pregunta: cuando entra una llamada, ¿del otro
> lado hay una persona o una voz sintética? Tenemos un endpoint vivo ahora mismo; se lo pueden
> llamar ustedes desde su máquina."

Dar la URL en voz alta y escrita: `http://64.177.80.133:8000/detect`.

## 2. El problema (30 s)

> "El audio llega como el peor caso posible: telefonía a 8 kHz, mu-law, estéreo, entre 1 y 4
> minutos. La voz sintética de hoy no se delata por prosodia obvia; se delata por la cadena que la
> produjo. Nuestro trabajo fue encontrar esa señal y, sobre todo, medir cuánto de nuestro propio
> número es real."

## 3. Arquitectura C2 (50 s)

> "Una señal, bien hecha y auditada."

1. **Canal 0, el que llama.** El canal 1 es la agente de Altur, el mismo TTS en las dos clases: no
   lo usamos para decidir.
2. **120 features LFCC** del caller: 60 coeficientes —20 estáticos, 20 Δ y 20 ΔΔ— con su media y
   su desviación, calculadas en NumPy sobre el segmentado del freeze de Bini.
3. **Regresión logística estandarizada + calibración Platt**, umbral fijado por exactitud
   balanceada. Todo el ajuste ocurre dentro de un cross-fitting sobre `train`.
4. **Bundle versionado** — modelo, calibrador, orden de features, hashes, protocolo. La API carga el
   bundle sin saber qué modelo es; cambiar de modelo es cambiar de bundle, no de código. Si un byte
   del bundle no cuadra, `/health/ready` devuelve 503 en vez de predecir.

> "Y lo que **no** afirmamos: corrimos controles contra nosotros mismos y los dos salieron
> positivos. Solo el silencio del caller separa con AUC 0.9994, y el canal de la agente con 0.74.
> Buena parte de lo que C2 está midiendo es cadena de grabación, no voz. Está escrito en
> `/version`, en el campo `limitations`, y el juez lo puede leer sin preguntarnos."

## 4. La llamada real (60 s)

Con el cliente **oficial** de Altur, sin capa nuestra en medio:

```bash
./.venv/bin/python data/official/check_endpoint.py \
  --url http://64.177.80.133:8000/detect --split train --n 5 \
  --manifest data/manifest.csv --audio-dir data/audio
```

Y una sola llamada, la más pesada que tenemos, para enseñar el presupuesto de 30 s:

```bash
curl -sS -m 30 -o /dev/null -D - -X POST \
  -H 'Content-Type: audio/wav' --data-binary @data/audio/call_bd2262567810.wav \
  -w 'http=%{http_code} total=%{time_total}s\n' \
  http://64.177.80.133:8000/detect
```

**Medido el 2026-09-13 contra el contenedor desplegado**, cuerpo JSON de **11.69 MB** (más del
doble de los ~5 MB que describe el reto): cinco corridas entre **1.66 s y 3.55 s** de extremo a
extremo, con **59.5–62.3 ms** de inferencia en el servidor. La inferencia varía 3 ms y el total
varía 2 s sobre el mismo cuerpo: **lo que se mide es la subida, no el modelo.** Margen contra el
límite de 30 s: **~9×** en el peor caso medido.

En total, hoy: **78 llamadas al endpoint desplegado, 0 errores, ninguna por encima de 3.4 s**
(`val` n=20 y n=20, `train` n=30, y las cinco del cuerpo grande).

## 5. Cómo se lee la respuesta (40 s)

```json
{"is_synthetic": false, "confidence": 0.999889}
```

> "`is_synthetic` es el veredicto y es lo único obligatorio del contrato. `confidence` es la
> probabilidad calibrada **de la clase que reportamos**, no `p(sintética)`: si decimos humano con
> 0.96, es 96 % de confianza en *humano*. La mandamos siempre, en todas las respuestas, para que
> puedan calcular AUC y calibración y usarla de desempate."

🔴 **No decir que `confidence` es accuracy.** Y si alguien pregunta por la confianza alta: el
detector **no tiene un "no sé"** — sobre entradas que no son voz (ruido blanco, silencio, un tono
de 440 Hz) contesta `is_synthetic=false` con confianza de 0.9997 a 1.0. Es una logística sobre
features estandarizadas; fuera de distribución la sigmoide satura. Lo sabemos porque lo medimos.

## 6. Resiliencia operativa (40 s)

- **Errores:** base64 inválido devuelve **HTTP 400** con `{"error":"bad_base64"}` — sin stack trace
  y sin inventar un veredicto. *(verificado 2026-09-13)*
- **Bundle corrupto:** un byte cambiado deja `/health` en 200 (`detector_loaded=false`),
  `/health/ready` en **503** con la razón, y `/detect` en 503. El proceso queda vivo a propósito:
  un proceso muerto no diagnostica. Nunca se cae en una predicción constante por accidente.
- **Proceso caído:** `--restart unless-stopped`; readiness vuelve en 0.31 s de aplicación más
  1.52 s de arranque de contenedor *(medido en A4)*.
- **Recursos:** imagen de ~70 MB, sin GPU, sin salida a internet en el camino de inferencia
  (se corrió `predict()` con `socket.socket` parcheado para lanzar excepción, y pasa).
- **Fallback:** la misma imagen y el mismo bundle en la laptop, servida por LAN.

## 7. Fallback si falla la red (no se finge nada)

**Regla: si el endpoint remoto no contesta, se dice en voz alta y se cambia de URL.** No se
presenta una corrida local como si hubiera salido del servidor.

> "El servidor no está contestando desde esta red. Cambio a la copia local, mismo bundle,
> mismo `run_id`. Es la misma respuesta; lo que ya no estoy demostrando es la red."

```bash
# En la laptop que sirve
ALTUR_BUNDLE_DIR=models/current ./.venv/bin/uvicorn altur.api:app --host 0.0.0.0 --port 8000 --workers 2
curl -sS http://127.0.0.1:8000/health/ready     # spectral_factory_lfcc_logreg@1, no constant@1
ip -4 addr show | grep -oP 'inet \K[\d.]+' | grep -v 127.0.0.1

# Desde la máquina cliente, con la IP de arriba
./.venv/bin/python data/official/check_endpoint.py --url http://<IP_LAN>:8000/detect \
  --split train --n 5 --manifest data/manifest.csv --audio-dir data/audio
```

**Verificado el 2026-09-13:** el fallback local da los **mismos 20/20 veredictos** que el servidor
sobre las mismas llamadas, con readiness en 2 s. La WiFi del Tec **no** aísla clientes, así que el
camino por LAN existe sin cable; el cable queda como plan C.

Orden de recuperación: **(1)** remoto por WiFi del Tec → **(2)** remoto por hotspot → **(3)** laptop
por LAN → **(4)** laptop por cable. Nunca `ALTUR_EMERGENCY_CONSTANT=1`: eso es una constante, no un
detector, y en una demo sería mentir.

## 8. Cierre (20 s)

> "Un endpoint vivo que cumple el contrato, con 15× de margen contra los 30 segundos y con un
> cuerpo del doble del tamaño que esperan. Un bundle versionado que se niega a predecir si no puede
> verificarse. Y una lista de limitaciones que publicamos nosotros mismos en `/version`, porque el
> número que importa no es el de nuestro `val` — es el del set oculto, con voces que nunca vimos."

---

## Qué NO decir

- **No** presentar la balanced accuracy de `val` como validación honesta. `val` se usó para elegir
  entre tres candidatos y ya se había mirado cuatro veces (D-A7.3): la cifra es optimista.
- **No** presentar el bundle train+val (`spectral_factory_lfcc_trainval_v1`): al entrenar con `val`
  se acabó el holdout y sus números son tautológicos. No está promovido y no se sirve.
- **No** mencionar C3 como disponible. No está promovido, no se sirve y no entra al guion.
- **No** describir C2 como late fusion LFCC + YIN. Lo servido es solo la rama LFCC.
- **No** comparar los milisegundos de inferencia del servidor con el tiempo de una llamada enviada
  por WiFi: son números de cosas distintas.
