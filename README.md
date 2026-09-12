# altur-detect

Servicio CPU para detectar si la voz del caller en una llamada telefonica es sintetica.
Proyecto del track Altur de HackMTY 2026.

`POST /detect` recibe WAV estereo a 8 kHz, con el caller en el canal 0 y el agente en el canal 1.

```json
{"is_synthetic": true, "confidence": 0.87}
```

## Estado

La API y el formato de bundle estan implementados. El candidato entrenado y sus cifras publicas
siguen pendientes de congelarse; este README no inventa resultados provisionales.

## Instalacion

Requiere Python 3.12 o posterior.

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m pytest
```

Para iniciar el servicio:

```bash
./.venv/bin/uvicorn altur.api:app --host 0.0.0.0 --port 8000 --workers 2
```

Tambien se puede usar `make setup`, `make test`, `make serve` y `make smoke`.

## Contrato HTTP

### `POST /detect`

Formatos aceptados:

- WAV crudo con `Content-Type: audio/wav`;
- JSON base64, por ejemplo `{"audio":"<BASE64_WAV>"}`;
- multipart con un campo de audio.

Respuesta exitosa:

```json
{"is_synthetic": false, "confidence": 0.61}
```

`is_synthetic` clasifica solo el canal 0. `confidence` esta acotada a `[0,1]`. Entradas invalidas
devuelven HTTP 400 con `error` y `detail`, sin stack trace. El nombre exacto del campo oficial,
timeout, concurrencia y tamanio maximo del benchmark permanecen por confirmar.

| Endpoint | Funcion |
|---|---|
| `POST /detect` | Prediccion |
| `GET /health` | Liveness del proceso |
| `GET /health/ready` | Readiness del detector |
| `GET /version` | Commit, bundle, protocolo y limitaciones sin secretos |

## Arquitectura

La API se mantiene independiente del modelo. Un bundle versionado declara schema, orden de
features, protocolo, umbral, limitaciones y SHA-256 de sus artefactos. Cambiar de candidato
significa cambiar `ALTUR_BUNDLE_DIR`, no reescribir `/detect`.

El arnes separa audio y metadatos: un extractor recibe `AudioExample`, nunca id, etiqueta, split
o procedencia. La seleccion se realiza con predicciones out-of-fold sobre `train`; `val` no se usa
para iterar.

Detalles: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Reproducibilidad

- El dataset se descarga y verifica por SHA-256 con `make data`.
- Grupos y folds se regeneran de forma determinista con `make protocol` y no se publican porque
  contienen identificadores.
- Cada bundle registra commit, protocolo, orden de features y hashes.
- Toda metrica publicable debe declarar `n`, unidad, split, protocolo, seed, commit y digest del
  codigo efectivo.
- Los tests usan audio sintetico generado; no incluyen llamadas reales.

La model card queda en [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) y el protocolo de datos en
[`docs/DATA_PROTOCOL_CARD.md`](docs/DATA_PROTOCOL_CARD.md).

## Privacidad

El audio, `manifest.csv`, IDs, caches, folds y predicciones por llamada no se redistribuyen. El
servicio no necesita enviar audio a terceros y la imagen objetivo funciona sin egress. Solo se
publican agregados que no permiten reconstruir una llamada.

## Limitaciones

- No podemos afirmar generalizacion a hablante no visto sobre el dataset oficial. El manifest no
  trae llave de union; grupo = llamada es un fallback conservador.
- La prevalencia cambia entre splits: `train` es 59.9 % sintetico y `val` 47.9 %. El prior del set
  oculto es desconocido y la confianza puede trasladarse mal.
- Las clases tienen cadenas de produccion distintas: el sintetico corta alrededor de 3400 Hz y el
  humano llega a Nyquist. Un AUC alto puede medir la cadena y no la voz. La causa sigue pendiente
  de confirmar con Altur.
- El dataset oficial no demuestra generalizacion a otro proveedor TTS o cadena telefonica.
- Las features conductuales son proxies y pueden fallar con agentes rapidos, humanos lentos o
  grabaciones humanas reproducidas.

Proveniencia de prevalencia: `n=282` train y `n=71` val; unidad = llamada; protocolo = conteo del
manifest v1.0; seed = no aplica; commit = `a5979a4`; digest efectivo = no aplica al conteo textual.
Proveniencia de banda: `n=282`, unidad = llamada, split = train, protocolo = `official_v1`, commit
= `a5979a4`; seed de submuestra y digest efectivo = `UNK`, por lo que no se cita AUC. El alcance
completo esta en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Equipo y procedencia

Este repositorio es el entregable del equipo **Chorizos Circuits** (HackMTY 2026, track Altur) y
reúne el trabajo de todos, no solo el del arnés.

| Persona | Aporte | Dónde vive |
|---|---|---|
| Joaquín | Integración: arnés de evaluación, protocolo, controles de confound, bundle, `/detect`, contenedor y failover | `src/altur/`, `scripts/`, `docs/` |
| Biniza | Modelo: auditoría forense de canal, rama prosódica (Shimmer CS3), rama espectral de referencia (LFCC), fusión tardía, calibración de referencia y stress test de robustez | `research/spectral_factory/` |
| Ricardo | Producto e investigación; set de robustez con seis condiciones (ruido, pitch, tempo, pasa-bajas, Opus) | Condiciones portadas como perturbaciones v2 |
| Regina | UX y storytelling | Pitch |

El trabajo de Biniza se desarrolló en un repositorio aparte y se incorporó aquí sin los archivos
por llamada que contenían identificadores del dataset. Los detalles están en
`research/spectral_factory/README.md`.

## Licencia

Pendiente de decision. [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md) separa dependencias de
inferencia, entrenamiento, investigacion y desarrollo sin elegir una licencia por Joaquin.
