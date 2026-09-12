# altur-detect

**¿Quién llama: una persona o una voz sintética?** `POST /detect` recibe una llamada telefónica
estéreo a 8 kHz y decide si el **caller** (canal 0) es humano o sintético. HackMTY 2026, track
Altur, equipo **Chorizos Circuits**.

```json
{
  "call_id": "call_...",
  "audio_base64": "<WAV completo en base64>",
  "sample_rate": 8000,
  "channels": 2
}
```

La respuesta es `{"is_synthetic": true, "confidence": 0.87}`. El booleano es obligatorio;
`confidence` es opcional para el contrato, pero este servicio siempre la entrega para que el juez
reporte AUC, calibración y desempates.

## Enfoque

**Una señal, bien hecha y auditada.** El detector que se sirve es acústico y deliberadamente simple:

1. **11 features espectrales del canal 0**, calculadas con NumPy sobre ventanas normalizadas a RMS
   unitario, para que el volumen no decida: energía relativa en seis bandas (0–300, 300–1000,
   1000–2000, 2000–3000, 3000–3400 y 3400–4000 Hz), planitud espectral, centroide, rolloff 85 %,
   flujo espectral y dispersión de energía.
2. **Regresión logística estandarizada + calibración Platt**, con umbral fijado por exactitud
   balanceada. Todo se ajusta **dentro** de un cross-fitting anidado sobre `train`; `val` no se
   usa para iterar.
3. **Un bundle versionado** (modelo, calibrador, orden de features, hashes, protocolo) que la API
   carga sin saber qué modelo es. Cambiar de modelo es cambiar de bundle, no de código.

**Resultado (OOF sobre `train`, n = 282 llamadas):** AUC **0.978** [0.961, 0.991], Brier 0.054.

### Por qué funciona, y lo que no afirmamos

El 0.978 hay que leerlo con dos controles que corrimos contra nosotros mismos:

- **Parte grande de la señal es la cadena de producción, no la voz.** El canal del **agente**, que
  es el mismo TTS en las dos clases, separa con AUC 0.64. Y **solo el silencio** separa con 0.97.
  El sintético corta en seco a ~3400 Hz; el humano llega a Nyquist.
- **Con poco audio se equivoca con seguridad.** A 5 s el AUC es 0.40 (invertido) con la confianza
  más alta de la curva. Por eso `/detect` **analiza siempre la llamada completa** y declara 20 s
  como mínimo.
- La conversación **no** está decidiendo: si se revuelven los tiempos de turno dejando el audio
  intacto, cambia el 2 % de los veredictos. Las features conductuales no suman (+0.0015, dentro
  del intervalo), así que el bundle es solo acústico.

No afirmamos que el número transfiera a motores o cadenas telefónicas no vistos: todo el sintético
del dataset sale de un solo proveedor TTS, y el colapso fuera de dominio es el modo de falla
documentado del campo. El set oculto comparte ese proceso de generación, pero con callers y voces
que no aparecen en ningún split.

## Rendimiento

En el contenedor, con 150 s de audio: inferencia secuencial **86 ms**; ráfaga de 16 peticiones
concurrentes con p95 555 ms; arranque a readiness 1.5 s con warm-up incluido. La imagen pesa
~70 MB, corre sin GPU y sin salida a internet, y solo depende de NumPy, FastAPI, uvicorn, pydantic
y python-multipart.

## Equipo y procedencia

| Persona | Aporte | Dónde vive |
|---|---|---|
| Joaquín | Integración: arnés de evaluación, protocolo, controles de confound, bundle, `/detect`, contenedor y failover | `src/altur/`, `scripts/`, `docs/` |
| Biniza | Modelo: auditoría forense de canal, rama prosódica (Shimmer CS3), rama espectral de referencia (LFCC), fusión tardía, calibración de referencia y stress test | `research/spectral_factory/` |
| Ricardo | Producto e investigación; set de robustez con seis condiciones (ruido, pitch, tempo, pasa-bajas, Opus) | Perturbaciones v2 |
| Regina | UX y storytelling | Pitch |

El trabajo de Biniza se hizo en un repositorio aparte y se incorporó aquí **sin** los archivos por
llamada que contenían identificadores del dataset (`research/spectral_factory/README.md`).

## Correr

```bash
make setup && make test          # Python 3.12
make serve                       # /detect en localhost:8000
make docker && make docker-run   # la imagen de producción
URL=http://127.0.0.1:8000 make smoke
```

El contrato oficial de `POST /detect` es una llamada por petición con `call_id`, `audio_base64`,
`sample_rate: 8000` y `channels: 2`; `audio_base64` contiene los bytes exactos del WAV completo.
El servicio también tolera WAV crudo y multipart, pero el camino oficial es el JSON anterior.
`confidence` es la probabilidad calibrada **de la clase reportada**. El juez permite 30 segundos
por llamada, las llamadas duran de 1 a 4 minutos y el body llega a unos 5 MB. Un timeout, status
distinto de 200 o respuesta sin `is_synthetic` booleano cuenta como error. La métrica principal es
balanced accuracy sobre callers y voces no vistos. También hay `GET /health`,
`GET /health/ready` y `GET /version`.

## Datos y privacidad

El audio, `manifest.csv`, los identificadores de llamada, los folds y las predicciones por llamada
**no se redistribuyen**, por los términos del dataset. Solo se publican agregados.
`make data` descarga el dataset oficial y verifica su SHA-256; `make protocol` regenera grupos y
folds de forma determinista.

Más detalle: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) ·
[`docs/DATA_PROTOCOL_CARD.md`](docs/DATA_PROTOCOL_CARD.md) ·
[`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md).
