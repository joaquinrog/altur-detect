# Data and protocol card

## Dataset

**FACT:** el release oficial contiene 353 llamadas en espanol mexicano. La unidad disponible es
la llamada, con WAV estereo a 8 kHz: ch0 caller y ch1 agente. El manifest separa `train` y `val`.
Proveniencia del conteo: `n=353`; unidad = llamada; split = train + val oficial; protocolo =
conteo del manifest v1.0; seed = no aplica; commit = `a5979a4`; digest efectivo = no aplica a un
conteo textual del manifest.

**FACT:** el dataset no se redistribuye. Este repositorio no incluye audio, `manifest.csv`,
identificadores, turns, grupos ni asignaciones de fold. No se intenta identificar a participantes.

## Etiqueta

La clase positiva es `synthetic = 1`. La etiqueta responde si la voz del caller es sintetica; no
es una etiqueta de replay, liveness, fraude, identidad ni comportamiento conversacional.

## Separacion de metadatos

`DatasetRecord` conserva metadatos para el arnes. `AudioExample` contiene solo audio y es el unico
objeto que recibe un extractor. Este limite reduce el riesgo de aprender IDs, split o procedencia.

## Protocolo de desarrollo

- Iteracion exclusivamente sobre `train` mediante predicciones out-of-fold.
- Una prediccion y una unidad independiente disponible por llamada.
- Ajuste de preprocesamiento, modelo, fusion, calibrador y umbral dentro de folds.
- Derivados y augmentaciones permanecen en el grupo de su llamada de origen.
- Controles obligatorios: canal 1, silence-only, `highpass_300`, `lowpass_3400` y codec.
- `val` no se carga para seleccionar features, modelos, hiperparametros ni umbrales.
- Cada resultado debe registrar `n`, unidad, split, protocolo, seed, commit y digest efectivo.

**TODO(A3):** completar numero de folds externos/internos, semillas, IDs versionados de protocolo y
digests desde los artefactos congelados. No inferirlos desde una corrida local.

## Agrupacion y generalizacion

### D-A1.2

**OBS:** el manifest oficial solo trae `anon_id`, etiqueta, split y duracion; no trae
`speaker_or_voice_id`, llamada donante ni otra llave de union. El protocolo usa una llamada por
grupo como fallback conservador y une derivados mediante `derived_from` cuando existe.

**Limitacion:** **no podemos afirmar generalizacion a hablante no visto sobre el dataset oficial**.
La agrupacion por llamada evita mezclar segmentos de una misma llamada, pero no demuestra que dos
llamadas pertenezcan a personas o voces diferentes.

## Prevalencia

### D-A1.5

**OBS:** `train` tiene 169/282 llamadas sinteticas (59.9 %) y `val` 34/71 (47.9 %). La
prevalencia no es estable entre splits; la del set oculto es **UNK**.

Metadatos: `n=282` train y `n=71` val; unidad = llamada; split = oficial; protocolo = conteo del
manifest v1.0; seed = no aplica; commit = `a5979a4`; digest efectivo = no aplica a conteo textual.

Consecuencia **INF:** reportar calibracion y umbral junto con el prior de ajuste; no prometer que
la confianza permanece calibrada bajo otro prior o cambio de dominio.

## Riesgo de cadena de produccion

### D-A1.6

**OBS:** en `train`, la clase sintetica corta alrededor de 3400 Hz y la humana conserva energia
hasta Nyquist. Las dos clases tienen cadenas de produccion distintas.

Metadatos disponibles: `n=282`; unidad = llamada; split = `train`; protocolo = `official_v1`;
seed de la submuestra espectral = **UNK**; commit = `a5979a4`; digest efectivo = **UNK**. No se
cita aqui el AUC heredado porque su registro no satisface aun la proveniencia publica exigida.

Consecuencia **INF:** un AUC alto puede medir codec, filtrado o captura en vez de voz. La causa
queda pendiente de la pregunta 4 del booth sobre TTS, `output_format` y cadena telefonica. Hasta
resolverla, todo claim de deteccion vocal debe sobrevivir controles de canal y banda.

## Privacidad, retencion y publicacion

- Audio y manifest permanecen en almacenamiento privado e ignorado por Git.
- Caches, predicciones por llamada, grupos y folds tampoco se publican.
- Solo se permiten tablas y figuras agregadas sin IDs ni espectrogramas individuales.
- No se redistribuyen muestras ni fragmentos de audio.
- **TODO(integrador):** documentar retencion y borrado operativo antes de cualquier captura nueva.

## Limitaciones adicionales

- El dataset oficial no demuestra generalizacion a otro proveedor TTS o cadena telefonica.
- Segmentos de una llamada no son observaciones independientes.
- El numero real de hablantes y voces es **UNK**.
- El prior y la composicion del set oculto son **UNK**.
- **TODO(A3):** agregar limitaciones especificas del candidato sin usar `val`.
