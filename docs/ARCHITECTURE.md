# Arquitectura

## Objetivo y estado

**FACT:** `altur-detect` clasifica si la voz del canal 0 de una llamada es humana o sintetica.
El canal 1 corresponde al agente. La entrada objetivo es WAV estereo a 8 kHz y la salida publica
es `is_synthetic` mas `confidence`.

**FACT:** la API, decodificacion, bundles y detector constante existen en el commit base de este
borrador. **TODO(A3):** registrar aqui el candidato entrenado, ramas efectivas, calibrador y regla
de fusion cuando A3 los congele. Este documento no presupone un ganador.

## Camino de inferencia

```text
cliente
  |
  | WAV crudo, multipart o JSON base64
  v
FastAPI /detect
  |
  | limite de bytes + decodificacion WAV estricta
  v
AudioExample (audio solamente; sin id, etiqueta, split ni procedencia)
  |
  | detector cargado desde un bundle verificado
  v
Prediction
  |
  v
{"is_synthetic": bool, "confidence": float}
```

**FACT:** `src/altur/io.py` usa `wave` de la biblioteca estandar. Mono nunca se duplica para
fabricar el canal 1; si se permite, se procesa explicitamente como modo degradado ch0-only.

**FACT:** `src/altur/api.py` no cambia al cambiar de modelo. `ALTUR_BUNDLE_DIR` selecciona un
bundle. La respuesta oficial conserva dos campos salvo que `ALTUR_RESPONSE_EXTRAS=1` se active
despues de confirmar que el consumidor acepta extras.

## Contrato del bundle

Un bundle es un directorio versionado, no un pickle suelto. Su manifest declara al menos:

- version del schema;
- commit y estado limpio/sucio;
- identificadores de protocolo y corrida;
- extractores y orden exacto de features;
- umbral y politica de confianza;
- SHA-256 de cada archivo;
- limitaciones conocidas.

La carga rechaza archivos ausentes, alterados o no declarados. **TODO(A4.1):** documentar el
formato del bundle entrenado, su digest agregado y la exportacion NumPy final. **TODO(A4.2):**
confirmar que un bundle solicitado pero invalido deja readiness en 503 y nunca activa una
prediccion constante silenciosa.

## Camino experimental

```text
manifest privado -> DatasetRecord -> carga de AudioExample
                                      |
                                      v
                            extractores registrados
                                      |
                                      v
                              cache por contenido
                                      |
                                      v
                         folds externos agrupados (OOF)
                                      |
                        +-------------+-------------+
                        |                           |
                    rama acustica          rama conductual
                        |                           |
                        +-------------+-------------+
                                      |
                         fusion + calibracion anidadas
                                      |
                                      v
                           una prediccion por llamada
```

**FACT:** `AudioExample` no contiene identificadores, etiqueta, split ni procedencia. Esos datos
viven en `DatasetRecord` y no se entregan a un extractor. Esto reduce fuga por interfaz, pero no
la hace fisicamente imposible en Python.

**FACT:** las decisiones se toman con predicciones out-of-fold sobre `train`. `val` no se usa
para iterar. Ajuste de modelo, fusion, calibracion y umbral deben ocurrir dentro de los folds
correspondientes.

**TODO(A3):** completar el diagrama con los nombres versionados de modelos, fusion y calibracion;
documentar semillas, digests efectivos y condiciones de gauntlet desde sus artefactos.

## Controles contra atajos

- Canal 1: control para detectar senales de cadena de produccion que no pertenecen al caller.
- Regiones sin habla: control para detectar clasificacion por ruido, codec o grabacion.
- `highpass_300` y `lowpass_3400`: diagnosticos obligatorios del confound de banda.
- Perturbaciones de codec: comprueban que una mejora limpia no dependa de una codificacion.
- Una prediccion por llamada: segmentar no aumenta el numero de unidades independientes.

## Despliegue

La imagen objetivo es CPU-only, corre como usuario sin privilegios y contiene el bundle. No
necesita descargar modelos ni enviar audio fuera del proceso. Liveness (`/health`) y readiness
(`/health/ready`) son distintas; `/version` expone trazabilidad sin secretos ni rutas locales.

**TODO(A4.3):** adjuntar evidencia por digest de imagen para inferencia sin egress, inspeccion de
material prohibido, tiempo de arranque, memoria y latencia. La simulacion local y el ensayo fisico
de failover se describen en `RUNBOOK_FAILOVER.md`.

## Privacidad

El audio, `manifest.csv`, grupos, folds por llamada, caches y artefactos por llamada permanecen
fuera del repositorio publico. No se redistribuyen datos ni se intenta identificar participantes.
Solo se pueden publicar resultados agregados que no permitan reconstruir una llamada.

## Limitaciones obligatorias

### D-A1.2: no hay prueba de hablante no visto

**FACT:** el manifest oficial no trae llave de union de hablante o voz. **INF:** usar una llamada
por grupo es el fallback conservador disponible, pero no permite demostrar independencia por
hablante. Por tanto, **no podemos afirmar generalizacion a hablante no visto sobre el dataset
oficial**. Se declara de forma explicita.

### D-A1.5: la prevalencia cambia

**OBS:** `train` contiene 169/282 llamadas sinteticas (59.9 %) y `val` 34/71 (47.9 %). La
prevalencia no es estable entre splits y la del set oculto es **UNK**.

Proveniencia de la cifra: `n=282` y `n=71`; unidad independiente disponible = llamada; splits =
`train` y `val`; protocolo = conteo del manifest oficial v1.0; seed = no aplica; commit de
protocolo = `a5979a4`; digest de codigo efectivo = no aplica a un conteo textual. **INF:** una
calibracion o umbral que asuma el prior de `train` puede trasladarse mal.

### D-A1.6: las clases tienen cadenas de produccion distintas

**OBS:** en `train`, la clase sintetica presenta un corte abrupto alrededor de 3400 Hz mientras
la humana conserva energia hasta Nyquist. **INF:** un AUC alto puede estar midiendo la cadena de
produccion y no la voz.

Proveniencia disponible: `n=282` llamadas para los descriptores de D-A1.6; unidad independiente =
llamada; split = `train`; protocolo = `official_v1`; seed = **UNK** para la submuestra espectral;
commit = `a5979a4`; digest de codigo efectivo = **UNK** en el registro heredado. Por esa carencia,
este borrador no cita el AUC ni eleva el corte a metrica reproducible. **TODO(A3/integrador):**
enlazar el artefacto con seed y digest antes de cualquier claim cuantitativo publico.

La interpretacion sigue pendiente de la pregunta 4 del booth: que TTS, `output_format` y cadena
telefonica produjeron el caller sintetico. Hasta resolverla, no se atribuye la separacion a
fisiologia vocal ni a capacidad cross-vendor.
