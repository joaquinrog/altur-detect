# Especificacion de portado: spectral factory

## Alcance y snapshot

Esta especificacion describe como portar dos familias de features desde
`binivazqua/chorizos-circuits-spectral-factory` al arnes de `altur-detect`. No autoriza copiar
datos, scores, folds, calibradores ni resultados del repo fuente.

**FACT:** la fuente se leyo con `gh api`, sin clonar dentro de este repo, al commit
`9583e3a6c9677d3df920637868416131ba757d10`, fechado `2026-09-12T16:29:30Z`. El tree del commit es
`843808e4a0abbc3f04b907083c11d5eab2f569cf`. Todas las rutas y lineas citadas corresponden a ese
snapshot.

**FACT:** no se leyo ni se reproduce `persona3_prosodic/data/splits.csv`. El unico identificador
que aparecio en un artefacto de latencia no se copia aqui.

## Veredicto rapido

| Familia | Semantica | Dependencias actuales | `product_safe` | Tipo de port |
|---|---|---|---|---|
| LFCC 120-dim | FACT: completamente especificada | NumPy + librosa + SciPy | **True por licencia** | **Mecanico en semantica, pero requiere adaptar/reimplementar infraestructura** para `AudioExample`, segmentacion no-oracle e imagen minima |
| Shimmer CS3 12-dim | FACT: especificada salvo defaults internos de Praat | NumPy + Parselmouth/Praat | **False** | **Mecanico solo como extractor de investigacion; no mecanico para producto** porque conservar la semantica sin GPL exige reimplementar y revalidar F0/pulsos |

## Contrato comun de entrada

**FACT:** ambos builders leen WAV, toman solo canal 0 y cargan turns por archivo antes de llamar al
extractor (`persona2_reference_spectral/build_spectral_dataset.py:L31-L41` y
`persona3_prosodic/build_prosodic_dataset.py:L27-L39`).

**FACT:** en nuestro arnes un extractor solo recibe `AudioExample`; `anon_id`, etiqueta, split y
grupo no existen en ese objeto (`src/altur/types.py:L72-L86`). La segmentacion permitida llega en
`ex.seg`, y `oracle@1` es solo de entrenamiento (`src/altur/types.py:L42-L57`).

Requisitos comunes del port:

1. Recibir exclusivamente `ex: AudioExample`.
2. Leer exclusivamente `ex.ch0`, `ex.sr` y `ex.seg`.
3. Declarar `channels=(0,)` y `needs_seg=True`.
4. No aceptar ni devolver `anon_id`.
5. No abrir manifest, turns, audio, splits ni ningun path global desde el extractor.
6. No usar segmentacion `oracle@1` en un bundle productivo.
7. Devolver siempre `(features, diagnostics)` JSON-finito, con schema y orden fijos.
8. Prefijar toda feature con el nombre registrado; el runner lo exige
   (`src/altur/runner.py:L150-L158`, `src/altur/runner.py:L275-L279`).

## Familia 1: LFCC 120-dim

### Definicion exacta

**FACT:** la representacion por frame tiene 60 dimensiones: 20 estaticas, 20 deltas y 20
delta-deltas (`persona2_reference_spectral/lfcc_features.py:L1-L7`, `L24-L30`). El pooling produce
media y desviacion estandar para cada dimension, en ese orden (`lfcc_features.py:L114-L130`).

### Orden exacto de columnas del builder

**FACT:** `build_spectral_dataset.py` crea primero `anon_id,label,fold`, luego ejecuta
`row.update(feats)` (`persona2_reference_spectral/build_spectral_dataset.py:L76-L84`). Esas tres
columnas son metadata y no pertenecen al extractor.

**FACT:** `_lfcc_dim_names()` construye `static0..19`, despues `delta0..19` y despues
`deltadelta0..19` (`lfcc_features.py:L133-L137`). `pool_lfcc()` inserta `_mean` y luego `_std` para
cada nombre (`lfcc_features.py:L123-L130`). Por tanto, las 120 columnas de feature salen
exactamente en este orden:

```text
static0_mean, static0_std, static1_mean, static1_std,
static2_mean, static2_std, static3_mean, static3_std,
static4_mean, static4_std, static5_mean, static5_std,
static6_mean, static6_std, static7_mean, static7_std,
static8_mean, static8_std, static9_mean, static9_std,
static10_mean, static10_std, static11_mean, static11_std,
static12_mean, static12_std, static13_mean, static13_std,
static14_mean, static14_std, static15_mean, static15_std,
static16_mean, static16_std, static17_mean, static17_std,
static18_mean, static18_std, static19_mean, static19_std,
delta0_mean, delta0_std, delta1_mean, delta1_std,
delta2_mean, delta2_std, delta3_mean, delta3_std,
delta4_mean, delta4_std, delta5_mean, delta5_std,
delta6_mean, delta6_std, delta7_mean, delta7_std,
delta8_mean, delta8_std, delta9_mean, delta9_std,
delta10_mean, delta10_std, delta11_mean, delta11_std,
delta12_mean, delta12_std, delta13_mean, delta13_std,
delta14_mean, delta14_std, delta15_mean, delta15_std,
delta16_mean, delta16_std, delta17_mean, delta17_std,
delta18_mean, delta18_std, delta19_mean, delta19_std,
deltadelta0_mean, deltadelta0_std, deltadelta1_mean, deltadelta1_std,
deltadelta2_mean, deltadelta2_std, deltadelta3_mean, deltadelta3_std,
deltadelta4_mean, deltadelta4_std, deltadelta5_mean, deltadelta5_std,
deltadelta6_mean, deltadelta6_std, deltadelta7_mean, deltadelta7_std,
deltadelta8_mean, deltadelta8_std, deltadelta9_mean, deltadelta9_std,
deltadelta10_mean, deltadelta10_std, deltadelta11_mean, deltadelta11_std,
deltadelta12_mean, deltadelta12_std, deltadelta13_mean, deltadelta13_std,
deltadelta14_mean, deltadelta14_std, deltadelta15_mean, deltadelta15_std,
deltadelta16_mean, deltadelta16_std, deltadelta17_mean, deltadelta17_std,
deltadelta18_mean, deltadelta18_std, deltadelta19_mean, deltadelta19_std
```

En nuestro arnes cada nombre gana el prefijo `spectral_factory.lfcc.`, sin cambiar el orden.

### Reproducibilidad del orden

**FACT:** el orden de columnas de features no depende de un glob. Depende del orden de insercion de
dos diccionarios construido por loops deterministas (`lfcc_features.py:L123-L137`). Python conserva
ese orden.

**OBS:** el orden de filas del CSV si es no determinista: `Pool.imap_unordered()` entrega resultados
en orden de finalizacion y el builder los agrega en ese orden
(`build_spectral_dataset.py:L61-L68`, `L76-L84`). Esto no cambia el schema, pero es un bug de
reproducibilidad byte-a-byte. El port no escribe CSV y no debe heredar ese comportamiento.

### Parametros numericos y operaciones

| Parametro | Valor literal | Fuente |
|---|---:|---|
| sample rate esperado | 8000 Hz | `lfcc_features.py:L24` |
| ventana | 20 ms = 160 muestras a 8 kHz | `lfcc_features.py:L25`, `L40-L43` |
| hop | 10 ms = 80 muestras a 8 kHz | `lfcc_features.py:L26`, `L40-L43` |
| `n_fft` | `max(256, frame_length)`; a 8 kHz = 256 | `lfcc_features.py:L27`, `L70-L74` |
| ventana de STFT | Hamming | `lfcc_features.py:L73-L75` |
| centrado/padding de STFT | no se pasan; defaults efectivos dependen de la version de librosa, que es **UNK** | `lfcc_features.py:L73-L74` |
| filtros lineales | 20 triangulos | `lfcc_features.py:L28`, `L46-L64` |
| rango de filtros | 0 Hz a `sr/2`; a 8 kHz = 0-4000 Hz | `lfcc_features.py:L51-L53` |
| espectro | potencia `abs(STFT)^2` | `lfcc_features.py:L73-L78` |
| piso antes de log | `1e-10` | `lfcc_features.py:L79` |
| DCT | tipo II, `norm="ortho"`, primeros 20 | `lfcc_features.py:L81-L82` |
| delta | orden 1, width inicial 9 | `lfcc_features.py:L30`, `L94-L101` |
| delta-delta | orden 2, mismo width | `lfcc_features.py:L94-L103` |
| modo/borde de delta | no se pasa; default efectivo de librosa y version son **UNK** | `lfcc_features.py:L99-L101` |
| llamada con menos de 3 frames | devuelve `None` | `lfcc_features.py:L85-L97` |
| target RMS | -20 dBFS = 0.1 full-scale | `lfcc_features.py:L32-L37` |
| piso de RMS para aplicar gain | `speech_rms > 1e-8` | `lfcc_features.py:L177-L180` |
| pooling | media + `np.std` poblacional (`ddof=0`) | `lfcc_features.py:L123-L129` |

**FACT:** el RMS se calcula concatenando solo los chunks de habla ch0; se aplica un gain comun a
cada chunk antes de LFCC (`lfcc_features.py:L167-L180`). Los LFCC se calculan por chunk y los frames
se concatenan despues, por lo que delta y delta-delta no cruzan fronteras de turnos
(`lfcc_features.py:L182-L191`).

### Dependencias y seguridad de producto

| Operacion | Dependencia real | Fuente |
|---|---|---|
| arrays, STFT/pooling auxiliar | NumPy | `lfcc_features.py:L18-L22` |
| STFT y deltas | librosa | `lfcc_features.py:L21`, `L73-L74`, `L99-L103` |
| DCT-II | `scipy.fft.dct` | `lfcc_features.py:L22`, `L81` |
| WAV/CSV/builder | soundfile + pandas | `build_spectral_dataset.py:L15-L20`, `L31-L41`, `L83-L84` |

**FACT:** NumPy es `BSD-3-Clause`, librosa `ISC` y SciPy `BSD-3-Clause`; ninguna de esas licencias
obliga a marcar la familia no-productiva. **Decision de contrato:** LFCC es `product_safe=True` por
licencia.

**FACT:** librosa esta en el extra `research` y SciPy en `train`; la imagen de inferencia declara
ser minima y no instala ninguna (`pyproject.toml:L7-L16`, `L18-L26`). Soundfile y pandas son solo
infraestructura del builder y no deben entrar al extractor.

### Metadata y bloqueadores

- **FACT:** el calculo acepta `anon_id` y lo copia al dataclass, pero no lo usa para ningun valor de
  feature (`lfcc_features.py:L107-L111`, `L140-L194`). El port debe eliminar ese argumento.
- **FACT:** el builder usa `splits_df` para elegir llamadas y adjuntar `label` y `fold`
  (`build_spectral_dataset.py:L53-L58`, `L76-L81`). Esto esta fuera del calculo, pero copiar el
  builder dentro del extractor seria un bloqueador rojo.
- **FACT/BLOQUEADOR ROJO:** el extractor fuente usa turns oficiales como VAD oracle
  (`build_spectral_dataset.py:L31-L38`; `lfcc_features.py:L140-L173`). En nuestro producto solo
  puede usar `ex.seg` generado en inferencia; `oracle@1` no es desplegable.
- **FACT:** no usa label, split ni duracion para calcular features.

### Registro objetivo y cambios

```python
@extractors.register(
    "spectral_factory.lfcc",
    version=1,
    channels=(0,),
    needs_seg=True,
    license="BSD-3-Clause",
    product_safe=True,
    budget_ms=150,
)
def extract(ex: AudioExample) -> ExtractorResult:
    ...
```

El bloque anterior especifica el port NumPy-only. Si el integrador autoriza conservar la
implementacion de librosa/SciPy, el metadata debe declarar `license="ISC AND BSD-3-Clause"`; sigue
siendo `product_safe=True` por licencia, pero deja de cumplir la imagen minima actual.

`budget_ms=150` es **INF/objetivo de aceptacion**, no una medicion del port. El source warm cabe;
el source cold no. El port debe medir ambos sobre 150 s antes de congelarse.

Cambios obligatorios:

1. Eliminar `anon_id`, paths, pandas, soundfile y `SpectralLatent`.
2. Obtener turnos ch0 de `ex.seg`; diagnosticar segmentacion ausente/oracle.
3. Prefijar las 120 claves con `spectral_factory.lfcc.`.
4. Congelar explicitamente el orden anterior en el spec de experimento; no derivarlo de un CSV.
5. Devolver valores finitos aun en audio corto mediante politica congelada de default o rechazo.
6. Diagnosticos minimos: `segmentation_missing`, `segmentation_oracle`, `n_turns_ch0`,
   `n_usable_turns`, `n_lfcc_frames`, `n_short_turns`, `speech_rms_before`, `gain_applied`,
   `delta_width_min`, `nonfinite_count` y `mask_source`.
7. Para conservar la imagen minima, reimplementar STFT, DCT y deltas con NumPy, o pedir al
   integrador una decision explicita para mover librosa/SciPy al camino de inferencia.
8. Congelar explicitamente `center`, padding y tratamiento de bordes para STFT/deltas; el source
   los deja en defaults de una version de librosa no fijada.

**Clasificacion:** la traduccion matematica es mecanica y no requiere redisenar la feature. El port
al contrato/producto no es copy-paste: requiere adaptar segmentacion y, bajo la politica actual de
imagen, reimplementar primitivas.

## Familia 2: Shimmer CS3 12-dim

### Definicion exacta

**FACT:** Parselmouth/Praat estima F0 sobre el canal completo; los turns ch0 restringen runs
contiguos voiced. Dentro de cada run se aproximan pulsos glotales, se mide amplitud local y se
calcula APQ3/CS3 en ventanas de pulsos (`persona3_prosodic/prosodic_features.py:L66-L78`,
`L81-L97`, `L100-L179`, `L182-L202`).

### Orden exacto de columnas del builder

**FACT:** el builder crea `anon_id,label,fold` y luego hace `row.update(feats)`
(`persona3_prosodic/build_prosodic_dataset.py:L76-L84`). Esas tres columnas no pertenecen al
extractor.

**FACT:** `stats()` inserta `mean,std,p10,p90`; se invoca para shimmer, delta de shimmer y
delta-delta de shimmer, en ese orden (`prosodic_features.py:L230-L248`). Las 12 columnas exactas son:

```text
shimmer_mean, shimmer_std, shimmer_p10, shimmer_p90,
dshimmer_mean, dshimmer_std, dshimmer_p10, dshimmer_p90,
ddshimmer_mean, ddshimmer_std, ddshimmer_p10, ddshimmer_p90
```

En nuestro arnes cada nombre gana el prefijo `spectral_factory.shimmer_cs3.`.

### Reproducibilidad del orden

**FACT:** no depende de glob. Depende de insercion determinista en dict
(`prosodic_features.py:L233-L248`).

**OBS:** el orden de filas es no determinista por `Pool.imap_unordered()`
(`build_prosodic_dataset.py:L58-L67`, `L76-L84`). El schema de columnas permanece estable.

### Parametros numericos y operaciones

| Parametro | Valor literal | Fuente |
|---|---:|---|
| F0 minimo | 60 Hz | `prosodic_features.py:L45` |
| F0 maximo | 450 Hz | `prosodic_features.py:L46` |
| hop de F0 | 10 ms | `prosodic_features.py:L47-L50`, `L73-L75` |
| hop a 8 kHz | 80 muestras | `prosodic_features.py:L49-L50` |
| ventana de analisis de F0 | **UNK:** no se pasa a `Sound.to_pitch()` | `prosodic_features.py:L73-L75` |
| pulsos por ventana CS3 | 10 | `prosodic_features.py:L52`, `L157-L179` |
| hop de ventana CS3 | 5 pulsos, 50 % | `prosodic_features.py:L53`, `L157-L179` |
| minimo APQ3 | 3 pulsos | `prosodic_features.py:L54`, `L161-L175` |
| busqueda de pulso | +/- 0.25 periodos | `prosodic_features.py:L117-L139` |
| semiventana de amplitud | 1.5 ms | `prosodic_features.py:L146-L154` |
| formula | `100 * mean(abs(A_i - mean(A_{i-1:i+1}))) / mean(A)` | `prosodic_features.py:L169-L175` |
| delta | primera diferencia `np.diff` | `prosodic_features.py:L230` |
| delta-delta | segunda diferencia `np.diff` | `prosodic_features.py:L231` |
| pooling | media, std poblacional, percentiles 10 y 90 | `prosodic_features.py:L233-L242` |
| normalizacion de audio | ninguna | **OBS:** no aparece entre `L66-L202` |

**OBS/Bug documental:** el docstring de `estimate_f0_track()` todavia dice hop de 20 ms, pero la
constante y la llamada ejecutable usan 10 ms (`prosodic_features.py:L47-L50`, `L66-L75`). Para el
port manda el codigo ejecutable: 10 ms.

### Dependencias y seguridad de producto

| Operacion | Dependencia real | Fuente |
|---|---|---|
| arrays, pulsos, APQ3, diff y pooling | NumPy | `prosodic_features.py:L40-L43`, `L100-L248` |
| F0 | `parselmouth.Sound.to_pitch` (Praat) | `prosodic_features.py:L43`, `L66-L78` |
| WAV/CSV/builder | soundfile + pandas | `build_prosodic_dataset.py:L14-L18`, `L27-L39`, `L83-L89` |
| librosa | no se usa en la implementacion actual de la feature | `prosodic_features.py:L40-L43` |
| SciPy | no se usa en la implementacion actual de la feature | `prosodic_features.py:L40-L43` |

**FACT:** Parselmouth es GPL-3.0-or-later y nuestro registro rechaza dependencias GPL para bundle
productivo (`src/altur/registry.py:L150-L160`; `docs/LICENSE_AUDIT.md`). **Decision de contrato:**
la implementacion fiel actual es `product_safe=False`.

### Metadata y bloqueadores

- **FACT:** `anon_id` solo se copia al dataclass de salida; no participa en el calculo
  (`prosodic_features.py:L205-L209`, `L251-L257`). Debe eliminarse.
- **FACT:** el builder usa splits para enumerar llamadas y adjuntar label/fold
  (`build_prosodic_dataset.py:L51-L56`, `L76-L81`). Copiarlo al extractor seria un bloqueador rojo.
- **FACT/BLOQUEADOR ROJO:** usa turns oficiales como VAD oracle
  (`build_prosodic_dataset.py:L27-L36`; `prosodic_features.py:L57-L63`, `L182-L202`). Produccion
  debe usar `ex.seg` no-oracle.
- **FACT:** label, split y duracion no entran al calculo de las 12 features.
- **UNK/BLOQUEADOR DE EQUIVALENCIA:** el source no congela todos los defaults internos de
  `Sound.to_pitch()` ni la version exacta de Parselmouth/Praat. Una reimplementacion no puede
  prometer equivalencia numerica solo con esta spec.

### Registro objetivo y cambios

Port mecanico de investigacion:

```python
@extractors.register(
    "spectral_factory.shimmer_cs3",
    version=1,
    channels=(0,),
    needs_seg=True,
    license="GPL-3.0-or-later",
    product_safe=False,
    budget_ms=200,
)
def extract(ex: AudioExample) -> ExtractorResult:
    ...
```

`budget_ms=200` es **INF/objetivo de aceptacion** basado en una sola medicion source; debe medirse
de nuevo. Este extractor sirve para comparacion experimental y el empaquetado debe rechazarlo.

Para un port productivo:

1. Reimplementar el tracker F0 con una dependencia permisiva o codigo propio.
2. Congelar ventana, padding, interpolacion, criterios voiced/unvoiced y todos los defaults hoy
   heredados de Praat.
3. Revalidar equivalencia de F0, pulsos, CS3 y las 12 columnas; no asumirla.
4. Usar `ex.seg` no-oracle y prohibir paths/metadata.
5. Prefijar las 12 claves.
6. Diagnosticos minimos: `segmentation_missing`, `segmentation_oracle`, `n_pitch_frames`,
   `n_voiced_frames`, `voiced_fraction`, `n_voiced_runs`, `n_pulses`, `n_short_runs`,
   `n_shimmer_values`, `f0_failure_rate`, `nonfinite_count` y `mask_source`.
7. Si la tasa de fallo difiere mas de 10 % entre clases en evaluacion, marcar la familia
   `confounded`; el extractor no recibe la etiqueta, el runner hace la auditoria.

**Clasificacion:** mecanico como wrapper de investigacion `product_safe=False`; **no mecanico para
producto**. La frontera no es solo de wiring: cambiar Praat cambia el algoritmo medido.

## Corrupcion: no forma parte de los extractores

`persona3_prosodic/corruption.py` no define columnas LFCC o Shimmer. No debe copiarse dentro de
ningun extractor. Si se porta, pertenece al registro `transforms` y debe respetar el reparto
congelado de perturbaciones.

Parametros fuente auditados:

- mu-law `mu=255` y cuantizacion a 256 niveles (`corruption.py:L47-L55`);
- A-law `A=87.6` (`corruption.py:L58-L79`);
- Butterworth orden 4, 300-3400 Hz, con techo `0.98 * Nyquist` (`corruption.py:L82-L89`);
- ruido 15-25 dB SNR (`corruption.py:L92-L99`, `L121-L124`);
- pitch shift -1 a +1 semitono y time stretch 0.9-1.1 (`corruption.py:L102-L110`, `L123-L124`);
- probabilidad global 0.5 y familias `channel,pitch_shift,time_stretch`
  (`corruption.py:L113-L128`).

**OBS/Bug de reproducibilidad:** `TelephonyAugmenter` crea `self._rng` desde `seed`, pero
`add_gaussian_noise()` usa `np.random.normal` global (`corruption.py:L92-L99`, `L128-L147`). La
semilla declarada no controla el ruido. No portar ese bug.

## Costo fuente por llamada

**OBS transcrito, no remedido:** `latency_benchmark.json` separa extraccion e inferencia
(`persona1_reference_calibration/data/latency_benchmark.json:L4-L27`). El benchmark cronometra
carga, ambas extracciones, inferencia por rama, fusion y formato por separado
(`benchmark_latency.py:L97-L140`). Entrena los modelos fuera del cronometro sobre todos los datos
disponibles (`benchmark_latency.py:L64-L94`) y ejecuta una pasada cold seguida de una warm
(`benchmark_latency.py:L144-L165`).

| Familia | Extraccion cold | Extraccion warm | Inferencia cold | Inferencia warm |
|---|---:|---:|---:|---:|
| LFCC 120-dim | 1150.698 ms | 37.562 ms | 0.323 ms | 0.318 ms |
| Shimmer CS3 12-dim | 132.244 ms | 136.729 ms | 0.468 ms | 0.414 ms |

Costos compartidos del mismo registro: I/O warm 6.343 ms, fusion 0.299 ms, Platt+formato 0.025 ms,
total warm 181.690 ms; total cold 1291.007 ms. La memoria pico reportada es 36.632 MB warm y
62.617 MB cold (`latency_benchmark.json:L4-L27`).

Metadatos obligatorios de estas metricas: `n=1`; unidad independiente = una llamada real de 150 s;
split = **UNK** y no se deriva porque requeriria cruzar su identificador con `splits.csv`; protocolo
= llamada mas cercana a 150 s del manifest, modelos ajustados sobre todas las filas, una pasada cold
y una segunda warm (`benchmark_latency.py:L51-L67`, `L144-L165`); seed = no aplica al camino
determinista cronometrado, aunque el solver no fija `random_state`; commit =
`9583e3a6c9677d3df920637868416131ba757d10`; digest del codigo efectivo = tree Git
`843808e4a0abbc3f04b907083c11d5eab2f569cf`.

**INF:** solo sirven para dimensionar el primer budget del port. No demuestran p50/p95, no aislan
variabilidad entre llamadas, hardware o procesos, y LFCC incumple 1 s en cold en esa unica corrida.

## Calibracion y respuesta: no portar junto con las features

**FACT:** `calibrate.py` aplica Platt out-of-fold usando sus folds y despues ajusta parametros de
"produccion" sobre todas las filas (`persona1_reference_calibration/calibrate.py:L58-L95`). Esos
folds incluyen `val`; por tanto sus parametros no entran a nuestro bundle.

**FACT:** `format_detect_response.py` fija threshold 0.5 y define `confidence` como probabilidad de
la clase predicha (`persona1_reference_calibration/format_detect_response.py:L44-L59`). En nuestro
proyecto la semantica oficial de `confidence` sigue **UNK** hasta confirmacion. No heredarla.

La familia de features entrega features y diagnostics. Modelo, fusion, calibrador, threshold y
formato HTTP permanecen componentes separados del arnes.

## QUE NO SE PUEDE CITAR

### Regla comun

**FACT:** sus folds cubren las 353 llamadas, equivalentes a nuestro `pooled5_v1` con
`val_contaminated=True`. `fusion_ablation.py` extrae label/fold del mismo dataframe y calcula OOF
sobre esos folds (`persona3_prosodic/fusion_ablation.py:L37-L63`, `L73-L82`, `L95-L139`).
`calibrate.py` vuelve a usar esos folds y el calibrador final ajusta todas las filas
(`calibrate.py:L58-L95`).

Metadatos comunes de las cifras de modelo siguientes: `n=353` salvo donde se indique otro valor;
unidad = llamada; split = train + val mezclados; protocolo = `pooled5_v1`,
`val_contaminated=True`; seed = **UNK**; commit =
`9583e3a6c9677d3df920637868416131ba757d10`; digest efectivo disponible = tree Git
`843808e4a0abbc3f04b907083c11d5eab2f569cf`. Como el seed no esta registrado y `val` participo,
**ninguna cifra de desempeno sobrevive como claim de nuestro modelo**.

### Inventario de cifras publicadas y veredicto

| Cifra publicada por spectral factory | Fuente | ¿Sobrevive? | Motivo preciso |
|---|---|---|---|
| Forense: 40+40 clips; umbral AUC 0.85; bandwidth ~0.82; comb/ZCR ~0.79 | `README.md:L17-L44` | **No** | Submuestra exploratoria, seed/digest efectivo del run UNK y no es nuestro protocolo OOF. No se porta ningun espectrograma individual. |
| Atajo RMS: ~9 dB; human -24.1 dBFS, synthetic -15.2 dBFS; AUC ~0.88 | `persona2_reference_spectral/README.md:L40-L51`; `HALLAZGOS_LOG.md:L11-L16` | **No como resultado** | Sirve para justificar normalizacion, no para medir el port; split pooled y run sin seed. |
| `static3_mean`: AUC 0.943, luego 0.749 bajo corrupcion | `persona2_reference_spectral/README.md:L53-L60` | **No** | Pooled/contaminado; ademas contradice el 0.735 del log. |
| `static3_mean`: AUC 0.943, luego 0.735 bajo corrupcion | `HALLAZGOS_LOG.md:L18-L25` | **No** | Contradiccion interna con 0.749; no hay una cifra canonica que transportar. |
| LFCC global: ~1.00 o >0.99 incluso con L2 `C=0.001` | `persona2_reference_spectral/README.md:L62-L77`; `HALLAZGOS_LOG.md:L27-L36` | **No** | `pooled5_v1`, val contaminado y posible fuga de motor/voz. |
| Ablacion LogReg: prosodica AUC 0.828276/EER 25.4745 %, espectral 0.999967/0.579639 %, fusion 0.999934/0.825944 % | `persona3_prosodic/data/matriz_ablacion.csv:L1-L4`; generacion en `fusion_ablation.py:L79-L82`, `L99-L139` | **No** | Las 353 incluyen val; grupos por llamada no prueban hablante/voz ni vendor no vistos. |
| Ajustes prosodicos historicos: AUC 0.842 -> 0.803 -> 0.839; EER 24.32 % -> 23.49 % | `HALLAZGOS_LOG.md:L104-L118`, `L129-L143`; `persona3_prosodic/README.md:L44-L55` | **No** | Cambian tracker/modelo/config entre puntos y todos usan el pool contaminado. No son una ablacion controlada transportable. |
| Robustez resumida: clean AUC espectral/prosodica/fusion 1.0000/0.8283/0.9999 | `HALLAZGOS_LOG.md:L181-L190` | **No** | Clean y corrupciones se evaluan sobre las 353 pooled; val ya influyo en desarrollo. |
| Ruido: 0.9714/0.8005/0.9756 | `HALLAZGOS_LOG.md:L184-L190` | **No** | Mismo protocolo contaminado. |
| Pitch +1: 0.9745/0.6250/0.9652 | `HALLAZGOS_LOG.md:L184-L190` | **No** | Mismo protocolo contaminado. |
| Time-stretch 0.9x: 0.9893/0.6208/0.9845 | `HALLAZGOS_LOG.md:L184-L190` | **No** | Mismo protocolo contaminado. |
| Low-pass 3400 Hz: 0.9951/0.8370/0.9960 | `HALLAZGOS_LOG.md:L184-L201` | **No** | Mismo protocolo contaminado; no refuta D-A1.6. |
| Opus 16 kbps: 0.9985/0.7480/0.9990 | `HALLAZGOS_LOG.md:L184-L201` | **No** | Mismo protocolo contaminado. |
| Augment prosodico clean: 0.8283 -> 0.8088, delta -0.0195 | `HALLAZGOS_LOG.md:L247-L264` | **No** | Pooled; una sola copia aleatoria por llamada; seed UNK. |
| Augment prosodico pitch +1: 0.6250 -> 0.7078, delta +0.0828 | `HALLAZGOS_LOG.md:L247-L264` | **No** | Pooled y no reproduce el regimen estocastico propuesto. |
| Augment prosodico stretch 0.9x: 0.6208 -> 0.6969, delta +0.0761 | `HALLAZGOS_LOG.md:L247-L264` | **No** | Pooled y no reproduce el regimen estocastico propuesto. |
| Platt prosodico: Brier 0.162925 -> 0.159381 | `persona1_reference_calibration/data/calibration_summary.csv:L1-L4`; codigo en `calibrate.py:L141-L170` | **No** | Platt se ajusta/evalua con folds pooled que incluyen val. |
| Platt espectral: Brier 0.014412 -> 0.016655 | `persona1_reference_calibration/data/calibration_summary.csv:L3` | **No** | Ademas de contaminado, Platt empeora el Brier publicado. |
| Platt fusion: Brier 0.004483 -> 0.003128 | `persona1_reference_calibration/data/calibration_summary.csv:L4` | **No** | Contaminado; el calibrador final usa todas las filas incluidas las de val. |
| Parametros Platt A/B de las tres ramas | `calibration_summary.csv`; `calibrate.py:L82-L95`, `L194-L203` | **No** | Son artefactos ajustados con filas que incluyen val, no solo una metrica. No cargarlos. |
| Exactitud de respuestas formateadas | `format_detect_response.py:L74-L90` | **No** | Reusa labels y filas pooled; ademas la semantica de confidence no esta confirmada. |

**Unica reutilizacion permitida de numeros fuente:** los tiempos del apartado de costo pueden
usarse como **estimacion inicial de budget**, siempre con su bloque completo de metadatos `n=1` y
sin presentarlos como benchmark de nuestro port. Todos los AUC, EER, Brier, p-values, deltas,
exactitudes y parametros Platt deben volver a generarse exclusivamente OOF sobre `train`, con
nuestro protocolo, seed, commit y digest efectivo.

## Criterios de aceptacion del port

### LFCC

- 120 claves exactas, ordenadas y prefijadas como arriba.
- Ningun acceso a metadata o filesystem.
- Segmentacion no-oracle en producto.
- Finito y determinista para el mismo `AudioExample`.
- Diagnosticos completos, incluidos fallos y audio corto.
- `product_safe=True` solo si el codigo efectivo no importa dependencias restringidas.
- Benchmark cold y warm de 150 s contra `budget_ms=150`.

### Shimmer

- Extractor fiel con Parselmouth queda `product_safe=False` y no entra al bundle.
- Un candidato productivo requiere reimplementacion permisiva y prueba de equivalencia separada.
- 12 claves exactas, ordenadas y prefijadas.
- Defaults F0 totalmente congelados; ningun default implicito de biblioteca.
- Diagnosticos de F0/pulsos y tasa de fallo por clase auditada fuera del extractor.
- Benchmark cold y warm antes de fijar budget definitivo.

## Preguntas abiertas

- **UNK:** equivalencia numerica tolerable entre una reimplementacion NumPy de LFCC y
  librosa/SciPy; debe congelarla el integrador antes del test dorado.
- **UNK:** ventana y defaults completos usados internamente por la version de Praat del source.
- **UNK:** segmentador productivo que sustituira `turns.json` para estas familias.
- **UNK:** si el integrador acepta ampliar la imagen con SciPy/librosa o exige NumPy puro.
- **UNK:** significado oficial de `confidence`; no se toma del repo fuente.
