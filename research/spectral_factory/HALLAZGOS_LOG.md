# Registro de Hallazgos y Auditoría de Modelo

```
Fecha y Hora: 12 de Septiembre de 2026 - 01:00 AM (Local Time)
Módulo: persona2_reference_spectral / persona3_prosodic
Repositorio: chorizos-circuits-spectral-factory

ASUNTO: Auditoría de Atajos de Canal/Volumen y Fuga de Motor TTS en
Experimentos de Ablación

1. ATAJO DE VOLUMEN (HALLAZGO):
   - Se identificó una diferencia de ~9 dB RMS promedio entre clases en
     tramos hablados (Humano: -24 dBFS vs. Sintético: -15 dBFS).
   - Acción tomada: se implementó normalización RMS del Canal 0 (llamante)
     a -20 dBFS, previa a la extracción de LFCC, para forzar a ambas clases
     a competir con la misma potencia promedio.

2. ATAJO DE ANCHO DE BANDA (CONFIRMADO):
   - La característica `static3_mean` de LFCC obtiene un AUC de 0.943 sola,
     pero cae a 0.735 bajo el módulo de corrupción de canal (G.711 μ-law/
     A-law + paso-banda 300-3400 Hz). Esto confirma que el atajo de canal
     detectado en el análisis forense original (ancho de banda telefónico,
     AUC≈0.82) sigue presente — LFCC lo re-expresa de forma más eficiente en
     un solo coeficiente DCT — y que la corrupción en el DataLoader es
     indispensable para destruirlo.

3. FUGA DE MOTOR DE SÍNTESIS (TTS ENGINE LEAKAGE — HIPÓTESIS ABIERTA):
   - El AUC global del modelo se mantiene > 0.99 incluso bajo regularización
     L2 agresiva (C=0.001).
   - Diagnóstico: `GroupKFold` agrupado por `anon_id` (llamada/hablante)
     protege contra la fuga de la identidad del hablante, pero NO contra la
     fuga del motor de síntesis (TTS/vocoder). Si la clase `synthetic`
     proviene de pocos motores TTS conocidos, el modelo memoriza la firma
     del vocoder en vez de aprender síntesis en general.
   - Riesgo: en el set de evaluación oculto (con motores TTS no vistos, según
     el propio README del dataset), el rendimiento podría colapsar.

TAREAS Y NEXT STEPS:
   - Mantener activas las alertas automáticas en `fusion_ablation.py` y
     `train_isolated_spectral.py` (ya implementadas, se disparan cuando
     AUC > 0.97).
   - Consultar/verificar la diversidad de motores TTS en la clase sintética
     del dataset (pregunta abierta, no resoluble solo con los datos locales
     — requiere info de los organizadores del reto o inspección manual).
   - Priorizar la rama prosódica (Shimmer CS3) y considerar una pérdida tipo
     OC-Softmax para mitigar la memorización de vocoders en el backbone real
     de Persona 2.
```

## Respecto a qué normalizamos (confirmación técnica)

La normalización RMS implementada en `persona2_reference_spectral/lfcc_features.py`
(`extract_spectral_latent`) es exactamente el enfoque recomendado:

- **Se normaliza contra un valor absoluto fijo, no contra otra llamada ni
  contra el Canal 1.** El objetivo es -20 dBFS (`TARGET_RMS_DBFS = -20.0`,
  `TARGET_RMS = 10**(-20/20) = 0.1` en amplitud lineal, full-scale=1.0),
  aplicado por igual a cada clip.
- **El RMS se calcula solo sobre los tramos hablados del Canal 0**
  (`turns/<anon_id>.json`, canal 0 = llamante), nunca sobre el clip completo.
  Calcularlo sobre el clip completo habría contaminado la estimación de nivel
  con el mismo piso de ruido de silencio donde el análisis forense original
  ya había encontrado el atajo de canal.
- **Se descartó deliberadamente usar la razón Canal 0 / Canal 1 como
  referencia.** El Canal 1 (agente del banco) viene del servidor interno a
  volumen prácticamente constante, así que normalizar contra él no cancela
  nada nuevo — solo reintroduce la misma variable de ganancia por otra vía.
  Normalizar el Canal 0 contra un valor objetivo fijo en su propio eje es lo
  que efectivamente elimina el atajo, y así quedó implementado.

Con el fix aplicado (y confirmado con un re-run completo: los números no
cambian frente a un target RMS distinto, como se espera de un simple
reescalado — la normalización por `StandardScaler` aguas abajo ya lo
compensaba), el hallazgo 1 queda resuelto; los hallazgos 2 y 3 siguen abiertos
y documentados en `persona2_reference_spectral/README.md`.


---

## Adenda — 12 de Septiembre de 2026, ~08:00 AM (Local Time)

```
ASUNTO: Fix de latencia (Tarea de Persona 1: benchmark <1s) -- Opcion B ejecutada

DECISION: Reemplazar librosa.pyin por parselmouth (Praat, autocorrelacion en C)
en persona3_prosodic/prosodic_features.py, siguiendo la razon dada: la capa
probabilistica/Viterbi de pYIN (decidir voiced/unvoiced con incertidumbre) es
redundante porque turns.json ya se usa como VAD oracle. Rango de F0 ajustado a
80-350 Hz (de 60-500 Hz) y hop a 20ms (de 10ms).

RESULTADO -- LATENCIA:
  - Extraccion prosodica sola: 7.4s -> 0.115s (~65x) sobre la llamada de
    referencia de 150s.
  - Latencia end-to-end WARM (servidor vivo, lo que importa en produccion):
    7.47s -> 0.159s. Objetivo <1s: CUMPLIDO con margen amplio.
  - Se descubrio y corrigio un segundo problema de medicion en el propio
    benchmark: la extraccion espectral (LFCC) mostraba ~1s en la PRIMERA
    llamada del proceso por costo de inicializacion de FFT/BLAS (cold start),
    no un costo real por peticion. benchmark_latency.py ahora mide un warm-up
    descartado + la medicion real, y reporta cold-start aparte (1.12s) para
    que el equipo decida si les importa segun como desplieguen el endpoint
    (servidor persistente: no importa; serverless/reinicio frecuente: si).

RESULTADO -- PRECISION (COSTO REAL, NO GRATIS):
  - AUC out-of-fold de la rama prosodica aislada: 0.842 -> 0.803 (mejor modelo
    cambio de LogisticRegression a RandomForest).
  - Esto CONTRADICE la expectativa inicial de "no pierdes precision en los
    periodos gloticos" -- si hay una perdida medible, atribuible al rango de
    F0 mas angosto y/o al tracker distinto (Praat autocorrelacion vs. pYIN
    probabilistico). Se documenta la discrepancia en vez de ocultarla (regla
    de la sesion: nombrar cuando el resultado empirico contradice la
    expectativa).

DECISION DEL EQUIPO: se acepta el trade-off porque el objetivo de latencia
era la restriccion dura (bloqueaba el pipeline completo); recuperar AUC con
el mismo tracker (rango mas ancho, hop mas fino, dado que ahora hay margen de
sobra en el presupuesto de latencia) queda como optimizacion pendiente si
sobra tiempo antes de la entrega.

ARCHIVOS ACTUALIZADOS: persona3_prosodic/prosodic_features.py,
persona3_prosodic/data/latents_prosodic.csv (regenerado),
persona1_reference_calibration/benchmark_latency.py (metodologia warm/cold),
persona1_reference_calibration/data/latency_benchmark.json (regenerado).
```


---

## Adenda 2 — 12 de Septiembre de 2026, ~09:00 AM (Local Time)

```
ASUNTO: Recuperacion de AUC sin sacrificar latencia (dado el margen de 841ms
libres del fix anterior)

DECISION: Ensanchar el rango de F0 de 80-350 Hz a 60-450 Hz (cubre creaky
voice masculino <80Hz y picos agudos femeninos >350Hz) y afinar el hop de
20ms a 10ms, en persona3_prosodic/prosodic_features.py.

RESULTADO -- PRECISION (recuperada):
  - AUC aislado de la rama prosodica: 0.803 -> 0.839 (RandomForest), EER
    23.49% -- IGUALA o supera levemente el numero original con pYIN
    (AUC=0.842 con LogisticRegression, EER=24.32%). Se recupero casi toda la
    perdida documentada en la Adenda 1.
  - Robustez a corrupcion de canal: se mantiene (caida promedio de AUC
    -0.075, es decir el AUC en promedio SUBE tras la corrupcion -- la rama
    prosodica sigue siendo una senal ortogonal al atajo de canal).

RESULTADO -- LATENCIA (se mantiene holgada):
  - Extraccion prosodica sola: 115ms -> 137ms (+22ms por el hop mas fino).
  - Total end-to-end WARM: 159ms -> 182ms. Sigue MUY por debajo del
    presupuesto de 1000ms (818ms libres, de 841ms que habia antes).
  - Cold start (sin cambios, ya identificado como costo de FFT/BLAS del
    primer request, no relacionado con este ajuste): ~1.29s.

CONCLUSION: el ajuste recupero practicamente toda la precision perdida en el
fix de latencia anterior, gastando solo ~23ms adicionales de un presupuesto
de latencia que sobraba por casi 850ms. No quedan trade-offs pendientes
conocidos en la rama prosodica por ahora.

NOTA PARA DESPLIEGUE (no implementada aqui, responsabilidad de quien sirva
el endpoint POST /detect): correr una peticion de warm-up (con audio
cualquiera) al arrancar el contenedor/servidor, para que la PRIMERA peticion
real de los jueces no pague el costo de cold-start de FFT/BLAS (~1.1-1.3s
extra solo esa vez).

ARCHIVOS ACTUALIZADOS: persona3_prosodic/prosodic_features.py,
persona3_prosodic/data/{latents_prosodic,scores_prosodic,scores_fusion,
matriz_ablacion,robustness_check,robustness_summary}.csv (regenerados),
persona1_reference_calibration/data/* (recalibrado y rebenchmark).
```


---

## Adenda 3 — 12 de Septiembre de 2026, ~10:15 AM (Local Time)

```
ASUNTO: Fase 1 -- Stress Test de Robustez sobre el ZIP real de Persona 2
(robustness_phase1 (2).zip, 353 llamadas x 6 condiciones, leer-one-fold-out)

MATRIZ DE RESILIENCIA (AUC out-of-fold, nunca evaluado con un modelo que vio
la llamada, ni en su version limpia):

  condicion          auc_espectral  auc_prosodica  auc_fusion
  clean              1.0000         0.8283         0.9999
  noise_snr10        0.9714         0.8005         0.9756
  pitch_up1st        0.9745         0.6250         0.9652
  timestretch_0.9x   0.9893         0.6208         0.9845
  lowpass_3400hz     0.9951         0.8370         0.9960
  opus_16kbps        0.9985         0.7480         0.9990

HALLAZGO 1 -- CONTRADICE LA HIPOTESIS DEL EQUIPO (rama espectral):
  Se esperaba que lowpass/opus (atacan directamente el ancho de banda
  telefonico, el atajo de canal ya conocido) tumbaran el AUC espectral. Paso
  lo CONTRARIO: son las condiciones donde MENOS cae (delta -0.005 y -0.0015),
  mientras que noise_snr10 (que no ataca canal/bandwidth especificamente) es
  la que mas la daña, y aun asi solo -0.029. Ninguna corrupcion logra mover
  significativamente el AUC espectral. Esto REFUERZA (no descarta) la
  hipotesis abierta de la Adenda previa sobre posible fuga de motor TTS /
  pocas voces sinteticas -- si el atajo dominante fuera bandwidth, lowpass/
  opus deberian ser las mas destructivas, y son las MENOS.

HALLAZGO 2 -- VULNERABILIDAD NUEVA, NO ANTICIPADA (rama prosodica):
  Se esperaba que Shimmer se mantuviera estable (senal ortogonal al canal).
  Es cierto para ruido/lowpass/opus (se mantiene o hasta mejora). Pero
  pitch_up1st y timestretch_0.9x la DESTRUYEN: AUC 0.8283 -> 0.6250 y 0.6208
  respectivamente (delta ~-0.20 en ambos casos). Explicacion tecnica: Shimmer
  se calcula pitch-synchronous (periodos gloticos derivados de F0) -- alterar
  directamente el pitch o la duracion ataca la base matematica del calculo,
  a diferencia de ruido/codec/filtro que no tocan la periodicidad en si.

RECOMENDACION PARA FASE 2 (TelephonyAugmenter, P=0.5): activar pitch_up1st y
timestretch_0.9x tiene mas justificacion empirica que lowpass/opus (que casi
no mueven nada en ninguna rama); noise_snr10 tambien vale la pena por ser la
que mas daña a la espectral, aunque el efecto sea chico. La pregunta de fondo
(por que la rama espectral es casi perfecta BAJO TODAS las condiciones) sigue
sin resolverse -- sigue apuntando a verificar diversidad de motores TTS en el
dataset, no a que falten las corrupciones correctas en el entrenamiento.

METODOLOGIA: modelos entrenados leave-one-fold-out (5 folds de
persona3_prosodic/data/splits.csv) sobre los MISMOS latents limpios ya
existentes (latents_prosodic.csv, latents_spectral.csv); cada llamada
corrompida se evaluo con el modelo de los OTROS 4 folds, nunca con uno
entrenado incluyendola (ni siquiera en su version limpia) -- las 353 llamadas
del ZIP de Ricardo son el dataset completo, no un held-out nuevo, asi que
evaluar con el modelo final habria sido optimista/no confiable.

SIN EXTRAER EL ZIP A DISCO: 12GB descomprimido vs. 17GB libres en la maquina.
Se leyo cada wav/json directo del zip a un buffer en memoria (io.BytesIO),
sin escribir nada a disco.

ARCHIVOS NUEVOS: phase1_stress_test/eval_stress_test.py,
phase1_stress_test/README.md, phase1_stress_test/data/{scores_por_condicion,
matriz_resiliencia_corrupcion,delta_vs_clean}.csv
```


---

## Adenda 4 — 12 de Septiembre de 2026, ~11:00 AM (Local Time)

```
ASUNTO: Validacion de la Fase 2 propuesta (augmentacion pitch-shift/time-stretch
en TelephonyAugmenter) contra el ZIP REAL de Ricardo -- resultado parcial, no
la hipotesis completa

METODOLOGIA (para que la prueba fuera honesta, no circular): se extendio
TelephonyAugmenter con pitch-shift (+-1 semitono) y time-stretch (0.9x-1.1x),
se genero UNA version aumentada por llamada (family elegida al azar por
llamada), se entreno leave-one-fold-out (clean UNION augmented en train,
nunca evaluando una llamada con un modelo que la vio) y se valido -- esto es
lo importante -- contra el audio YA corrompido por Ricardo con SU PROPIA
implementacion independiente (no la nuestra), leido directo del zip.

RESULTADO:
  condicion          sin_augmentacion   con_augmentacion   delta
  clean              0.8283             0.8088             -0.0195
  pitch_up1st        0.6250             0.7078             +0.0828
  timestretch_0.9x   0.6208             0.6969             +0.0761

VEREDICTO: la hipotesis del equipo (subir de ~0.62 a >0.80) NO SE CONFIRMA
completa. La augmentacion SI ayuda de forma real y medible (+0.08 y +0.076 de
AUC, mejora consistente en ambas condiciones, no ruido), pero se queda a mitad
de camino (~0.70-0.71), y ademas cuesta un poco en clean (-0.02).

HIPOTESIS DE POR QUE NO LLEGO A >0.80 (no verificada, queda abierta):
  Este experimento genero UNA sola copia aumentada por llamada, agregada una
  vez al set de entrenamiento de un clasificador clasico (regresion
  logistica) -- un regimen de augmentacion mucho mas debil que lo que haria
  un DataLoader real de PyTorch con augmentacion estocastica POR EPOCH (cada
  llamada veria docenas de valores distintos de pitch/rate a lo largo del
  entrenamiento, no uno solo fijo). Es esperable que un entrenamiento real
  con augmentacion estocastica recupere mas terreno que esta prueba con un
  clasificador clasico y una sola copia por llamada.

RECOMENDACION: la direccion (activar pitch-shift/time-stretch en el
DataLoader real de Persona 2) sigue siendo la correcta -- la mejora es real,
solo no tan grande como se esperaba con este placeholder simplificado. Vale
la pena que Persona 2 confirme si un entrenamiento con augmentacion
estocastica real (multiples epochs, multiples valores aleatorios por
llamada) cierra mas la brecha. No se debe presentar ante los jueces como
"resuelto a >0.80" -- la version honesta es "mejora real y medible, camino
correcto, recuperacion parcial con este placeholder".

ARCHIVOS NUEVOS: persona3_prosodic/corruption.py (extendido con pitch_shift/
time_stretch + rescale_turns), phase1_stress_test/train_augmented_prosodic.py,
phase1_stress_test/data/{latents_prosodic_augmented,
scores_prosodica_augmentada_vs_zip_real,comparacion_augmentacion_prosodica}.csv
```
