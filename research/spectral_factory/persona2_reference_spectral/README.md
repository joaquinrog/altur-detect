# Rama espectral — implementación de REFERENCIA (placeholder de Persona 2)

**Esto NO es el entregable real de Persona 2.** Persona 2 está construyendo el
módulo de corrupción/robustez (Fase 1: stress test con 5 tipos de degradación
sobre el fold de validación — ruido, pitch shift, time stretch, low-pass, Opus)
y, después de eso, el clasificador espectral real con LFCC + un backbone
LCNN/ECAPA-TDNN entrenado.

Mientras esa Fase 1 + el backbone real no estén listos, este módulo genera un
`latents_spectral.csv` de **referencia** (LFCC 60-dim: 20 estáticos + 20 Δ + 20
ΔΔ, ventana 20ms / salto 10ms a 8kHz — la misma especificación exacta de la
Tarea 2.2 de Persona 2 — pooleado a un vector fijo + regresión logística en vez
de un backbone entrenado) para poder ejercitar HOY la fusión tardía y la matriz
de ablación de 3 escenarios (Tarea 3.3), en vez de esperar sin poder validar
nada del lado de integración.

Cuando Persona 2 entregue su `latents_spectral.csv` real (mismo contrato, ver
`../persona3_prosodic/CONTRATOS.md` sección 3), se reemplaza el archivo en
`../persona3_prosodic/data/latents_spectral.csv` y `fusion_ablation.py` no
necesita ningún cambio de código — es exactamente el mecanismo de "drop-in
replacement" que ya usamos para `splits.csv` y `TelephonyAugmenter`.

## Orden de ejecución

```bash
python -m persona2_reference_spectral.build_spectral_dataset --workers 10
python -m persona2_reference_spectral.train_isolated_spectral
python -m persona3_prosodic.fusion_ablation
```

`train_isolated_spectral.py` hace tres cosas en una corrida: (1) compara EER
con vs. sin corrupción en el train fold (Tarea 2.3/5 del prompt de Persona 2),
(2) corre un forense por feature individual (clean vs. corrupted, mismo método
que `analisis_forense/`), y (3) un barrido de regularización para distinguir
sobreajuste de señal real. Los tres son necesarios para interpretar el AUC
correctamente — ver la sección de hallazgos abajo antes de creer el número.

> Registro formal de auditoría (mismo contenido, formato de log): ver [`../HALLAZGOS_LOG.md`](../HALLAZGOS_LOG.md).

## ⚠️ Hallazgos — leer antes de usar estos números en el pitch

Este placeholder **no dio un resultado limpio**, y eso en sí es información
útil para el equipo, no un fracaso del placeholder:

1. **Atajo de volumen/loudness encontrado y corregido.** La primera versión
   (sin normalizar RMS) mostraba `static0_mean` (energía promedio de la banda
   de frecuencia más baja) con **~9 dB de diferencia promedio entre clases**
   (human -24.1 dB vs. synthetic -15.2 dB, AUC≈0.88 esa sola feature). Es case
   con toda seguridad un artefacto de nivel de grabación/normalización, no una
   diferencia de síntesis real. Se corrigió normalizando el RMS del audio de
   habla antes de calcular LFCC (ver `lfcc_features.py`).

2. **El atajo de canal ya conocido (ancho de banda) reaparece, re-expresado
   por LFCC.** Después de corregir el volumen, `static3_mean` (un coeficiente
   DCT que captura la *forma* del espectro, no el nivel) seguía en AUC=0.943
   por sí solo. Al aplicar la corrupción de canal (G.711+paso-banda) a esta
   misma feature, su AUC **cae a 0.749** — la firma exacta de un atajo de
   canal, coherente con el ancho de banda telefónico (AUC≈0.82) que ya había
   encontrado el análisis forense original, solo que LFCC lo re-expresa de
   forma más eficiente en un solo coeficiente. La corrupción de canal está
   funcionando como debería sobre esta feature específica.

3. **El AUC global de la rama (~1.00, incluso con solo las features "dynamic"
   delta/delta-delta) sigue siendo sospechosamente perfecto, y esto NO se
   explica por sobreajuste ni por el atajo de canal.** Un barrido de
   regularización L2 (C de 1.0 a 0.001) muestra que el AUC se mantiene por
   encima de 0.99 incluso con regularización muy fuerte — si fuera solo
   sobreajuste (120 features, ~280 muestras de train por fold), un C muy
   pequeño lo habría colapsado, y no lo hizo. **Hipótesis abierta, sin
   descartar:** la clase `synthetic` de este dataset podría venir de muy pocas
   voces/motores TTS distintos, lo que le permitiría a cualquier clasificador
   "memorizar" la huella espectral de esas pocas voces en vez de aprender
   síntesis en general. `GroupKFold` por `anon_id` (llamada/hablante) protege
   contra fuga de **hablante**, pero NO contra este tipo de fuga de **motor de
   síntesis** — son problemas distintos. Si el set oculto de evaluación usa
   voces/TTS nunca vistos (como dice el README del dataset), este número
   podría no generalizar en absoluto.

**Recomendación concreta para el equipo:** antes de confiar en cualquier AUC
cercano a 1.0 de la rama espectral (placeholder o real), preguntar a los
organizadores (o revisar manualmente una muestra de audios) cuántos motores/
voces TTS distintos hay en la clase `synthetic`. Es la pregunta más importante
sin responder de todo este análisis, y afecta a cualquier rama del pipeline,
no solo a esta.

Los números crudos (`feature_forensics_spectral.csv`,
`corruption_ablation_spectral.csv`, el barrido de regularización impreso por
`train_isolated_spectral.py`) quedan guardados para que el equipo los revise
directamente, no solo mi interpretación.

## Por qué SÍ vale la pena este placeholder (y no es solo relleno)

- Verifica que el contrato de datos (`CONTRATOS.md`) realmente funciona de
  punta a punta antes de que la integración real dependa de él bajo presión de
  tiempo.
- Encontró dos atajos distintos (volumen y, re-confirmado, ancho de banda) y
  una alerta metodológica importante (posible fuga de motor TTS) ANTES de que
  el equipo invirtiera tiempo entrenando el backbone real sobre las mismas
  trampas.
- El extractor de LFCC + enmascaramiento VAD + normalización RMS es reutilizable
  tal cual por Persona 2 si decide no reinventar esa parte — la diferencia real
  está en el backbone (LCNN/ECAPA entrenado vs. pooling+regresión logística aquí).

## Decisiones de diseño (placeholder, documentadas para no confundir con el
   entregable final)

- **VAD con `turns.json` (oracle)**: igual que la rama prosódica, LFCC solo se
  calcula sobre los tramos de habla del canal 0. Si no se hiciera esto, LFCC
  capturaría el piso de ruido de silencio — exactamente donde el análisis
  forense encontró el atajo de canal. Un backbone real (LCNN/ECAPA) también
  necesitaría este mismo cuidado, no es exclusivo del placeholder.
- **Normalización de RMS antes de LFCC**: ver hallazgo 1 arriba — sin esto, la
  rama aprendía volumen, no espectro.
- **Pooling estadístico (media+std) en vez de un backbone entrenado**: con 353
  llamadas, entrenar una CNN/TDNN desde cero en este placeholder no aportaría
  una señal confiable — el placeholder existe para probar la integración, no
  para competir con el backbone real de Persona 2.
- **Sin corrupción de canal en el CSV compartido**: `latents_spectral.csv`
  se calcula sobre audio limpio (igual que `latents_prosodic.csv`), porque es
  el vector que se usa en inferencia/fusión. La comparación "con corrupción vs
  sin corrupción" (Tarea 2.3/5 del prompt de Persona 2) vive en
  `train_isolated_spectral.py`, junto con el forense por feature y el barrido
  de regularización.
