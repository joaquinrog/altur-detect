# Rama Prosódica / Calidad Vocal — Persona 3

Implementa las Tareas 3.1–3.3 del plan de experimentos (Shimmer CS3(ΔΔ) vía un
pitch tracker rápido, enmascaramiento VAD, pooling → vector latente, robustez a
corrupción de canal, entrenamiento aislado y fusión tardía). Ver `CONTRATOS.md`
para los formatos exactos que conectan esto con Persona 1 (splits, calibración)
y Persona 2 (corrupción, rama espectral) sin que nadie tenga que tocar el
código del otro.

## Orden de ejecución

```bash
# 0. Genera data/splits.csv (fallback local; se reemplaza por el de Persona 1 cuando exista)
python -m persona3_prosodic.splits

# 1. Extrae el vector latente prosódico de las 353 llamadas (~10s con 10 workers)
python -m persona3_prosodic.build_prosodic_dataset --workers 10

# 2. Verifica que Shimmer sobrevive a la corrupción de canal (Tarea 4)
python -m persona3_prosodic.robustness_check

# 3. Entrena la rama prosódica aislada y genera scores para Persona 1 (Tarea 5)
python -m persona3_prosodic.train_isolated_branch

# 4. Fusión tardía + matriz de ablación (Tarea 3.3) — corre en modo "solo prosódica"
#    hasta que Persona 2 entregue data/latents_spectral.csv
python -m persona3_prosodic.fusion_ablation
```

## Archivos que produce (todos bajo `persona3_prosodic/data/`)

| Archivo | Contenido |
|---|---|
| `splits.csv` | GroupKFold de 5 folds por `anon_id` (fallback local, ver `CONTRATOS.md`) |
| `latents_prosodic.csv` | Vector latente prosódico (12 features) por llamada |
| `latents_prosodic_failures.csv` | Llamadas donde no se pudo extraer suficiente voz sonora |
| `robustness_check.csv` / `robustness_summary.csv` | Features limpias vs corrompidas + AUC antes/después (Tarea 4) |
| `scores_prosodic.csv` | Score out-of-fold por llamada, listo para calibración de Persona 1 |
| `matriz_ablacion.csv` | AUC/EER de espectral sola / prosódica sola / fusión |
| `scores_fusion.csv` | Score de fusión out-of-fold (solo si ya existe `latents_spectral.csv`) |

## Decisiones de diseño ya justificadas en el código (para defender frente a jueces)

- **parselmouth (Praat) en vez de librosa.pyin** para F0 — **revisado dos
  veces** tras el benchmark de latencia (ver `HALLAZGOS_LOG.md`, adendas 1 y 2):
  pYIN tardaba 7.4s sobre una llamada de 150s, el 99% de la latencia total del
  pipeline. Su capa probabilística/Viterbi (decidir voiced/unvoiced con
  incertidumbre) era redundante porque ya usamos `turns.json` como VAD oracle.
  Cambiar a parselmouth (autocorrelación de Praat, en C) bajó la extracción a
  ~115ms (~65x más rápido). Primer ajuste (rango 80–350 Hz, hop 20ms) costó
  AUC real (0.842→0.803, contrario a lo que se esperaba inicialmente). Con
  841ms de presupuesto de latencia todavía libres, se ensanchó el rango a
  60–450 Hz y se afinó el hop a 10ms: AUC recuperado a 0.839 (RandomForest,
  EER 23.49% — iguala o mejora el número original), a un costo de solo +23ms
  de latencia (159ms→182ms warm, muy por debajo del presupuesto de 1s).
- **Pooling estadístico (media+std+percentiles)** en vez de Global Average
  Pooling o pooling atencional — ver `pool_shimmer_series()` en
  `prosodic_features.py`. Elegido por el tamaño chico del dataset (353 llamadas).
- **`turns.json` como VAD oracle** en vez de un VAD entrenado — decisión
  explícita de fase 1, documentada como simplificación reemplazable sin tocar el
  resto del pipeline.
