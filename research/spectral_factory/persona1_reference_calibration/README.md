# Calibración + latencia — implementación de REFERENCIA (placeholder de Persona 1)

**Esto NO es el entregable real de Persona 1 (Regi).** Es un placeholder para
no bloquear la integración mientras ella no está disponible — mismo patrón que
`persona2_reference_spectral/` para la rama de Persona 2. Cuando Regi esté de
vuelta, puede revisar/reemplazar cualquiera de estas piezas sin que el resto
del pipeline necesite cambios, siempre que respete los mismos contratos.

## Qué resuelve (Tareas 3, 4, 5 y 6 de su prompt maestro)

| Script | Tarea | Qué hace |
|---|---|---|
| `calibrate.py` | 3 | Escala de Platt sobre `scores_<branch>.csv` (out-of-fold, mismos folds de `data/splits.csv`), reporta Brier score antes/después, grafica curva de calibración |
| `format_detect_response.py` | 6 | Convierte el score calibrado en el JSON exacto de `POST /detect` (`{"is_synthetic": bool, "confidence": float}`) |
| `benchmark_latency.py` | 5 | Mide latencia end-to-end (extracción + inferencia + fusión + calibración) en CPU sobre una llamada real de ~150s, desglosado por etapa, no solo el total |
| — | 4 (agregación por tramos de 150s) | **No aplica a este pipeline** — ver nota abajo |

## Nota sobre la Tarea 4 (agregación por tramos de 150 segundos)

El prompt original de Persona 1 asume que "los clips se analizan por tramos de
150 segundos" y pide una función que agregue varias predicciones por tramo en
una sola predicción por llamada. **Nuestro pipeline actual no trocea las
llamadas**: tanto Shimmer (persona3_prosodic) como LFCC (persona2_reference_spectral)
poolean estadísticamente sobre TODA la llamada de una vez (mean/std/percentiles),
así que ya producen un solo vector por llamada sin necesidad de ventaneo ni
agregación posterior. Esa función solo haría falta si en algún momento el
backbone real de Persona 2 (LCNN/ECAPA) impone un límite de longitud de
secuencia que obligue a trocear el audio en tramos — no es el caso hoy. Se deja
documentado aquí para que Regi no busque un bug donde no lo hay.

## Orden de ejecución

```bash
python -m persona1_reference_calibration.calibrate
python -m persona1_reference_calibration.format_detect_response --branch fusion
python -m persona1_reference_calibration.benchmark_latency
```

`calibrate.py` corre primero porque genera `platt_params_production.json`, que
los otros dos scripts consumen.

## Archivos que produce (bajo `persona1_reference_calibration/data/`)

| Archivo | Contenido |
|---|---|
| `scores_<branch>_calibrated.csv` | Scores originales + `score_calibrated` por rama |
| `calibration_curve_<branch>.png` | Curva de calibración antes/después de Platt |
| `calibration_summary.csv` | Brier score crudo vs. calibrado, por rama |
| `platt_params_production.json` | Parámetros `{A, B}` del calibrador final, listos para producción |
| `detect_responses_<branch>.jsonl` | Una línea por llamada, en el formato exacto de `POST /detect` |
| `latency_benchmark.json` | Desglose de latencia + memoria pico, sobre una llamada real de ~150s |

## Decisiones de diseño (placeholder, documentadas)

- **Platt sobre el logit del score, no sobre el score directo**: los scores de
  entrada ya son probabilidades (`predict_proba`), no logits sin acotar. Ajustar
  la regresión logística de Platt sobre `logit(score)` es la variante estándar
  cuando el input ya viene en escala [0,1] — si se ajustara directo sobre el
  score crudo, la mayoría de los casos ya estarían casi en 0 o 1 y Platt no
  tendría margen para aprender nada útil.
- **Calibración out-of-fold usando los MISMOS folds que ya existen** (`data/splits.csv`)
  en vez de un nuevo split aleatorio — así el Brier score reportado es
  honesto (no se calibra y evalúa con los mismos puntos) y consistente con el
  resto del pipeline.
- **`confidence` = probabilidad calibrada de la clase PREDICHA**, no siempre
  `P(synthetic)` — si el modelo predice `human`, confidence es `P(human)`. Es
  la convención que mejor conecta con "premiar calibración" del README del
  reto.
- **El benchmark de latencia entrena los modelos finales SIN cronometrarlos**:
  solo se mide lo que correría por cada llamada nueva en producción
  (extracción + inferencia + calibración), no el entrenamiento offline.
