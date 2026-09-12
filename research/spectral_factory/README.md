# research/spectral_factory — el trabajo del repo del modelo, incorporado

Este directorio trae el trabajo de **Biniza** (con los placeholders de las personas 1 y 2),
desarrollado en `binivazqua/chorizos-circuits-spectral-factory`. `altur-detect` es el único
entregable del equipo, así que lo que tiene que verse vive aquí (D-A5.1).

**Snapshot de origen:** commit `9583e3a` (2026-09-12 16:29 UTC). Sin historial: el historial
fuente lleva identificadores de llamada en todos sus commits.

## Qué hay

| Carpeta | Qué es |
|---|---|
| `analisis_forense/` + `forensic_analysis.py` | Auditoría de atajos de canal antes de entrenar: ocho features de bajo nivel, figuras agregadas |
| `persona3_prosodic/` | Rama prosódica Shimmer CS3 (12 dims), `TelephonyAugmenter`, fusión tardía y contratos entre personas |
| `persona2_reference_spectral/` | Rama espectral de referencia: LFCC 60 + Δ + ΔΔ, VAD oracle, RMS a −20 dBFS, forense por feature |
| `persona1_reference_calibration/` | Platt, formato de respuesta de `/detect` y benchmark de latencia warm/cold |
| `phase1_stress_test/` | Matriz de resiliencia contra el set de robustez de seis condiciones y la validación de la augmentación |
| `HALLAZGOS_LOG.md` | Registro de auditoría con fecha: atajos de volumen y banda, fuga de motor TTS y adendas 1–4 |
| `README_ORIGINAL.md` | El README original del repo fuente |

## Qué se excluyó, y por qué

Los términos de Altur dicen *"do not redistribute"*. Un identificador de llamada **junto con su
etiqueta** es una fila de la hoja de respuestas del dataset. Por eso:

- **Se excluyeron 15 archivos por llamada** (~9 mil apariciones): `splits.csv`,
  `latents_prosodic.csv`, `latents_spectral.csv`, `latents_spectral_corrupted.csv`,
  `robustness_check.csv`, `scores_{prosodic,spectral,fusion}.csv`,
  `scores_*_calibrated.csv`, `detect_responses_fusion.jsonl`, `latents_prosodic_augmented.csv`,
  `scores_por_condicion.csv` y `scores_prosodica_augmentada_vs_zip_real.csv`.
- **Se sanearon 2 archivos**: los ejemplos de `persona3_prosodic/CONTRATOS.md` y el campo
  `anon_id` de `latency_benchmark.json` pasan a `call_EXAMPLE000N`.
- **Se conservan todos los agregados** que no llevan ids: matrices de ablación y de resiliencia,
  deltas, resumen de robustez, resumen de calibración y figuras.

`tests/test_research_sin_ids.py` falla si vuelve a aparecer un identificador bajo este directorio.

## Cómo se relaciona con el arnés

- **No entra a la imagen** (`.dockerignore`) y no pasa por Ruff: es código de investigación. Usa
  librosa, scikit-learn y Parselmouth (GPL-3).
- **Sus cifras no se citan como generalización.** Usan `GroupKFold` sobre las 353 llamadas, así
  que `val` queda contaminado (D-A3.1). La cifra de la rama prosódica (0.839) es **la rama
  aislada**, no la tubería.
- Los contratos de frontera están en `docs/MODEL_HANDOFF.md`, y la especificación de portado de
  LFCC y Shimmer en `docs/PORT_SPEC_SPECTRAL_FACTORY.md`.
- Las seis condiciones de robustez se portan como **perturbaciones v2** del arnés, con RNG por
  audio (D-A5.3). `corruption.py` usa `np.random.normal` global en el ruido, así que su semilla no
  lo controla.

## Correrlo

Los scripts buscan `audio/`, `turns/` y `manifest.csv` en la raíz de **este** directorio
(`Path(__file__).parent.parent`). Para correrlos, enlaza el dataset aquí sin commitearlo
(`.gitignore` ya excluye `audio/`):

```bash
cd research/spectral_factory
ln -s ../../data/audio audio && ln -s ../../data/turns turns && ln -s ../../data/manifest.csv manifest.csv
pip install librosa scipy numpy pandas scikit-learn soundfile praat-parselmouth matplotlib
python -m persona3_prosodic.splits   # regenera splits.csv localmente; no se commitea
```
