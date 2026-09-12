# Contratos de datos — Rama Prosódica (Persona 3) ↔ Persona 1 / Persona 2

Esto define los formatos EXACTOS que este módulo consume y produce, para que el
trabajo de Persona 1 (splits, calibración) y Persona 2 (corrupción, rama espectral)
se pueda conectar sin que nadie tenga que renegociar columnas o firmas de función
a medio hackathon. Si tu entregable respeta el contrato de abajo, todo lo demás
(mi código) sigue funcionando sin tocarlo.

> **Estado actual (actualizado):** `data/latents_spectral.csv` y
> `data/scores_spectral.csv` YA existen — pero son un placeholder de referencia
> (LFCC + regresión logística), no el backbone real de Persona 2. Antes de citar
> cualquier AUC de la rama espectral o de la fusión en el pitch, lean
> `persona2_reference_spectral/README.md` (sección "Hallazgos"): se encontraron
> dos atajos (volumen y ancho de banda) y una alerta abierta sobre posible fuga
> de motor TTS que hace que el AUC~1.00 actual no sea confiable tal cual.

## 1. Splits (Persona 1 → todos)

Archivo: `data/splits.csv`

```
anon_id,label,fold
call_EXAMPLE0001,synthetic,0
call_EXAMPLE0002,human,3
...
```

- `fold` ∈ {0,1,2,3,4}, asignado por `GroupKFold(n_splits=5)` agrupando por
  `anon_id` sobre las 353 llamadas (no solo el split "train" del manifest
  original — el manifest.csv de Altur ya no aplica aquí, este es el split propio
  del equipo para poder correr CV interno, dado que el set oculto de evaluación
  del reto es aparte y no lo controlamos).
- Ninguna llamada puede aparecer en más de un fold (es un grouping, no un split
  aleatorio) — eso ya lo garantiza `GroupKFold`, no hay nada que verificar a mano
  más que confirmar `anon_id` único por fila.
- Mientras Persona 1 no entregue este archivo, `splits.py` en este módulo genera
  uno localmente con el mismo formato (mismo seed) para no bloquear a nadie. El
  día que Persona 1 entregue el suyo, solo se reemplaza el archivo — cero cambios
  de código en ningún otro módulo.

## 2. Corrupción de canal (Persona 2 → Persona 3, y viceversa)

Clase esperada, en cualquier archivo que la exponga (ver `corruption.py` en este
módulo para la implementación de referencia que Persona 3 usa mientras Persona 2
entrega la suya):

```python
class TelephonyAugmenter:
    def __init__(self, p: float = 0.5, snr_db_range: tuple[float, float] = (15, 25),
                 seed: int | None = None): ...

    def __call__(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Devuelve (audio_posiblemente_corrompido, meta).
        meta = {"corrupted": bool, "codec": "mu-law" | "a-law" | None}
        Debe aplicarse simétricamente: la misma probabilidad p y la misma
        degradación para audio humano y sintético — no debe recibir el label
        como argumento, por diseño, para que sea físicamente imposible que la
        corrupción dependa de la clase.
        """
```

- Persona 3 (este módulo) usa esta clase en `robustness_check.py` para medir si
  Shimmer sigue siendo discriminativo después de la corrupción. Si Persona 2
  entrega una clase con esta misma firma, solo se cambia el `import` en
  `robustness_check.py` (una línea) — el resto no se toca.
- Si la firma real de Persona 2 difiere, la solución es un wrapper de una línea
  de su lado o del mío, nunca reescribir la lógica de negocio de ninguno de los
  dos módulos.

## 3. Vectores latentes por rama (Persona 2 y Persona 3 → fusión)

Cada rama entrega un CSV con esta forma mínima:

```
anon_id,label,fold,<feature_1>,<feature_2>,...
```

- Persona 3 entrega `data/latents_prosodic.csv` (columnas de features con
  prefijo `shimmer_`, `dshimmer_`, `ddshimmer_`).
- Persona 2 entrega `data/latents_spectral.csv` (cualquier nombre de columna
  numérica es válido; `fusion_ablation.py` autodetecta todas las columnas que no
  sean `anon_id,label,fold` y las trata como features). **Por ahora este archivo
  lo llena un placeholder de referencia (`persona2_reference_spectral/`, LFCC
  60-dim), no el backbone real de Persona 2 — ver el caveat al inicio de este
  documento antes de confiar en el AUC que sale de él.**
- `fusion_ablation.py` no rompe si `latents_spectral.csv` no existe todavía: corre
  solo la ablación de la rama prosódica y avisa qué archivo falta para completar
  la matriz de 3 escenarios (espectral sola / prosódica sola / fusión). Cuando el
  AUC de esa rama (o de la fusión) supera 0.97, imprime una alerta explícita para
  no reportarlo sin el caveat.

## 4. Scores por rama (Persona 2 y Persona 3 → Persona 1, calibración)

Cada rama entrega un CSV con esta forma exacta (mismo nombre de columnas en las
tres ramas, para que Persona 1 escriba UN solo calibrador que sirva para las tres):

```
anon_id,label,fold,branch,score
call_EXAMPLE0001,synthetic,0,prosodic,0.73
```

- `score` es la salida cruda (probabilidad o logit, documentado por quien lo
  genera) del clasificador aislado de esa rama, obtenida **out-of-fold**
  (`cross_val_predict` o equivalente) para que Persona 1 pueda calibrar sin fuga
  de datos.
- `branch` ∈ {"prosodic", "spectral", "fusion"}.
- Persona 3 entrega `data/scores_prosodic.csv` desde `train_isolated_branch.py`.
- El mismo archivo, con `branch="fusion"`, lo entrega `fusion_ablation.py` una vez
  que existan ambos vectores latentes.

## 5. Calibración y respuesta final (Persona 1 → Joaquín)

**Placeholder de referencia en `persona1_reference_calibration/` (Regi no disponible
por ahora)**: calibra los `score` de cada rama con Platt scaling, produce
`platt_params_production.json` y convierte el score calibrado en el JSON exacto
que espera `POST /detect`: `{"is_synthetic": bool, "confidence": float}`. Ver
`persona1_reference_calibration/README.md` para el detalle y las decisiones de
diseño. Mismo mecanismo de reemplazo: cuando Regi entregue su propia calibración,
solo se reemplazan estos scripts (o simplemente se ignoran si ella construye algo
distinto) — no afecta a `fusion_ablation.py` ni a ninguna rama.

**Hallazgo del benchmark de latencia** (`benchmark_latency.py`): sobre una llamada
real de ~150s, el pipeline actual tarda ~7.5s totales, MUY por encima del objetivo
de <1s del equipo. El 99% del tiempo se va en la extracción de Shimmer (pYIN de
librosa) de la rama prosódica — la inferencia de los modelos (regresión logística)
toma <1ms por rama. Esto es un cuello de botella real a resolver antes de la
entrega final, no una limitación de los modelos en sí.

## Resumen de quién produce qué

| Archivo | Lo genera | Lo consume |
|---|---|---|
| `data/splits.csv` | Persona 1 (o `splits.py` como fallback local) | Todos |
| `TelephonyAugmenter` | Persona 2 (o `corruption.py` como referencia local) | Persona 2, Persona 3 (`robustness_check.py`) |
| `data/latents_prosodic.csv` | Persona 3 (`build_prosodic_dataset.py`) | `fusion_ablation.py` |
| `data/latents_spectral.csv` | Persona 2 (hoy: placeholder en `persona2_reference_spectral/`) | `fusion_ablation.py` |
| `data/scores_prosodic.csv` | Persona 3 (`train_isolated_branch.py`) | Persona 1 (calibración) |
| `data/scores_spectral.csv` | Persona 2 (hoy: placeholder) | Persona 1 (calibración) |
| `data/scores_fusion.csv` | `fusion_ablation.py` (Persona 3) | Persona 1 (calibración) |
| `data/feature_forensics_spectral.csv` | placeholder (`train_isolated_spectral.py`) | Equipo (auditoría, ver README de esa carpeta) |
| `data/corruption_ablation_spectral.csv` | placeholder (`train_isolated_spectral.py`) | Equipo (Tarea 2.3/5 de Persona 2) |
| `persona1_reference_calibration/data/platt_params_production.json` | placeholder (`calibrate.py`) | `format_detect_response.py`, Joaquín |
| `persona1_reference_calibration/data/detect_responses_<branch>.jsonl` | placeholder (`format_detect_response.py`) | Joaquín (formato final `POST /detect`) |
| `persona1_reference_calibration/data/latency_benchmark.json` | placeholder (`benchmark_latency.py`) | Equipo (cuello de botella: extracción prosódica, ~7.5s vs objetivo <1s) |
