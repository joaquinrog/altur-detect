# MODEL_HANDOFF — el contrato entre `altur-detect` y el repo del modelo

> **Qué es esto.** La otra mitad del equipo construye el modelo en
> `binivazqua/chorizos-circuits-spectral-factory`. Este repo es el arnés: evalúa, calibra,
> empaqueta y sirve. Este documento fija los formatos exactos de la frontera, para que nadie
> renegocie columnas a las tres de la mañana.
>
> Es la contraparte de su `persona3_prosodic/CONTRATOS.md`, que ya nombra a Joaquín como
> consumidor de su calibración. Esto responde qué se acepta y en qué forma.
>
> Convención epistémica FACT / OBS / INF / UNK, como todo en este repo.

---

## 0. La frase corta

**Ustedes entregan features. Nosotros ponemos los folds, la calibración, el umbral y el prior.**

No es burocracia: es la única forma de que el número que reportemos signifique algo fuera de
este dataset.

---

## 1. Qué entregan — el camino de menor fricción

Un CSV por familia de features, con la forma que **ya producen** sus `build_*.py`:

```
anon_id,label,<feature_1>,<feature_2>,...
call_EXAMPLE0001,synthetic,0.412,-1.883,...
```

**La única diferencia con lo que producen hoy: sin la columna `fold`.**

Los folds los pone nuestro protocolo. Si el CSV trae `fold`, lo ignoramos y lo decimos en el
reporte — no lo usamos ni en silencio ni por accidente.

Reglas del archivo:

| Regla | Por qué |
|---|---|
| `anon_id` tal como viene del manifest oficial | Es la llave de unión. No lo renombren |
| `label` ∈ {`human`, `synthetic`} | `synthetic` = 1 es la clase positiva (D-A1.1) |
| Toda columna que no sea `anon_id` ni `label` se trata como feature numérica | Igual que hace su `fusion_ablation.py` |
| Sin NaN, sin inf | Nuestro `nan_policy` es `reject`. Un NaN aborta la corrida, no se imputa en silencio |
| **El orden de columnas es el contrato** | Se congela en un YAML de `configs/experiments/`. Si el orden cambia entre corridas, los resultados no son comparables y la caché no lo detecta |

> 🔴 **Lo que más nos urge saber de ustedes:** si el orden de columnas de su `build_*.py` sale
> de un `dict`, un `glob` o un `set`, no es estable entre corridas. Díganlo y lo congelamos
> nosotros; es más barato que descubrirlo el domingo.

### Si prefieren entregar scores en vez de features

Se acepta, con una condición: **regenerados bajo `official_v1`**, con los folds que nosotros
congelamos. Los scores actuales no sirven — ver §3.

Formato, el mismo suyo: `anon_id,label,branch,score`, con `score` = `P(synthetic)` cruda
out-of-fold y `branch` documentado.

**Preferimos features.** Con features podemos correr ablaciones, controles de confound y la
curva de truncación. Con scores solo podemos calibrar.

---

## 2. Qué hacemos nosotros con eso

1. **Registrar la familia como extractor** del arnés, con su metadata obligatoria
   (`channels`, `needs_seg`, `license`, `product_safe`, `budget_ms`).
2. **Re-correr bajo `official_v1`** — fit sobre las 282 de `train`, cross-fitting anidado con
   grupos, escalado y modelo **dentro de cada fold**.
3. **Ablación**: leave-one-branch-out y leave-one-extractor-out, para saber qué aporta cada
   familia y no solo cuánto suma el conjunto.
4. **Controles de confound** (§5): ch1-only y silence-only sobre sus features acústicas.
5. **Calibración, umbral y prior** dentro del cross-fitting.
6. **Bundle** con manifest, orden de features, hashes y `requirements.lock`.

El artefacto que devolvemos: un AUC outer-OOF con intervalo, que **sí** es citable, y una lista
explícita de qué claims sobreviven a los controles.

---

## 3. Por qué sus números actuales no son transportables

**OBS.** Su `CONTRATOS.md` §1 declara textualmente que *"el manifest.csv de Altur ya no aplica
aquí"*: usan `GroupKFold(n_splits=5)` sobre las **353** llamadas, agrupando por `anon_id`.

Eso es exactamente nuestro protocolo `pooled5_v1`, que `protocol.py` marca
`val_contaminated=True`. Sus cifras publicadas hoy:

| escenario | AUC | EER % | n | features |
|---|---:|---:|---:|---:|
| prosódica sola | 0.8283 | 25.47 | 353 | 12 |
| espectral sola | **0.999967** | 0.58 | 353 | 120 |
| fusión | 0.999934 | 0.83 | 353 | 132 |

Tres consecuencias, ninguna opinable:

1. **No son comparables** con un AUC outer-OOF de `official_v1`. Distinto protocolo, distinto n,
   distinta contaminación.
2. **Su `platt_params_production.json` está ajustado sobre filas que incluyen `val`.** No lo
   usamos. La calibración se rehace de cero.
3. **Un AUC de 0.99997 no es un resultado, es la firma de un confound.** Su propio README ya
   marca dos atajos (volumen y ancho de banda) más una alerta abierta de fuga del motor TTS.

**Esto no es una crítica.** Agrupar sobre las 353 es razonable para iterar rápido cuando el set
de evaluación real es oculto — nos ahorra tiempo que ustedes ya hayan explorado el espacio. Lo
que no se puede es citar esos números como generalización delante de los jueces.

**Y coincidimos, por tres vías independientes:**

| Vía | Qué encontró |
|---|---|
| Su audit forense | Piso de ruido y planitud espectral del silencio separan las clases |
| Nuestro D-A1.6 | El sintético corta en seco a ~3400 Hz; el humano llega a Nyquist |
| Nuestro D-A2.3 | `mulaw_roundtrip` es un **no-op bit-exacto** sobre 282/282 — el corpus ya está cuantizado en μ-law |

---

## 4. La frontera de licencia

**Ninguna feature que dependa de Parselmouth u openSMILE entra al bundle.** No es una preferencia:
`registry.audit_product_safety()` la rechaza, y el empaquetado falla.

- `praat-parselmouth` — GPL-3. Meterlo a la imagen obliga a licenciar todo bajo GPL-3 y tumba el
  argumento de Feasibility (*"¿podría un banco desplegar esto?"*), que es lo que sostiene una
  imagen de inferencia con cinco dependencias.
- openSMILE — licencia de investigación de audEERING. No comercial.

**OBS 🔴.** Su rama prosódica **ya importa Parselmouth en el camino de inferencia**, desde el
commit *"Fix latency bottleneck: replace pyin with parselmouth"*. Es una decisión forzada
pendiente, no un riesgo hipotético. Detalle y las tres salidas en
[`LICENSE_AUDIT.md`](LICENSE_AUDIT.md).

**OBS.** El argumento de latencia que lo motivó está viejo. Su `CONTRATOS.md` cita ~7.5 s, pero
su propio `latency_benchmark.json` mide **181 ms warm** y 1291 ms cold sobre 150 s. El objetivo
de <1 s ya se cumple en warm — hay presupuesto para reimplementar shimmer sin Parselmouth.

Una familia con `product_safe=False` **sí se puede registrar y evaluar**: sirve como comparador
de investigación. Solo no se empaqueta.

---

## 5. Los controles que vamos a correr sobre sus features

No son un examen. Son el argumento más fuerte que tiene el equipo para el pitch, y su propio
README los pide.

| Control | Qué mide | Se aplica a |
|---|---|---|
| **ch1-only** | Las mismas features sobre el canal del **agente**, con las mismas etiquetas. El agente es el mismo TTS en ambas clases: si separa, separa el canal | Solo features acústicas |
| **silence-only** | Solo tramos sin voz. Separación fuerte ⇒ canal o cadena de generación, no fisiología | Features acústicas |
| **`highpass_300` / `lowpass_3400`** | Cuánto AUC sobrevive al igualar las bandas | Todas |
| **`behavior_scramble`** | Revuelve tiempos, deja el audio intacto | Features conductuales |

**Ningún control veta por un umbral inventado.** Lo que produce un control positivo es *qué
conclusiones quedan invalidadas*, no un número escondido. Se reportan con intervalo aunque salgan
negativos.

Nuestro piso de referencia, para que sepan contra qué se compara: **ch1 da AUC OOF 0.6520** con
features acústicas mínimas y las mismas etiquetas. Eso es confound puro, no ruido.

---

## 6. Lo que nunca cruza la frontera

| No cruza | Por qué |
|---|---|
| `anon_id` en reportes o figuras | Términos del dataset: no identificar |
| Audio, fragmentos, espectrogramas de una llamada | No redistribuir. Solo figuras agregadas |
| `manifest.csv` o cualquier derivado con etiqueta + id | Es dataset |

> 🔴 **Pendiente de su lado.** `binivazqua/chorizos-circuits-spectral-factory` es **público**, y
> `persona3_prosodic/data/splits.csv`, `scores_*.csv` y `latents_*.csv` llevan `anon_id` + etiqueta.
> Nosotros no commiteamos `groups_v1.csv` por exactamente esta razón (D-A1.4), con el repo aún
> privado. Dos salidas: repo privado hasta el cierre, o esos CSV fuera del historial.
> No es un problema técnico, es de términos del dataset, y corre mientras esté arriba.

---

## 7. Cómo se enchufa, en concreto

Una familia de features entra al arnés como un extractor registrado desde su propio módulo.
**Nadie edita `registry.py`** — es archivo de edición exclusiva del integrador.

```python
from altur.registry import extractors
from altur.types import AudioExample, ExtractorResult

@extractors.register(
    "sf.spectral.lfcc", version=1,
    channels=(0,), needs_seg=False,
    license="BSD-3-Clause", product_safe=True, budget_ms=200,
)
def extract(ex: AudioExample) -> ExtractorResult:
    # `ex` es lo ÚNICO que se recibe. Ni id, ni etiqueta, ni split, ni procedencia.
    return features, diagnostics
```

- `features` — `dict[str, float]`, claves prefijadas con el namespace, **en el orden del
  contrato**, finitas, ni bool.
- `diagnostics` — todo lo que ayude a auditar y no sea feature (RMS crudo, fracción activa,
  número de frames). **Obligatorio**: si un diagnóstico difiere > 10 % entre clases, el arnés
  marca la familia como `confounded`.

Después: un YAML en `configs/experiments/` congelando `feature_order`, y
`python scripts/check_extractor.py sf.spectral.lfcc@1` antes de mandarlo.

**Regla 3, la que más se rompe:** un extractor recibe `AudioExample` y nada más. Si necesitan
metadata para calcular una feature, la feature está mal.

---

## 8. Lo que necesitamos de ustedes, en orden

1. **Confirmar si el orden de columnas de sus `build_*.py` es determinista.** Es lo único que
   puede invalidar una corrida entera en silencio.
2. **Un CSV de latentes por familia sin la columna `fold`.**
3. **Decir qué familias dependen de Parselmouth**, para marcarlas `product_safe=False` y que
   Joaquín pueda cerrar la licencia.
4. **Sacar los `anon_id` del repo público**, o volverlo privado hasta el cierre.

Nada de esto bloquea que sigan iterando. Los cuatro se responden en una tarde.
