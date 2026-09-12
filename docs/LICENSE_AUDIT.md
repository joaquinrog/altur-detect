# Auditoria de licencias

**Alcance:** dependencias directas declaradas en `pyproject.toml` al commit `9b4a127`. Esta matriz
permite que Joaquin elija la licencia del repo; **este documento no elige una licencia** y no es
asesoria legal.

Convencion: **FACT** = licencia declarada por el proyecto; **INF** = consecuencia operativa que
debe validar quien decida; **UNK** = falta confirmar sobre el artefacto final.

## Inferencia

| Dependencia | Restriccion | SPDX | Uso | Imagen | Estado |
|---|---|---|---|---|---|
| `numpy` | `>=1.26,<3` | `BSD-3-Clause` | calculo numerico | si | product-safe |
| `fastapi` | `>=0.115,<1` | `MIT` | API HTTP | si | product-safe |
| `uvicorn[standard]` | `>=0.30,<1` | `BSD-3-Clause` | servidor ASGI | si | product-safe; auditar transitivas del extra al congelar lock |
| `pydantic` | `>=2.7,<3` | `MIT` | validacion y modelos | si | product-safe |
| `python-multipart` | `>=0.0.9` | `Apache-2.0` | payload multipart | si | product-safe |

## Entrenamiento (`train`)

| Dependencia | Restriccion | SPDX | Imagen | Estado |
|---|---|---|---|---|
| `scipy` | `>=1.11` | `BSD-3-Clause` | no | product-safe por licencia |
| `scikit-learn` | `>=1.5` | `BSD-3-Clause` | no | product-safe por licencia; exportar inferencia a NumPy |
| `lightgbm` | `>=4.3` | `MIT` | no | product-safe por licencia |
| `pandas` | `>=2.2` | `BSD-3-Clause` | no | product-safe por licencia |
| `matplotlib` | `>=3.6` | `PSF-2.0` | no | product-safe por licencia |
| `pyyaml` | `>=6.0` | `MIT` | no | product-safe por licencia |
| `webrtcvad` | `>=2.0.10` | `MIT` | no en el Dockerfile actual | product-safe por licencia |

## Investigacion (`research`)

| Dependencia | Restriccion | SPDX/licencia | Imagen | Estado |
|---|---|---|---|---|
| `praat-parselmouth` | `>=0.4` | `GPL-3.0-or-later` | **no** | **product_safe=False**; copyleft al distribuir una obra combinada |
| `librosa` | `>=0.10` | `ISC` | no | product-safe por licencia, pero el extra completo no es product-safe |
| openSMILE | no declarado | `LicenseRef-audEERING-Research` | **no** | **product_safe=False**; solo investigacion/no comercial sin licencia comercial |

**FACT:** `praat-parselmouth` usa GPL-3. **FACT:** openSMILE usa la licencia de investigacion de
audEERING; alcanza tambien el uso indirecto de features extraidas en un producto. Ninguno puede
entrar a la imagen de inferencia. openSMILE no esta actualmente en `pyproject.toml`; se incluye
porque es un veto de producto explicito y una posible herramienta de investigacion.

## Camino experimental (`torch`)

| Dependencia | Restriccion | SPDX | Imagen | Estado |
|---|---|---|---|---|
| `torch` | `>=2.3` | `BSD-3-Clause` | no | product-safe por licencia; Camino B permanece NO-GO |

## Desarrollo (`dev`)

| Dependencia | Restriccion | SPDX | Imagen | Estado |
|---|---|---|---|---|
| `pytest` | `>=8.0` | `MIT` | no | product-safe por licencia |
| `httpx` | `>=0.27` | `BSD-3-Clause` | no | product-safe por licencia |
| `ruff` | `>=0.6` | `MIT` | no | product-safe por licencia |

## Build

| Dependencia | Restriccion | SPDX | Imagen | Estado |
|---|---|---|---|---|
| `setuptools` | `>=68` | `MIT` | solo build | product-safe por licencia |

## Lo que esta y no esta probado

- **FACT:** la matriz cubre todas las dependencias directas declaradas en `pyproject.toml`.
- **UNK:** no existe lockfile final de A4.1. Por tanto, faltan versiones exactas, transitivas,
  notices y licencias de los wheels realmente distribuidos.
- **UNK:** `uvicorn[standard]` activa dependencias transitivas opcionales segun plataforma; deben
  auditarse desde el lock final, no desde una instalacion local mutable.
- **FACT:** la licencia del dataset es independiente de la licencia del codigo. El audio y
  `manifest.csv` no se redistribuyen.
- **INF:** una licencia permisiva de una dependencia no elimina sus obligaciones de copyright y
  notice.

## 🔴 El caso no es hipotetico: el repo del modelo ya importa Parselmouth

**OBS.** La otra mitad del equipo construye el modelo en el repo publico
`binivazqua/chorizos-circuits-spectral-factory`. Su rama prosodica usa
`praat-parselmouth` **dentro del camino de inferencia** desde el commit
*"Fix latency bottleneck: replace pyin with parselmouth in prosodic branch"* (2026-09-12).

Esto cambia el estatus de todo lo de arriba. En este repo Parselmouth vive en el extra
`research`, que no se instala ni se distribuye — un riesgo evitado. En el suyo ya es una
dependencia de ejecucion de una rama que, si entra al bundle, entra a la imagen.

**INF.** Mientras esa rama sea candidata a produccion, la eleccion de licencia no es libre:
o se resuelve la dependencia, o la opcion permisiva queda descartada por construccion.

**OBS — el argumento por el que la metieron ya no se sostiene solo.** Su
`CONTRATOS.md` justifica el cambio con un pipeline de ~7.5 s contra un objetivo de <1 s. Ese
numero esta viejo: su propio `persona1_reference_calibration/data/latency_benchmark.json` mide
**181 ms en warm** y 1291 ms en cold sobre una llamada de 150 s, con 62 MB de pico. El objetivo
de <1 s ya se cumple en warm.

**Las tres salidas, y lo que cuesta cada una:**

| Salida | Consecuencia |
|---|---|
| La rama prosodica **no entra al bundle** | La imagen sigue permisiva y minima. Se pierde el eje prosodico en produccion; queda como diagnostico. Es coherente con D-A2.2, que ya degrado el eje conductual a diagnostico por otra razon. |
| **Reimplementar shimmer** sin Parselmouth | Conserva el eje y la licencia permisiva. Cuesta tiempo de ingenieria y hay que re-validar que el shimmer reimplementado mide lo mismo. El benchmark sugiere que el presupuesto de latencia da margen. |
| Aceptar **GPL-3 para todo** | Desbloquea la rama tal cual, pero obliga a distribuir todo el codigo correspondiente bajo GPL y debilita el argumento de Feasibility ("podria un banco desplegar esto") que sostiene la imagen minima. |

**Quien decide:** Joaquin. **Cuando:** antes de A4.1, porque el bundle es donde la frontera se
vuelve fisica — `registry.audit_product_safety()` rechaza `product_safe=False` y no hay forma de
empaquetar la rama sin tomar la decision primero.

## Opciones para decision de Joaquin

| Opcion viable | Consecuencia |
|---|---|
| Licenciar el core propio con `MIT`, `BSD-3-Clause` o `Apache-2.0` | Compatible con el stack productivo permisivo. Exige mantener Parselmouth/openSMILE fuera de la imagen y evitar distribuir integraciones que conviertan el core en una obra GPL o sujeta a uso no comercial. `Apache-2.0` agrega una concesion expresa de patentes; MIT/BSD son mas breves. |
| Licenciar todo el codigo distribuido como `GPL-3.0-or-later` | Simplifica la compatibilidad con Parselmouth si su integracion se distribuye como obra combinada, pero obliga a distribuir el codigo fuente correspondiente bajo GPL. **No** vuelve comercialmente utilizable openSMILE ni relicencia el dataset. |
| Separar core productivo permisivo y herramientas de investigacion GPL | Conserva una imagen permisiva y aisla Parselmouth en un paquete/proceso claramente separado. Requiere limites de distribucion, imports y documentacion verificables; openSMILE sigue siendo research-only salvo licencia comercial. |
| No distribuir el extra `research` | Reduce el riesgo de copyleft del artefacto publicado, pero no sustituye revisar si hay codigo propio derivado o imports residuales. La investigacion interna puede conservarse sin entrar al release. |

## Gate antes de elegir o publicar

1. Joaquin elige la licencia del codigo; este documento se detiene antes de esa decision.
2. A4.1 produce un lock con versiones y hashes exactos.
3. Generar SBOM y notices desde la imagen final por digest.
4. Confirmar que no existen imports, binarios, configuraciones ni features derivadas de
   `product_safe=False` dentro de la imagen.
5. Revisar compatibilidad de la opcion elegida con el contenido propio realmente distribuido.
6. Registrar la decision fuera de este documento en el archivo controlado por el integrador.
