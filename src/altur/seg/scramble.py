"""`behavior_scramble` — la prueba estrella del plan.

**Qué pregunta.** Altur define `synthetic` sobre el *caller*, no sobre la conversación. Si
revolvemos el comportamiento conversacional de una llamada **humana** y el detector empieza a
decir `synthetic`, entonces el eje conductual está decidiendo y la definición que estamos
optimizando no es la que pide el reto. Eso es CP2-bis, y es un bug de definición, no una
métrica que se reporta y ya.

**Por qué es un `turn_source` y no un `transform`.** El runner aplica los transforms y
**después** corre el segmentador sobre el audio transformado, descartando cualquier
`Segmentation` que el transform traiga (`runner.py:243`). Eso es correcto y es el candado
que evita arrastrar segmentaciones entre condiciones — pero significa que un transform no
puede alterar los turnos. Un segmentador sí.

**Qué hace.** Conserva el audio **bit a bit** y las duraciones de cada turno; permuta los
silencios entre turnos, por canal y de forma independiente. Eso destruye la relación temporal
entre caller y agente —latencia de respuesta y solapamientos— sin tocar una sola muestra.

La semilla se deriva del audio, así que la misma llamada produce siempre el mismo revuelto y
dos llamadas distintas producen revueltos distintos. Sin estado global, sin `random_state`
suelto.
"""

import hashlib
from itertools import pairwise

import numpy as np

from altur.registry import turn_sources
from altur.seg.vad import energy
from altur.types import AudioExample, Segmentation, Turn

_PARAMS = {
    "base": "energy@1",
    "mode": "permute_gaps",
    "preserves_audio": True,
    "preserves_durations": True,
    "seed_source": "sha256(ch0 bytes)",
}


def _seed_from_audio(ex: AudioExample) -> int:
    h = hashlib.sha256(np.asarray(ex.ch0).tobytes())
    if ex.ch1 is not None:
        h.update(np.asarray(ex.ch1).tobytes())
    return int.from_bytes(h.digest()[:8], "big")


def _scramble_channel(
    turns: list[Turn], rng: np.random.Generator, duration_s: float
) -> list[Turn]:
    """Conserva duraciones y orden; permuta los huecos, incluido el previo al primer turno."""
    if len(turns) < 2:
        return list(turns)
    ordenados = sorted(turns, key=lambda t: t.start)
    duraciones = [t.end - t.start for t in ordenados]

    huecos = [ordenados[0].start]
    for anterior, siguiente in pairwise(ordenados):
        huecos.append(max(0.0, siguiente.start - anterior.end))

    permutados = [huecos[i] for i in rng.permutation(len(huecos))]

    salida: list[Turn] = []
    cursor = 0.0
    for hueco, dur in zip(permutados, duraciones, strict=True):
        cursor += hueco
        inicio = min(cursor, max(0.0, duration_s - dur))
        salida.append(Turn(ordenados[0].channel, inicio, inicio + dur))
        cursor = inicio + dur
    return salida


@turn_sources.register("behavior_scramble", version=1, is_oracle=False, **_PARAMS)
def behavior_scramble(ex: AudioExample) -> Segmentation:
    """Turnos de `energy@1` con los silencios permutados. El audio no se toca."""
    base = energy(ex)
    rng = np.random.default_rng(_seed_from_audio(ex))
    duracion = ex.duration_s

    revueltos: list[Turn] = []
    for canal in (0, 1):
        del_canal = [t for t in base.turns if t.channel == canal]
        revueltos.extend(_scramble_channel(del_canal, rng, duracion))

    return Segmentation(
        tuple(revueltos),
        source="behavior_scramble@1",
        params=dict(_PARAMS) | {"base_source": base.source},
    )
