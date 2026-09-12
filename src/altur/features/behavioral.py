"""Extractor conductual mínimo basado únicamente en ``AudioExample.seg``.

Un turno es la unión de segmentos consecutivos del mismo canal si se tocan o se
solapan. La latencia de una respuesta es ``agent.start - caller.end`` para el
primer turno del agente que empieza después del caller; los solapes no son
respuestas. Un barge-in es un turno del caller que empieza dentro de un turno
del agente y solo cuenta si ya hubo un turno del agente antes en la llamada.
"""

from __future__ import annotations

import math

import numpy as np

from ..registry import extractors
from ..types import AudioExample, ExtractorResult, Turn


def _turns(ex: AudioExample) -> tuple[tuple[Turn, ...], int]:
    """Recorta a la duración del audio, descarta intervalos inválidos y fusiona."""
    if ex.seg is None:
        return (), 0
    valid: list[Turn] = []
    invalid = 0
    for turn in ex.seg.turns:
        if turn.channel not in (0, 1) or not all(
            math.isfinite(value) for value in (turn.start, turn.end)
        ):
            invalid += 1
            continue
        start = max(0.0, turn.start)
        end = min(ex.duration_s, turn.end)
        if end <= start:
            invalid += 1
            continue
        valid.append(Turn(turn.channel, start, end))

    merged: list[Turn] = []
    for turn in sorted(valid, key=lambda item: (item.start, item.end, item.channel)):
        if merged and merged[-1].channel == turn.channel and turn.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = Turn(previous.channel, previous.start, max(previous.end, turn.end))
        else:
            merged.append(turn)
    return tuple(merged), invalid


@extractors.register(
    "behavioral",
    version=1,
    channels=(0, 1),
    needs_seg=True,
    license="BSD-3-Clause",
    product_safe=True,
    budget_ms=10,
)
def extract(ex: AudioExample) -> ExtractorResult:
    features = {"behavioral.lat_med": 0.0, "behavioral.caller_bargein_rate": 0.0}
    diagnostics: dict[str, int | bool] = {
        "segmentation_missing": ex.seg is None,
        "mono_input": ex.is_mono,
        "n_invalid_turns": 0,
        "n_caller_turns": 0,
        "n_agent_turns": 0,
        "n_responses": 0,
        "n_bargein_opportunities": 0,
        "n_bargeins": 0,
    }
    turns, invalid = _turns(ex)
    diagnostics["n_invalid_turns"] = invalid
    if ex.seg is None:
        return features, diagnostics

    callers = tuple(turn for turn in turns if turn.channel == 0)
    agents = tuple(turn for turn in turns if turn.channel == 1)
    diagnostics["n_caller_turns"] = len(callers)
    diagnostics["n_agent_turns"] = len(agents)

    latencies: list[float] = []
    for caller in callers:
        response = next((agent for agent in agents if agent.start >= caller.end), None)
        if response is not None:
            latencies.append(response.start - caller.end)
    if latencies:
        features["behavioral.lat_med"] = float(np.median(latencies))
    diagnostics["n_responses"] = len(latencies)

    for caller in callers:
        prior_agent = any(agent.start < caller.start for agent in agents)
        if not prior_agent:
            continue
        diagnostics["n_bargein_opportunities"] += 1
        if any(agent.start <= caller.start < agent.end for agent in agents):
            diagnostics["n_bargeins"] += 1
    opportunities = diagnostics["n_bargein_opportunities"]
    if opportunities:
        features["behavioral.caller_bargein_rate"] = float(
            diagnostics["n_bargeins"] / opportunities
        )
    return features, diagnostics
