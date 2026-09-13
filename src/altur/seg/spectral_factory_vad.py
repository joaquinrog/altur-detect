"""VAD por energía con piso de ruido adaptativo, portado del freeze del repo del modelo.

Fuente: `binivazqua/chorizos-circuits-spectral-factory@cd028c7`, `frozen_model/simple_vad.py`.
Es el VAD con el que se entrenó y midió C2 en D-A7.3, así que se porta **tal cual**: mismos
parámetros, misma aritmética por frame y mismo redondeo de los bordes. Cambiarlo aquí sin volver
a medir rompería la paridad entre lo medido y lo servido.

Solo segmenta el canal 0, que es como lo usa `spectral_factory.lfcc@1`, su único consumidor.
NumPy puro.
"""

from __future__ import annotations

import numpy as np

from ..registry import turn_sources
from ..types import AudioExample, Segmentation, Turn

_PARAMS = {
    "frame_ms": 30.0,
    "hop_ms": 10.0,
    "noise_percentile": 15.0,
    "margin_db": 12.0,
    "min_speech_ms": 150.0,
    "min_gap_ms": 200.0,
    "channels": [0],
}
# Frames por bloque al calcular energías. Solo acota memoria (una llamada de 273 s son ~27 k
# frames); no cambia el resultado.
_BLOCK = 4096


def frame_energies_db(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """dB por frame con la fórmula exacta del original: 20·log10(sqrt(mean(x²) + 1e-12) + 1e-12)."""
    x = np.asarray(audio, dtype=np.float64)
    n_frames = 1 + (len(x) - frame_len) // hop_len
    windows = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len][:n_frames]
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(0, n_frames, _BLOCK):
        block = np.ascontiguousarray(windows[i : i + _BLOCK]) ** 2
        rms = np.sqrt(block.mean(axis=1) + 1e-12)
        out[i : i + _BLOCK] = 20 * np.log10(rms + 1e-12)
    return out


def detect_speech_segments(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Tramos de habla (inicio, fin) en segundos, idénticos a `simple_vad.detect_speech_segments`."""
    frame_ms, hop_ms = _PARAMS["frame_ms"], _PARAMS["hop_ms"]
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if len(audio) < frame_len:
        return []

    energies_db = frame_energies_db(audio, frame_len, hop_len)
    n_frames = len(energies_db)
    threshold_db = np.percentile(energies_db, _PARAMS["noise_percentile"]) + _PARAMS["margin_db"]
    is_speech = energies_db > threshold_db

    # Tiempo del CENTRO de cada frame, como el original.
    frame_times = (np.arange(n_frames) * hop_len + frame_len / 2) / sr

    edges = np.diff(np.concatenate(([False], is_speech, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    half = frame_ms / 2000
    segments = [(frame_times[s] - half, frame_times[e] + half) for s, e in zip(starts, stops)]
    if not segments:
        return []

    merged = [segments[0]]
    min_gap_s = _PARAMS["min_gap_ms"] / 1000
    for s, e in segments[1:]:
        if s - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    min_speech_s = _PARAMS["min_speech_ms"] / 1000
    return [
        (float(round(max(0.0, s), 4)), float(round(e, 4)))
        for s, e in merged
        if (e - s) >= min_speech_s
    ]


@turn_sources.register("spectral_factory.energy_adaptive", version=1, is_oracle=False, **_PARAMS)
def energy_adaptive(ex: AudioExample) -> Segmentation:
    """Turnos del canal 0 con el VAD del freeze `cd028c7`."""
    turns = tuple(Turn(0, s, e) for s, e in detect_speech_segments(ex.ch0, ex.sr))
    return Segmentation(turns, source="spectral_factory.energy_adaptive@1", params=dict(_PARAMS))
