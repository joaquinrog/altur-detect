"""VADs pre-registrados; no leen anotaciones ni seleccionan parámetros."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..registry import turn_sources
from ..types import AudioExample, Segmentation, Turn

_FRAME_MS = 20
_HOP_MS = 10
_ENERGY_PARAMS = {
    "frame_ms": _FRAME_MS,
    "hop_ms": _HOP_MS,
    "threshold_db_above_noise_floor": 12,
    "pad_ms": 30,
    "merge_gap_ms": 450,
    "min_length_ms": 200,
}
_WEBRTC_PARAMS = {"mode": 2, "frame_ms": 20, "padding_ms": 300, "hysteresis": 0.90}


def _runs(mask: Iterable[bool]) -> list[tuple[int, int]]:
    """Devuelve intervalos [inicio, fin) de valores verdaderos."""
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, index + 1))
    return result


def _postprocess(
    flags: np.ndarray, *, frame_samples: int, hop_samples: int, n_samples: int, pad_samples: int,
    merge_gap_samples: int, min_length_samples: int,
) -> list[tuple[int, int]]:
    intervals = [
        (max(0, start * hop_samples - pad_samples), min(n_samples, end * hop_samples + frame_samples + pad_samples))
        for start, end in _runs(flags)
    ]
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= min_length_samples]


def _turns_for_channels(ex: AudioExample, segment_channel) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    for channel in (0, 1):
        if channel == 1 and ex.is_mono:
            continue
        turns.extend(Turn(channel, start / ex.sr, end / ex.sr) for start, end in segment_channel(ex.channel(channel)))
    return tuple(turns)


@turn_sources.register("energy", version=1, is_oracle=False, **_ENERGY_PARAMS)
def energy(ex: AudioExample) -> Segmentation:
    """RMS VAD con umbral relativo al percentil 10 de energía por canal."""
    frame_samples = ex.sr * _FRAME_MS // 1000
    hop_samples = ex.sr * _HOP_MS // 1000

    def segment_channel(samples: np.ndarray) -> list[tuple[int, int]]:
        if len(samples) < frame_samples:
            return []
        starts = np.arange(0, len(samples) - frame_samples + 1, hop_samples)
        frames = np.stack([samples[start : start + frame_samples] for start in starts])
        db = 20 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-6))
        noise_floor = float(np.percentile(db, 10))
        flags = db >= noise_floor + _ENERGY_PARAMS["threshold_db_above_noise_floor"]
        return _postprocess(
            flags,
            frame_samples=frame_samples,
            hop_samples=hop_samples,
            n_samples=len(samples),
            pad_samples=ex.sr * _ENERGY_PARAMS["pad_ms"] // 1000,
            merge_gap_samples=ex.sr * _ENERGY_PARAMS["merge_gap_ms"] // 1000,
            min_length_samples=ex.sr * _ENERGY_PARAMS["min_length_ms"] // 1000,
        )

    return Segmentation(_turns_for_channels(ex, segment_channel), source="energy@1", params=dict(_ENERGY_PARAMS))


@turn_sources.register("webrtc", version=1, is_oracle=False, **_WEBRTC_PARAMS)
def webrtc(ex: AudioExample) -> Segmentation:
    """WebRTC VAD modo 2 y la histéresis fija del baseline."""
    try:
        import webrtcvad
    except ImportError as exc:  # pragma: no cover - cubierto por la dependencia train
        raise RuntimeError(
            "webrtcvad es necesario para webrtc@1; instala las dependencias de entrenamiento "
            "con `pip install '.[train]'`"
        ) from exc

    frame_samples = ex.sr * _WEBRTC_PARAMS["frame_ms"] // 1000
    padding_frames = _WEBRTC_PARAMS["padding_ms"] // _WEBRTC_PARAMS["frame_ms"]
    required = int(np.floor(padding_frames * _WEBRTC_PARAMS["hysteresis"])) + 1

    def segment_channel(samples: np.ndarray) -> list[tuple[int, int]]:
        if len(samples) < frame_samples:
            return []
        pcm = np.clip(samples, -1, 1)
        pcm = np.rint(pcm * 32767).astype("<i2")
        vad = webrtcvad.Vad(_WEBRTC_PARAMS["mode"])
        voiced = np.array(
            [
                vad.is_speech(pcm[start : start + frame_samples].tobytes(), ex.sr)
                for start in range(0, len(pcm) - frame_samples + 1, frame_samples)
            ],
            dtype=bool,
        )
        # Se exige una ventana completa: 14/15 frames, equivalente a >90%.
        # El ring buffer se conserva dentro del turno para aplicar el padding retrospectivo.
        turns: list[tuple[int, int]] = []
        ring: list[bool] = []
        active = False
        start = 0
        for index, is_voiced in enumerate(voiced):
            ring.append(bool(is_voiced))
            if len(ring) > padding_frames:
                ring.pop(0)
            if not active and len(ring) == padding_frames and sum(ring) >= required:
                active = True
                start = (index - padding_frames + 1) * frame_samples
                ring.clear()
            elif active and len(ring) == padding_frames and len(ring) - sum(ring) >= required:
                active = False
                turns.append((start, min(len(samples), (index + 1) * frame_samples)))
                ring.clear()
        if active:
            turns.append((start, len(samples)))
        return turns

    return Segmentation(_turns_for_channels(ex, segment_channel), source="webrtc@1", params=dict(_WEBRTC_PARAMS))


@turn_sources.register("oracle", version=1, is_oracle=True)
def oracle(ex: AudioExample) -> Segmentation:
    """Expone solo la segmentación oracle ya inyectada por el runner de entrenamiento."""
    if ex.seg is None or not ex.seg.is_oracle:
        raise ValueError("oracle@1 requiere una segmentación oracle ya presente en AudioExample.seg")
    return ex.seg
