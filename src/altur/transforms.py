"""Deterministic, diagnostic audio transforms for the narrow-band gauntlet."""

from __future__ import annotations

import numpy as np

from .registry import transforms
from .types import AudioExample

_FILTER_TAPS = 129


def _check_rng(rng: np.random.Generator | None) -> None:
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")


def _fir(ex: AudioExample, cutoff_hz: float, *, highpass: bool,
         rng: np.random.Generator | None) -> AudioExample:
    _check_rng(rng)
    n = _FILTER_TAPS
    center = n // 2
    frequency = cutoff_hz / ex.sr
    indexes = np.arange(n, dtype=np.float64) - center
    kernel = 2.0 * frequency * np.sinc(2.0 * frequency * indexes)
    kernel *= np.hamming(n)
    kernel /= kernel.sum()
    if highpass:
        kernel = -kernel
        kernel[center] += 1.0
    filtered = np.convolve(ex.ch0.astype(np.float64), kernel, mode="same")
    return AudioExample(filtered.astype(np.float32), ex.ch1, sr=ex.sr, seg=ex.seg)


def _mulaw_encode_decode(samples: np.ndarray) -> np.ndarray:
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int32)
    sign = pcm < 0
    magnitude = np.minimum(np.abs(pcm) + 132, 32635)
    exponent = np.maximum(np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 7, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = (~((sign.astype(np.int32) << 7) | (exponent << 4) | mantissa)) & 0xFF

    decoded_code = (~encoded) & 0xFF
    decoded_sign = decoded_code & 0x80
    decoded_exponent = (decoded_code >> 4) & 0x07
    decoded_mantissa = decoded_code & 0x0F
    decoded = ((decoded_mantissa << 3) + 132) << decoded_exponent
    decoded = np.where(decoded_sign != 0, 132 - decoded, decoded - 132)
    return (decoded.astype(np.float64) / 32768.0).astype(np.float32)


@transforms.register(
    "lowpass_3400", version=1, use="holdout", purpose="diagnostic",
    changes_length=False, shifts_timing=False, channels=(0,), params={"cutoff_hz": 3400, "taps": 129},
)
def lowpass_3400(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    return _fir(ex, 3400.0, highpass=False, rng=rng)


@transforms.register(
    "highpass_300", version=1, use="holdout", purpose="diagnostic",
    changes_length=False, shifts_timing=False, channels=(0,), params={"cutoff_hz": 300, "taps": 129},
)
def highpass_300(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    return _fir(ex, 300.0, highpass=True, rng=rng)


@transforms.register(
    "mulaw_roundtrip", version=1, use="holdout", purpose="diagnostic",
    changes_length=False, shifts_timing=False, channels=(0,), params={"encoding": "G.711-mu-law", "bits": 8},
)
def mulaw_roundtrip(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    _check_rng(rng)
    return AudioExample(_mulaw_encode_decode(ex.ch0), ex.ch1, sr=ex.sr, seg=ex.seg)
