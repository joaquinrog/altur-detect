"""Extractor acustico A2 minimo, solo NumPy y sobre un ``AudioExample``.

Las features espectrales se calculan sobre cada ventana normalizada a RMS unitario.
El RMS absoluto se conserva unicamente como diagnostico de cadena.
"""

from __future__ import annotations

import numpy as np

from ..registry import extractors
from ..types import AudioExample, ExtractorResult

FRAME = 256
HOP = 128
BANDS = ((0, 300), (300, 1000), (1000, 2000), (2000, 3000), (3000, 3400), (3400, 4001))


def _names(prefix: str) -> tuple[str, ...]:
    return (
        *(f"{prefix}.band_{lo}_{hi}_norm" for lo, hi in BANDS),
        f"{prefix}.spectral_flatness_log",
        f"{prefix}.centroid_hz",
        f"{prefix}.rolloff85_hz",
        f"{prefix}.spectral_flux",
        f"{prefix}.energy_log_iqr",
    )


def _diagnostic_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.raw_rms",
        f"{prefix}.raw_rms_log_iqr",
        f"{prefix}.frac_sub300",
        f"{prefix}.frac_over3400",
        f"{prefix}.active_fraction",
        f"{prefix}.n_frames",
        f"{prefix}.n_active_frames",
        f"{prefix}.is_silent",
        f"{prefix}.channel_available",
        f"{prefix}.short_audio",
    )


FEATURE_ORDER = _names("a2.acoustic.ch0")
DIAGNOSTIC_ORDER = _diagnostic_names("a2.acoustic.ch0")
SPEECH_FEATURE_ORDER = _names("a2.acoustic.ch0.speech")
SILENCE_FEATURE_ORDER = _names("a2.acoustic.ch0.silence")


def _frames(signal: np.ndarray) -> tuple[np.ndarray, bool]:
    short = signal.size < FRAME
    if signal.size == 0:
        return np.zeros((1, FRAME), dtype=np.float64), True
    n = 1 + max(0, signal.size - FRAME + HOP - 1) // HOP
    starts = HOP * np.arange(n)
    out = np.zeros((n, FRAME), dtype=np.float64)
    for i, start in enumerate(starts):
        chunk = signal[start : start + FRAME]
        out[i, : chunk.size] = chunk
    return out, short


def _extract(ex: AudioExample, channel: int) -> ExtractorResult:
    prefix = f"a2.acoustic.ch{channel}"
    names = _names(prefix)
    diag_names = _diagnostic_names(prefix)
    try:
        signal = np.nan_to_num(
            np.asarray(ex.channel(channel), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        available = True
    except ValueError:
        signal = np.zeros(0, dtype=np.float64)
        available = False

    raw_rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    frames, short = _frames(signal)
    windowed = frames * np.hanning(FRAME)
    frame_rms = np.sqrt(np.mean(windowed * windowed, axis=1))
    active = frame_rms > max(1e-8, 0.1 * float(frame_rms.max(initial=0.0)))
    if not available or not np.any(active):
        values = np.zeros(len(names), dtype=np.float64)
        raw_log_iqr = 0.0
        sub300 = over3400 = 0.0
        is_silent = 1.0
        active_count = 0
    else:
        selected = windowed[active]
        normalized = selected / frame_rms[active, None]
        spectrum = np.abs(np.fft.rfft(normalized, axis=1)) ** 2
        total = np.maximum(spectrum.sum(axis=1), 1e-12)
        power = spectrum / total[:, None]
        freqs = np.fft.rfftfreq(FRAME, 1 / ex.sr)
        band_values = [float(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1).mean()) for lo, hi in BANDS]
        band_total = max(sum(band_values), 1e-12)
        band_values = [value / band_total for value in band_values]
        flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / np.maximum(
            power.mean(axis=1), 1e-12
        )
        centroid = (power * freqs[None, :]).sum(axis=1)
        cumulative = np.cumsum(power, axis=1)
        rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
        flux = np.sqrt(np.mean(np.diff(power, axis=0) ** 2, axis=1)) if len(power) > 1 else np.zeros(1)
        log_energy = np.log(np.maximum(frame_rms[active], 1e-12) / np.median(frame_rms[active]))
        raw_log_iqr = float(np.subtract(*np.percentile(np.log(np.maximum(frame_rms[active], 1e-12)), [75, 25])))
        sub300 = float(power[:, freqs < 300].sum(axis=1).mean())
        over3400 = float(power[:, freqs > 3400].sum(axis=1).mean())
        values = np.asarray(
            (*band_values, float(np.log(np.maximum(flatness, 1e-12)).mean()), float(centroid.mean()),
             float(rolloff.mean()), float(flux.mean()), float(np.subtract(*np.percentile(log_energy, [75, 25])))),
            dtype=np.float64,
        )
        is_silent = 0.0
        active_count = int(active.sum())

    features = dict(zip(names, values, strict=True))
    diagnostics = dict(zip(diag_names, (
        raw_rms, raw_log_iqr, sub300, over3400, active_count / len(frames),
        float(len(frames)), float(active_count), is_silent, float(available), float(short),
    ), strict=True))
    return features, diagnostics


def _extract_region(ex: AudioExample, *, region: str) -> ExtractorResult:
    """Extract one explicit VAD-defined ch0 region without level-based frame selection."""
    if ex.seg is None or ex.seg.is_oracle:
        raise ValueError("la variante regional requiere segmentacion recalculable")
    prefix = f"a2.acoustic.ch0.{region}"
    names = _names(prefix)
    signal = np.nan_to_num(np.asarray(ex.ch0, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    speech_mask = ex.seg.speech_mask(0, signal.size, ex.sr)
    selected_mask = speech_mask if region == "speech" else ~speech_mask
    frames, short = _frames(signal)
    mask_frames, _ = _frames(selected_mask.astype(np.float64))
    selected = np.any(mask_frames != 0.0, axis=1)
    n_region_frames = int(selected.sum())
    raw_rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0

    if not n_region_frames:
        values = np.zeros(len(names), dtype=np.float64)
        raw_log_iqr = sub300 = over3400 = 0.0
        empty_region = 1.0
    else:
        # Zeroing outside the supplied mask makes the region boundary explicit; no activity
        # threshold is inferred from the audio in these control extractors.
        region_frames = frames[selected] * mask_frames[selected]
        windowed = region_frames * np.hanning(FRAME)
        frame_rms = np.sqrt(np.mean(windowed * windowed, axis=1))
        normalized = windowed / np.maximum(frame_rms[:, None], 1e-12)
        spectrum = np.abs(np.fft.rfft(normalized, axis=1)) ** 2
        total = np.maximum(spectrum.sum(axis=1), 1e-12)
        power = spectrum / total[:, None]
        freqs = np.fft.rfftfreq(FRAME, 1 / ex.sr)
        band_values = [float(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1).mean()) for lo, hi in BANDS]
        band_total = max(sum(band_values), 1e-12)
        band_values = [value / band_total for value in band_values]
        flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / np.maximum(
            power.mean(axis=1), 1e-12
        )
        centroid = (power * freqs[None, :]).sum(axis=1)
        cumulative = np.cumsum(power, axis=1)
        rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
        flux = np.sqrt(np.mean(np.diff(power, axis=0) ** 2, axis=1)) if len(power) > 1 else np.zeros(1)
        log_rms = np.log(np.maximum(frame_rms, 1e-12))
        raw_log_iqr = float(np.subtract(*np.percentile(log_rms, [75, 25])))
        log_energy = log_rms - np.median(log_rms)
        sub300 = float(power[:, freqs < 300].sum(axis=1).mean())
        over3400 = float(power[:, freqs > 3400].sum(axis=1).mean())
        values = np.asarray(
            (*band_values, float(np.log(np.maximum(flatness, 1e-12)).mean()), float(centroid.mean()),
             float(rolloff.mean()), float(flux.mean()), float(np.subtract(*np.percentile(log_energy, [75, 25])))),
            dtype=np.float64,
        )
        empty_region = 0.0

    features = dict(zip(names, values, strict=True))
    diagnostics = {
        f"{prefix}.raw_rms": raw_rms,
        f"{prefix}.raw_rms_log_iqr": raw_log_iqr,
        f"{prefix}.frac_sub300": sub300,
        f"{prefix}.frac_over3400": over3400,
        f"{prefix}.n_region_frames": float(n_region_frames),
        f"{prefix}.empty_region": empty_region,
        f"{prefix}.short_audio": float(short),
        f"{prefix}.mask_source": ex.seg.source,
    }
    return features, diagnostics


@extractors.register("a2.acoustic.ch0", version=1, channels=(0,), needs_seg=False,
                     license="BSD-3-Clause", product_safe=True, budget_ms=150)
def extract(ex: AudioExample) -> ExtractorResult:
    return _extract(ex, 0)


@extractors.register("a2.acoustic.ch1", version=1, channels=(1,), needs_seg=False,
                     license="BSD-3-Clause", product_safe=True, budget_ms=150)
def extract_ch1(ex: AudioExample) -> ExtractorResult:
    return _extract(ex, 1)


@extractors.register("a2.acoustic.ch0.speech", version=1, channels=(0,), needs_seg=True,
                     license="BSD-3-Clause", product_safe=True, budget_ms=150)
def extract_speech(ex: AudioExample) -> ExtractorResult:
    return _extract_region(ex, region="speech")


@extractors.register("a2.acoustic.ch0.silence", version=1, channels=(0,), needs_seg=True,
                     license="BSD-3-Clause", product_safe=True, budget_ms=150)
def extract_silence(ex: AudioExample) -> ExtractorResult:
    return _extract_region(ex, region="silence")
