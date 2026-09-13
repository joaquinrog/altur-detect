"""LFCC 120-dim del repo del modelo, portado a NumPy puro (`docs/PORT_SPEC_SPECTRAL_FACTORY.md`).

Fuente: `binivazqua/chorizos-circuits-spectral-factory@cd028c7`,
`persona2_reference_spectral/lfcc_features.py`. El original usa librosa (STFT y delta) y SciPy
(DCT); aquí se reescriben esas tres operaciones para que la imagen siga sin librosa ni SciPy
(D-A4.8). Los dos UNK de la spec quedan fijados a lo que hace librosa 1.0.0, la versión con la
que se entrenó y midió C2 (D-A7.3): la STFT se centra con padding de ceros, y delta es un filtro
Savitzky-Golay con `mode="interp"`.

La equivalencia con el original se prueba en `tests/test_spectral_factory_port.py`.

**Desviación declarada.** Si el VAD no encuentra habla, el original devuelve `None` y su servidor
responde "humano, 0.5". Un extractor del arnés tiene que devolver features, así que aquí se usa
el canal completo como un solo tramo y queda marcado en los diagnósticos. Pasa con ruido blanco
(el warm-up de la API), no en las 282 llamadas de `train`.
"""

from __future__ import annotations

import math

import numpy as np

from ..registry import extractors
from ..types import AudioExample, ExtractorResult

NAMESPACE = "spectral_factory.lfcc"
FRAME_LENGTH = 160  # 20 ms a 8 kHz
HOP_LENGTH = 80  # 10 ms
N_FFT = 256
N_FILTERS = 20
N_STATIC = 20
DELTA_WIDTH = 9
# -20 dBFS sobre el habla del canal 0: quita el atajo de nivel que el equipo midió (AUC ~0.88
# solo con static0 sin normalizar). Ver la docstring del original.
TARGET_RMS = 10 ** (-20.0 / 20.0)


def _dim_names() -> list[str]:
    return (
        [f"static{i}" for i in range(N_STATIC)]
        + [f"delta{i}" for i in range(N_STATIC)]
        + [f"deltadelta{i}" for i in range(N_STATIC)]
    )


# Media y std intercaladas por dimensión, en el orden de `pool_lfcc` del original.
FEATURE_ORDER = tuple(f"{NAMESPACE}.{name}_{stat}" for name in _dim_names() for stat in ("mean", "std"))


def _hamming_periodic(n: int) -> np.ndarray:
    """`scipy.signal.get_window("hamming", n, fftbins=True)`."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / n)


_WINDOW = np.pad(
    _hamming_periodic(FRAME_LENGTH),
    ((N_FFT - FRAME_LENGTH) // 2, N_FFT - FRAME_LENGTH - (N_FFT - FRAME_LENGTH) // 2),
)


def _linear_filterbank(sr: int) -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    edges = np.linspace(0, sr / 2, N_FILTERS + 2)
    fb = np.zeros((N_FILTERS, n_bins))
    for i in range(N_FILTERS):
        lo, center, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs >= lo) & (freqs <= center)
        falling = (freqs > center) & (freqs <= hi)
        if center > lo:
            fb[i, rising] = (freqs[rising] - lo) / (center - lo)
        if hi > center:
            fb[i, falling] = (hi - freqs[falling]) / (hi - center)
    return fb


def _dct_ortho_matrix(n: int) -> np.ndarray:
    """DCT tipo II con `norm="ortho"` como matriz (n, n)."""
    k = np.arange(n).reshape(-1, 1)
    m = np.cos(np.pi * k * (2 * np.arange(n) + 1) / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


_DCT = _dct_ortho_matrix(N_FILTERS)


def power_spectrogram(chunk: np.ndarray) -> np.ndarray:
    """`abs(librosa.stft(chunk.astype(float32), n_fft=256, hop_length=80, win_length=160,
    window="hamming"))**2`, forma (n_bins, n_frames). La precisión simple del original se respeta:
    el espectro complejo se guarda en complex64 antes de tomar la magnitud."""
    y = np.asarray(chunk, dtype=np.float32)
    pad = N_FFT // 2
    padded = np.pad(y, (pad, pad))
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH][:n_frames]
    spec = np.fft.rfft(frames * _WINDOW, axis=1).astype(np.complex64)
    return (np.abs(spec) ** 2).T


def static_lfcc(chunk: np.ndarray, sr: int) -> np.ndarray:
    """LFCC estáticos, forma (n_frames, 20)."""
    filtered = _linear_filterbank(sr) @ power_spectrogram(chunk)
    return (_DCT @ np.log(filtered + 1e-10)).T


def _savgol_dot_coeffs(window: int, polyorder: int, deriv: int) -> np.ndarray:
    half = window // 2
    x = np.arange(-half, window - half, dtype=np.float64)
    a = x ** np.arange(polyorder + 1).reshape(-1, 1)
    y = np.zeros(polyorder + 1)
    y[deriv] = float(math.factorial(deriv))
    coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coeffs


def _polyder(p: np.ndarray, m: int) -> np.ndarray:
    n = len(p)
    if m >= n:
        return np.zeros_like(p[:1])
    dp = p[:-m].copy()
    for k in range(m):
        dp *= np.arange(n - k - 1, m - k - 1, -1).reshape((n - m,) + (1,) * (p.ndim - 1))
    return dp


def delta(data: np.ndarray, width: int, order: int) -> np.ndarray:
    """`librosa.feature.delta(data, width=width, order=order)` sobre el último eje, en NumPy.

    Es `scipy.signal.savgol_filter(data, width, polyorder=order, deriv=order, mode="interp")`:
    convolución en el interior y ajuste polinomial en los `width // 2` bordes de cada lado.
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[-1]
    half = width // 2
    out = np.zeros_like(x)
    coeffs = _savgol_dot_coeffs(width, order, order)
    windows = np.lib.stride_tricks.sliding_window_view(x, width, axis=-1)
    out[..., half : n - half] = windows @ coeffs
    t = np.arange(width)
    for start, stop, lo, hi in ((0, width, 0, half), (n - width, n, n - half, n)):
        poly = np.polyfit(t, x[..., start:stop].T, order)
        poly = _polyder(poly, order)
        i = np.arange(lo - start, hi - start).reshape(-1, 1)
        values = np.zeros((len(i), x.shape[0]))
        for pv in poly:
            values = values * i + pv
        out[..., lo:hi] = values.T
    return out


def lfcc_60(chunk: np.ndarray, sr: int) -> np.ndarray | None:
    """Estáticos + delta + delta-delta, forma (n_frames, 60). `None` con menos de 3 frames."""
    static = static_lfcc(chunk, sr)
    n_frames = static.shape[0]
    if n_frames < 3:
        return None
    width = DELTA_WIDTH
    if width >= n_frames:
        width = n_frames - 1 if (n_frames - 1) % 2 == 1 else n_frames - 2
        width = max(3, width)
    static_t = static.T
    full = np.concatenate([static_t, delta(static_t, width, 1), delta(static_t, width, 2)], axis=0)
    return full.T


def features_from_chunks(chunks: list[np.ndarray], sr: int) -> tuple[np.ndarray | None, float]:
    """Normaliza el nivel del habla conjunta y devuelve (media‖std por dimensión, ganancia)."""
    speech = np.concatenate(chunks).astype(np.float64)
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    gain = (TARGET_RMS / speech_rms) if speech_rms > 1e-8 else 1.0
    frames = [f for f in (lfcc_60(c * gain, sr) for c in chunks) if f is not None]
    if not frames:
        return None, gain
    full = np.concatenate(frames, axis=0)
    pooled = np.empty(2 * full.shape[1])
    pooled[0::2] = full.mean(axis=0)
    pooled[1::2] = full.std(axis=0)
    return pooled, gain


@extractors.register(
    NAMESPACE, version=1, channels=(0,), needs_seg=True,
    license="BSD-3-Clause", product_safe=True, budget_ms=150,
)
def extract(ex: AudioExample) -> ExtractorResult:
    audio = np.asarray(ex.ch0, dtype=np.float64)
    turns = ex.seg.by_channel(0) if ex.seg is not None else ()
    chunks = []
    for turn in turns:
        i0, i1 = max(0, int(turn.start * ex.sr)), min(len(audio), int(turn.end * ex.sr))
        if i1 - i0 >= FRAME_LENGTH:
            chunks.append(audio[i0:i1])

    fallback = not chunks
    if fallback and len(audio) >= FRAME_LENGTH:
        chunks = [audio]

    pooled, gain = features_from_chunks(chunks, ex.sr) if chunks else (None, 1.0)
    insufficient = pooled is None
    if insufficient:
        pooled = np.zeros(len(FEATURE_ORDER))

    features = {name: float(v) for name, v in zip(FEATURE_ORDER, pooled)}
    diagnostics = {
        "n_turns_ch0": len(turns),
        "n_chunks": len(chunks),
        "speech_seconds": round(sum(len(c) for c in chunks) / ex.sr, 3),
        "gain_db": round(20.0 * math.log10(gain), 3),
        "vad_fallback_full_channel": bool(fallback),
        "insufficient_audio": bool(insufficient),
    }
    return features, diagnostics
