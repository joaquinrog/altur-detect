"""Perturbaciones v2: las condiciones del set de robustez de seis condiciones (D-A5.3).

v1 (`transforms.py`) no tiene eje de códec real: el corpus ya viene en mu-law y
`mulaw_roundtrip@1` es un no-op bit-exacto (D-A2.3). La fase 1 de `research/spectral_factory`
evaluó seis condiciones: `clean`, `noise_snr10`, `pitch_up1st`, `timestretch_0.9x`,
`lowpass_3400hz` y `opus_16kbps`. Aquí se reimplementan como transforms registrados; el mapeo
condición → ref está congelado en `configs/perturbations/v2.yaml`.

Dos diferencias deliberadas con `research/spectral_factory/persona3_prosodic/corruption.py`:

- **RNG por audio.** El ruido usa el `rng` que deriva el runner del SHA-256 del audio. Nunca
  `np.random` global, que es el bug de reproducibilidad del original.
- **Condiciones fijas, no rangos.** Son condiciones de evaluación (`use="holdout"`), no de
  augmentación: mismo parámetro para todas las llamadas.

Pitch, tempo y Opus usan **ffmpeg** como proceso externo. Solo corren en el arnés de
experimentos: `/detect` nunca importa este módulo y la imagen no lleva ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from .registry import transforms
from .transforms import _check_rng
from .types import AudioExample, Segmentation, Turn

_SR = 8000
_SEMITONE = 2.0 ** (1.0 / 12.0)
_TEMPO_RATE = 0.9
_RAW_IN = ("-f", "f32le", "-ar", str(_SR), "-ac", "1", "-i", "pipe:0")
_RAW_OUT = ("-f", "f32le", "-ar", str(_SR), "-ac", "1", "pipe:1")


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("pitch_up1st, timestretch_0_9x y opus_16kbps requieren ffmpeg con libopus en el PATH")
    return path


def _run(args: list[str], data: bytes) -> bytes:
    result = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", *args],
        input=data, capture_output=True, check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise RuntimeError(f"ffmpeg falló: {detail}")
    return result.stdout


def _raw(samples: np.ndarray) -> bytes:
    return np.ascontiguousarray(samples, dtype="<f4").tobytes()


def _from_raw(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<f4").astype(np.float32)


def _filter(samples: np.ndarray, audio_filter: str) -> np.ndarray:
    return _from_raw(_run([*_RAW_IN, "-af", audio_filter, *_RAW_OUT], _raw(samples)))


def _fit_length(samples: np.ndarray, n: int) -> np.ndarray:
    """Recorta o rellena con ceros. Los filtros de ffmpeg pueden mover unas decenas de muestras."""
    if len(samples) >= n:
        return samples[:n]
    return np.concatenate([samples, np.zeros(n - len(samples), dtype=samples.dtype)])


@transforms.register(
    "noise_snr10", version=1, use="holdout", purpose="robustness",
    changes_length=False, shifts_timing=False, channels=(0,),
    params={"snr_db": 10.0, "distribution": "gaussian", "power_reference": "ch0_full"},
    source_condition="noise_snr10",
)
def noise_snr10(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    _check_rng(rng)
    if rng is None:
        raise ValueError("noise_snr10@1 necesita el rng que deriva el runner del audio; nunca usa np.random global")
    x = ex.ch0.astype(np.float64)
    power = float(np.mean(x**2))
    if power > 0.0:
        sigma = np.sqrt(power / 10.0 ** (10.0 / 10.0))
        x = np.clip(x + rng.normal(0.0, sigma, size=x.shape), -1.0, 1.0)
    return AudioExample(x.astype(np.float32), ex.ch1, sr=ex.sr, seg=ex.seg)


@transforms.register(
    "pitch_up1st", version=1, use="holdout", purpose="robustness",
    changes_length=False, shifts_timing=False, channels=(0,),
    params={"semitones": 1.0, "method": "ffmpeg asetrate+aresample+atempo", "length": "fit_to_input"},
    source_condition="pitch_up1st",
)
def pitch_up1st(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    _check_rng(rng)
    shifted = _filter(ex.ch0, f"asetrate={_SR * _SEMITONE:.6f},aresample={_SR},atempo={1.0 / _SEMITONE:.6f}")
    return AudioExample(_fit_length(shifted, ex.n_samples), ex.ch1, sr=ex.sr, seg=ex.seg)


def _rescale(seg: Segmentation | None, scale: float) -> Segmentation | None:
    if seg is None:
        return None
    turns = tuple(Turn(t.channel, t.start * scale, t.end * scale) for t in seg.turns)
    return Segmentation(turns, seg.source, {**seg.params, "time_scale": scale})


@transforms.register(
    "timestretch_0_9x", version=1, use="holdout", purpose="robustness",
    changes_length=True, shifts_timing=True, channels=(0, 1),
    params={"rate": _TEMPO_RATE, "method": "ffmpeg atempo", "turns": "rescaled_by_1/rate"},
    source_condition="timestretch_0.9x",
)
def timestretch_0_9x(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    """Rate 0.9 = más lento y más largo. Se estiran los DOS canales: AudioExample exige igual longitud."""
    _check_rng(rng)
    n_out = round(ex.n_samples / _TEMPO_RATE)
    ch0 = _fit_length(_filter(ex.ch0, f"atempo={_TEMPO_RATE}"), n_out)
    ch1 = None if ex.ch1 is None else _fit_length(_filter(ex.ch1, f"atempo={_TEMPO_RATE}"), n_out)
    return AudioExample(ch0, ch1, sr=ex.sr, seg=_rescale(ex.seg, 1.0 / _TEMPO_RATE))


@transforms.register(
    "opus_16kbps", version=1, use="holdout", purpose="robustness",
    changes_length=False, shifts_timing=False, channels=(0,),
    params={"codec": "libopus", "bitrate": "16k", "container": "ogg", "length": "fit_to_input"},
    source_condition="opus_16kbps",
)
def opus_16kbps(ex: AudioExample, rng: np.random.Generator | None = None) -> AudioExample:
    _check_rng(rng)
    encoded = _run([*_RAW_IN, "-c:a", "libopus", "-b:a", "16k", "-f", "ogg", "pipe:1"], _raw(ex.ch0))
    decoded = _from_raw(_run(["-i", "pipe:0", *_RAW_OUT], encoded))
    return AudioExample(_fit_length(decoded, ex.n_samples), ex.ch1, sr=ex.sr, seg=ex.seg)
