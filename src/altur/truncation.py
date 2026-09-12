"""Curva de truncación y perfil de latencia.

Responde el criterio de Latency del reto, que tiene dos preguntas distintas que es fácil
confundir en una sola cifra:

- **¿Cuántos segundos de audio necesita?** Es producto: cuánto tiene que esperar el sistema
  antes de poder decidir.
- **¿Cuánto tarda en calcular?** Es ingeniería: CPU sobre el audio que ya tiene.

Se reportan **por separado**. Un sistema que decide con 5 s de audio en 200 ms no es lo mismo
que uno que necesita la llamada entera y tarda 200 ms, y una cifra única los hace ver igual.

**No activa parada temprana en `/detect`.** La curva es evaluación offline; convertirla en una
política de inferencia es una decisión de producto que no se toma desde aquí.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from altur.models.base import ModelError
from altur.types import AudioExample

HORIZONS_S: tuple[float | None, ...] = (5.0, 10.0, 20.0, 30.0, 60.0, 120.0, None)


def truncate(ex: AudioExample, seconds: float | None) -> AudioExample:
    """Primeros `seconds` de la llamada. `None` devuelve la llamada completa.

    Trunca, no remuestrea ni rellena: una llamada más corta que el horizonte se devuelve tal
    cual, y su duración real se reporta. Rellenar con silencio inventaría audio que el sistema
    nunca tuvo.
    """
    if seconds is None:
        return ex
    if seconds <= 0:
        raise ModelError(f"el horizonte tiene que ser positivo; llegó {seconds}")
    n = round(seconds * ex.sr)
    if n >= ex.n_samples:
        return ex
    ch1 = None if ex.ch1 is None else np.asarray(ex.ch1)[:n]
    return AudioExample(ch0=np.asarray(ex.ch0)[:n], ch1=ch1, sr=ex.sr)


def _pct(vals: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if len(vals) else float("nan")


@dataclass(slots=True)
class Timing:
    """Tiempo de cómputo. Nada que ver con los segundos de audio consumidos."""

    label: str
    n: int
    p50_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, label: str, ms: Sequence[float]) -> "Timing":
        return cls(
            label=label,
            n=len(ms),
            p50_ms=_pct(ms, 50),
            p95_ms=_pct(ms, 95),
            max_ms=float(max(ms)) if ms else float("nan"),
        )


@dataclass(slots=True)
class HorizonPoint:
    horizon_s: float | None
    n: int
    audio_seconds_used_median: float
    audio_seconds_used_p95: float
    calls_shorter_than_horizon: int
    extract: dict[str, Any]
    model: dict[str, Any]
    total: dict[str, Any]
    auc: float = float("nan")
    mean_confidence: float = float("nan")
    agreement_with_full: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confidence(scores: Sequence[float]) -> np.ndarray:
    """Distancia al umbral neutro, escalada a [0, 1]. 0 = duda total, 1 = certeza."""
    s = np.asarray(scores, dtype=np.float64)
    return np.abs(s - 0.5) * 2.0


def measure_horizon(
    examples: Mapping[str, AudioExample],
    horizon_s: float | None,
    extract_fn: Callable[[AudioExample], Mapping[str, float]],
    predict_fn: Callable[[Sequence[Mapping[str, float]]], np.ndarray],
) -> tuple[HorizonPoint, dict[str, float], np.ndarray]:
    """Mide un horizonte: cronometra extracción y modelo por separado."""
    ids = list(examples)
    ms_extract: list[float] = []
    segundos_usados: list[float] = []
    filas: list[Mapping[str, float]] = []
    cortas = 0

    for cid in ids:
        recortado = truncate(examples[cid], horizon_s)
        segundos_usados.append(recortado.duration_s)
        if horizon_s is not None and recortado.duration_s < horizon_s - 1e-9:
            cortas += 1
        t0 = time.perf_counter()
        filas.append(extract_fn(recortado))
        ms_extract.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    scores = np.asarray(predict_fn(filas), dtype=np.float64)
    ms_modelo_total = (time.perf_counter() - t0) * 1000.0
    ms_modelo = [ms_modelo_total / max(len(ids), 1)] * len(ids)
    ms_total = [a + b for a, b in zip(ms_extract, ms_modelo, strict=True)]

    punto = HorizonPoint(
        horizon_s=horizon_s,
        n=len(ids),
        audio_seconds_used_median=float(np.median(segundos_usados)),
        audio_seconds_used_p95=_pct(segundos_usados, 95),
        calls_shorter_than_horizon=cortas,
        extract=asdict(Timing.from_samples("extract", ms_extract)),
        model=asdict(Timing.from_samples("model", ms_modelo)),
        total=asdict(Timing.from_samples("total", ms_total)),
        mean_confidence=float(np.mean(confidence(scores))),
        notes=[
            (
                "Segundos de audio consumidos y tiempo de computo son magnitudes distintas "
                "y se reportan por separado."
            ),
            (
                "El tiempo del modelo es el total del lote dividido entre n: por llamada es "
                "demasiado corto para cronometrarlo sin que domine el overhead del reloj."
            ),
        ],
    )
    if cortas:
        punto.notes.append(
            f"{cortas} llamadas son mas cortas que el horizonte y entran completas. No se "
            "rellenan con silencio: eso inventaria audio que el sistema nunca tuvo."
        )
    return punto, dict(zip(ids, scores, strict=True)), scores


def format_curve(puntos: Sequence[HorizonPoint]) -> str:
    cab = (
        f"{'horizonte':>10} {'audio med':>10} {'cortas':>7} {'extract p50':>12} "
        f"{'extract p95':>12} {'total p95':>10} {'AUC':>7} {'conf':>6} {'acuerdo':>8}"
    )
    filas = [cab, "-" * len(cab)]
    for p in puntos:
        h = "full" if p.horizon_s is None else f"{p.horizon_s:.0f}s"
        acuerdo = "—" if not np.isfinite(p.agreement_with_full) else f"{p.agreement_with_full:.3f}"
        auc = "—" if not np.isfinite(p.auc) else f"{p.auc:.4f}"
        filas.append(
            f"{h:>10} {p.audio_seconds_used_median:>10.1f} {p.calls_shorter_than_horizon:>7} "
            f"{p.extract['p50_ms']:>12.1f} {p.extract['p95_ms']:>12.1f} "
            f"{p.total['p95_ms']:>10.1f} {auc:>7} {p.mean_confidence:>6.3f} {acuerdo:>8}"
        )
    return "\n".join(filas)
