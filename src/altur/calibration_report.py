"""Bloque de calibración: descomposición de Brier, reliability con Wilson y prior shift.

Tres cosas que este módulo hace distinto de la versión v1 del plan, y por qué:

1. **`BS = REL − RES + UNC` NO es una identidad cuando se usan bins.** Solo lo es si todos
   los pronósticos dentro de un bin son idénticos. Con 3 bins por cuantiles no lo son, y el
   plan v1 lo afirmaba como identidad exacta. Aquí el **residual se calcula y se reporta**;
   si alguien lo ignora, que sea a la vista.

2. **Discriminación y calibración se reportan por separado.** Platt no cambia el ranking, así
   que no puede mover el AUC: leerlas juntas hace creer que calibrar "mejoró el modelo".

3. **El prior es configurable y se reporta su mapa.** D-A1.5: train es 59.9 % sintético y val
   47.9 %. Un umbral fijado sobre la tasa base de train está calibrado para una prevalencia
   que el set oculto probablemente no tiene.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from altur.models.base import ModelError

_EPS = 1e-15
DEFAULT_BINS = 3


def _pair(y: Any, s: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=np.int64).ravel()
    scores = np.asarray(s, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ModelError(f"y tiene {labels.shape} y los scores {scores.shape}")
    if labels.size == 0:
        raise ModelError("no hay nada que evaluar")
    if not np.isfinite(scores).all():
        raise ModelError("los scores contienen NaN o inf")
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise ModelError("los scores tienen que ser probabilidades en [0, 1]")
    malas = set(np.unique(labels)) - {0, 1}
    if malas:
        raise ModelError(f"y solo admite 0 y 1; llegaron {sorted(malas)}")
    return labels, scores


def log_loss(y: Any, s: Any) -> float:
    labels, scores = _pair(y, s)
    p = np.clip(scores, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1.0 - p)))


def brier(y: Any, s: Any) -> float:
    labels, scores = _pair(y, s)
    return float(np.mean((scores - labels) ** 2))


def grouped_brier(y: Any, s: Any, units: Sequence[str]) -> float:
    """Brier con cada unidad independiente pesando igual, no cada fila.

    Con grupos conservadores (1 llamada = 1 grupo, D-A1.2) coincide con el Brier original.
    Que coincidan **no** es una validación: es un síntoma de que no tenemos llave de unión.
    Cuando entren clones y donantes al corpus ampliado, dejarán de coincidir.
    """
    labels, scores = _pair(y, s)
    u = np.asarray(units)
    if u.shape[0] != labels.shape[0]:
        raise ModelError(f"hay {labels.shape[0]} filas y {u.shape[0]} unidades")
    por_unidad = [
        float(np.mean((scores[u == g] - labels[u == g]) ** 2)) for g in np.unique(u)
    ]
    return float(np.mean(por_unidad))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de Wilson al 95 %. Con n pequeño no se degrada como el normal."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    medio = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centro - medio)), float(min(1.0, centro + medio)))


@dataclass(slots=True)
class Bin:
    index: int
    lo: float
    hi: float
    n: int
    mean_score: float
    empirical_rate: float
    wilson95: tuple[float, float]


@dataclass(slots=True)
class BrierDecomposition:
    """REL, RES, UNC y el residual que la identidad con bins deja fuera."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    residual: float
    n_bins: int
    binning: str = "quantile"

    @property
    def identity_holds(self) -> bool:
        return abs(self.residual) < 1e-12

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["identity_holds"] = self.identity_holds
        d["note"] = (
            "BS = REL - RES + UNC + residual. Con bins el residual NO es cero salvo que "
            "todos los pronosticos de un bin sean identicos."
        )
        return d


def _quantile_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Asigna cada score a un bin por cuantiles. Bordes duplicados colapsan y se declara."""
    cortes = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(np.unique(cortes), scores, side="right")


def decompose_brier(y: Any, s: Any, n_bins: int = DEFAULT_BINS) -> BrierDecomposition:
    labels, scores = _pair(y, s)
    if n_bins < 1:
        raise ModelError("n_bins tiene que ser >= 1")
    idx = _quantile_bins(scores, n_bins)
    n = labels.size
    o_bar = float(labels.mean())

    rel = res = 0.0
    for b in np.unique(idx):
        m = idx == b
        n_k = int(m.sum())
        f_k = float(scores[m].mean())
        o_k = float(labels[m].mean())
        rel += n_k * (f_k - o_k) ** 2
        res += n_k * (o_k - o_bar) ** 2
    rel /= n
    res /= n
    unc = o_bar * (1.0 - o_bar)
    bs = float(np.mean((scores - labels) ** 2))
    return BrierDecomposition(
        brier=bs,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        residual=bs - (rel - res + unc),
        n_bins=int(np.unique(idx).size),
    )


def reliability(y: Any, s: Any, n_bins: int = DEFAULT_BINS) -> list[Bin]:
    """Curva de fiabilidad por cuantiles, con Wilson 95 % en cada bin."""
    labels, scores = _pair(y, s)
    idx = _quantile_bins(scores, n_bins)
    salida: list[Bin] = []
    for i, b in enumerate(np.unique(idx)):
        m = idx == b
        k = int(labels[m].sum())
        n_k = int(m.sum())
        salida.append(
            Bin(
                index=i,
                lo=float(scores[m].min()),
                hi=float(scores[m].max()),
                n=n_k,
                mean_score=float(scores[m].mean()),
                empirical_rate=k / n_k,
                wilson95=wilson_interval(k, n_k),
            )
        )
    return salida


def score_histogram(s: Any, n_bins: int = 20) -> tuple[list[float], list[int]]:
    """Histograma de scores. Va superpuesto a la reliability: sin él, un bin con 3 casos
    y un bin con 200 se ven igual de convincentes."""
    scores = np.asarray(s, dtype=np.float64).ravel()
    counts, edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    return [float(e) for e in edges], [int(c) for c in counts]


def shift_prior(s: Any, *, prior_from: float, prior_to: float) -> np.ndarray:
    """Reexpresa P(synthetic) bajo otra tasa base, moviendo las odds.

    `odds_nuevas = odds_viejas × (π_to / (1 − π_to)) × ((1 − π_from) / π_from)`

    Desacopla el score de la prevalencia con la que se entrenó. **No se elige `prior_to`
    mirando `val`**: el prior del bundle queda configurable y se reporta su mapa.
    """
    for nombre, v in (("prior_from", prior_from), ("prior_to", prior_to)):
        if not 0.0 < v < 1.0:
            raise ModelError(f"{nombre} tiene que estar en (0, 1); llegó {v}")
    scores = np.clip(np.asarray(s, dtype=np.float64), _EPS, 1.0 - _EPS)
    odds = scores / (1.0 - scores)
    factor = (prior_to / (1.0 - prior_to)) * ((1.0 - prior_from) / prior_from)
    nuevas = odds * factor
    return nuevas / (1.0 + nuevas)


@dataclass(slots=True)
class PriorPoint:
    prior: float
    brier: float
    log_loss: float
    mean_score: float
    positives_at_threshold: float


@dataclass(slots=True)
class CalibrationReport:
    n: int
    auc_unchanged_by_calibration: bool
    log_loss: float
    brier: float
    brier_grouped: float
    decomposition: dict[str, Any]
    bins: list[dict[str, Any]]
    histogram: dict[str, Any]
    prior_map: list[dict[str, Any]]
    observed_prior: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prior_sensitivity(
    y: Any, s: Any, *, prior_from: float, priors: Sequence[float], threshold: float = 0.5
) -> list[PriorPoint]:
    labels, scores = _pair(y, s)
    puntos = []
    for p in priors:
        movidos = shift_prior(scores, prior_from=prior_from, prior_to=p)
        puntos.append(
            PriorPoint(
                prior=float(p),
                brier=float(np.mean((movidos - labels) ** 2)),
                log_loss=log_loss(labels, movidos),
                mean_score=float(movidos.mean()),
                positives_at_threshold=float(np.mean(movidos > threshold)),
            )
        )
    return puntos


def build_report(
    y: Any,
    s: Any,
    units: Sequence[str],
    *,
    n_bins: int = DEFAULT_BINS,
    priors: Sequence[float] = (0.3, 0.4, 0.479, 0.5, 0.599, 0.7),
    conservative_groups: bool = True,
    extra_notes: Mapping[str, str] | None = None,
) -> CalibrationReport:
    labels, scores = _pair(y, s)
    observado = float(labels.mean())
    notas = [
        (
            "Discriminacion y calibracion se reportan por separado: Platt es monotono y no "
            "puede mover el AUC."
        ),
        "BS = REL - RES + UNC + residual. El residual esta calculado, no asumido cero.",
        "No se afirma que esta calibracion sea transportable al set oculto del juez.",
    ]
    if conservative_groups:
        notas.append(
            "Grupos conservadores (1 llamada = 1 grupo, D-A1.2): los intervalos son "
            "condicionales a las llamadas observadas y NO cubren dependencia latente entre "
            "llamadas del mismo hablante, voz o cadena."
        )
    if extra_notes:
        notas.extend(f"{k}: {v}" for k, v in extra_notes.items())

    edges, counts = score_histogram(scores)
    return CalibrationReport(
        n=int(labels.size),
        auc_unchanged_by_calibration=True,
        log_loss=log_loss(labels, scores),
        brier=brier(labels, scores),
        brier_grouped=grouped_brier(labels, scores, units),
        decomposition=decompose_brier(labels, scores, n_bins=n_bins).to_dict(),
        bins=[asdict(b) for b in reliability(labels, scores, n_bins=n_bins)],
        histogram={"edges": edges, "counts": counts},
        prior_map=[asdict(p) for p in prior_sensitivity(
            labels, scores, prior_from=observado, priors=priors
        )],
        observed_prior=observado,
        notes=notas,
    )
