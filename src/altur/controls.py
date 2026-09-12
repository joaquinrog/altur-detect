"""Controles de confound: qué separa las clases cuando no debería.

Módulo de **entrenamiento**: usa scipy, no entra a la imagen.

Tres reglas que impone:

1. **La familia de hipótesis se declara ANTES de correr.** Si se declara después, Benjamini-
   Hochberg corrige sobre las pruebas que sobrevivieron a la mirada, que es exactamente el
   sesgo que la corrección pretende evitar. `HypothesisFamily` lleva el momento de la
   declaración y `run_controls` falla si recibe una hipótesis que no estaba en la familia.

2. **Ningún veto por un umbral de AUC inventado.** La v1 del plan vetaba con
   `AUC ∉ [0.42, 0.58]`; ese número me lo inventé yo. Lo que produce un control positivo es
   **qué conclusiones quedan invalidadas**, no un veredicto escondido en un número.

3. **Ubicación y escala se prueban juntas.** Es el punto ciego documentado de CP0: nuestro
   AUC 0.504 sobre energía de banda y el Mann-Whitney de Biniza son **ambos** tests de
   ubicación, y comparten la ceguera a diferencias de escala — donde estaba la señal más
   fuerte. Por eso cada hipótesis corre Mann-Whitney (ubicación), Brown-Forsythe (escala) y
   el AUC de la distancia a la mediana (escala, en la misma unidad que el resto).
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from altur.models.base import ModelError

ALPHA = 0.05


def benjamini_hochberg(pvals: Sequence[float], alpha: float = ALPHA) -> tuple[np.ndarray, float]:
    """Devuelve los q-valores y el umbral de p que resulta rechazado.

    BH controla la tasa de falsos descubrimientos, no la familywise. Con n=353 y muchas
    features, Bonferroni dejaría el estudio sin potencia y BH es lo que el plan pide.
    """
    p = np.asarray(pvals, dtype=np.float64)
    if p.size == 0:
        return np.array([]), 0.0
    if not np.all((p >= 0) & (p <= 1)):
        raise ModelError("los p-valores tienen que estar en [0, 1]")
    m = p.size
    orden = np.argsort(p)
    ordenados = p[orden]
    q_ord = ordenados * m / np.arange(1, m + 1)
    q_ord = np.minimum.accumulate(q_ord[::-1])[::-1]
    q = np.empty_like(q_ord)
    q[orden] = np.minimum(q_ord, 1.0)
    rechazados = ordenados <= alpha * np.arange(1, m + 1) / m
    corte = float(ordenados[rechazados].max()) if rechazados.any() else 0.0
    return q, corte


def auc_score(y: Sequence[int], s: Sequence[float]) -> float:
    labels = np.asarray(y)
    scores = np.asarray(s, dtype=np.float64)
    finito = np.isfinite(scores)
    labels, scores = labels[finito], scores[finito]
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    orden = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ss = scores[orden]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[orden[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(labels.sum())
    n0 = float(labels.size - n1)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def median_distance_auc(y: Sequence[int], x: Sequence[float]) -> float:
    """AUC sobre |x − mediana|. Es el test de escala expresado en la misma unidad que el resto.

    Un Mann-Whitney sobre `x` no lo ve: si una clase se concentra en un valor y la otra se
    reparte, las medianas pueden coincidir y el test de ubicación dar 0.5 limpio.
    """
    valores = np.asarray(x, dtype=np.float64)
    return auc_score(y, np.abs(valores - float(np.median(valores))))


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Una pregunta concreta, declarada antes de mirar los datos."""

    name: str
    question: str
    invalidates_if_positive: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HypothesisFamily:
    """La familia sobre la que se corrige. Se congela antes de ejecutar nada."""

    name: str
    hypotheses: tuple[Hypothesis, ...]
    declared_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ModelError("una familia vacía no corrige nada")
        nombres = [h.name for h in self.hypotheses]
        if len(set(nombres)) != len(nombres):
            raise ModelError(f"hipótesis repetidas en la familia: {nombres}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(h.name for h in self.hypotheses)

    def get(self, name: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.name == name:
                return h
        raise ModelError(
            f"{name!r} no está en la familia declarada {self.names}. Añadir una hipótesis "
            "después de mirar los datos rompe la corrección por comparaciones múltiples."
        )


@dataclass(slots=True)
class ControlResult:
    """Un control corrido. Siempre se reporta, salga positivo o negativo."""

    name: str
    question: str
    n: int
    auc: float
    auc_ci95: tuple[float, float]
    p_location: float
    p_scale: float
    scale_auc: float
    q_location: float = float("nan")
    q_scale: float = float("nan")
    positive: bool = False
    invalidated_claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mann_whitney_p(y: np.ndarray, x: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    a, b = x[y == 1], x[y == 0]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _brown_forsythe_p(y: np.ndarray, x: np.ndarray) -> float:
    from scipy.stats import levene

    a, b = x[y == 1], x[y == 0]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(levene(a, b, center="median").pvalue)


def _bootstrap_auc_ci(
    y: np.ndarray, s: np.ndarray, units: Sequence[str], n_boot: int, seed: int
) -> tuple[float, float]:
    u = np.asarray(units)
    distintas = np.unique(u)
    idx_por_unidad = {g: np.flatnonzero(u == g) for g in distintas}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        elegidas = rng.choice(distintas, size=distintas.size, replace=True)
        idx = np.concatenate([idx_por_unidad[g] for g in elegidas])
        v = auc_score(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def run_controls(
    family: HypothesisFamily,
    observations: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    *,
    units: Mapping[str, Sequence[str]] | None = None,
    alpha: float = ALPHA,
    n_boot: int = 1000,
    seed: int = 20260912,
    conservative_groups: bool = True,
) -> list[ControlResult]:
    """Corre cada control de la familia y corrige con BH sobre TODA la familia.

    `observations[nombre] = (y, score)`. Un nombre que no esté en la familia declarada es un
    error, no una advertencia.
    """
    faltan = set(family.names) - set(observations)
    if faltan:
        raise ModelError(
            f"la familia declara {sorted(faltan)} pero no llegaron observaciones. "
            "Un control que no corre se reporta como no corrido, no se omite en silencio."
        )

    resultados: list[ControlResult] = []
    for nombre in family.names:
        h = family.get(nombre)
        y_raw, s_raw = observations[nombre]
        y = np.asarray(y_raw, dtype=np.int64)
        s = np.asarray(s_raw, dtype=np.float64)
        if y.shape != s.shape:
            raise ModelError(f"{nombre}: y {y.shape} y scores {s.shape} no cuadran")
        u = list(units[nombre]) if units and nombre in units else [f"u{i}" for i in range(y.size)]
        notas = []
        if conservative_groups:
            notas.append(
                "Grupos conservadores (1 llamada = 1 grupo, D-A1.2): el intervalo no cubre "
                "dependencia latente entre llamadas del mismo hablante, voz o cadena."
            )
        resultados.append(
            ControlResult(
                name=nombre,
                question=h.question,
                n=int(y.size),
                auc=auc_score(y, s),
                auc_ci95=_bootstrap_auc_ci(y, s, u, n_boot, seed),
                p_location=_mann_whitney_p(y, s),
                p_scale=_brown_forsythe_p(y, s),
                scale_auc=median_distance_auc(y, s),
                notes=notas,
            )
        )

    # BH sobre toda la familia y sobre AMBOS tipos de test, no uno por uno.
    p_loc = [r.p_location for r in resultados]
    p_esc = [r.p_scale for r in resultados]
    todos = np.array(
        [p if np.isfinite(p) else 1.0 for p in list(p_loc) + list(p_esc)], dtype=np.float64
    )
    q, _ = benjamini_hochberg(todos, alpha=alpha)
    n = len(resultados)
    for i, r in enumerate(resultados):
        r.q_location = float(q[i])
        r.q_scale = float(q[i + n])
        r.positive = bool(min(r.q_location, r.q_scale) <= alpha)
        if r.positive:
            r.invalidated_claims = list(family.get(r.name).invalidates_if_positive)
        else:
            r.notes.append(
                "Control negativo bajo BH. Se reporta igual: un control que no separa es "
                "evidencia, no ausencia de resultado."
            )
    return resultados


def format_controls(resultados: Sequence[ControlResult]) -> str:
    lineas = [
        f"{'control':<26} {'n':>4} {'AUC':>7} {'IC95':>18} {'q_ubic':>8} {'q_esc':>8} {'AUC_esc':>8} {'':>4}"
    ]
    lineas.append("-" * len(lineas[0]))
    for r in resultados:
        lo, hi = r.auc_ci95
        # Un control positivo que no invalida nada es una REFERENCIA: se declaró esperando
        # que separara. Marcarlo igual que un confound haría ruido donde hay señal.
        if r.positive:
            marca = "🔴" if r.invalidated_claims else "ref"
        else:
            marca = "ok"
        lineas.append(
            f"{r.name:<26} {r.n:>4} {r.auc:>7.4f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
            f"{r.q_location:>8.2e} {r.q_scale:>8.2e} {r.scale_auc:>8.4f} {marca:>4}"
        )
    invalidados = [(r.name, c) for r in resultados if r.positive for c in r.invalidated_claims]
    if invalidados:
        lineas.append("")
        lineas.append("Claims invalidados por los controles positivos:")
        for nombre, claim in invalidados:
            lineas.append(f"  [{nombre}] {claim}")
    return "\n".join(lineas)
