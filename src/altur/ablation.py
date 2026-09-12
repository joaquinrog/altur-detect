"""Ablación OOF: qué aporta cada rama, y cuánto de eso es señal.

Todo escenario se evalúa con el mismo protocolo y los mismos folds congelados. Lo que se
compara son **predicciones externas**, nunca internas.

Dos reglas que el módulo impone y no deja negociar:

1. **Ningún número sale sin intervalo.** Con n=282 y `val` de 71 llamadas, diferencias de
   0.02 AUC no son señal (`AGENTS.md` regla 4). Un ranking sin incertidumbre invita a leer
   ruido como mejora.
2. **El bootstrap es por unidad independiente**, no por clip. Hoy los grupos son
   conservadores — 1 llamada = 1 grupo (D-A1.2) — así que el intervalo **no cubre**
   dependencia latente entre llamadas del mismo hablante o de la misma cadena. Eso se
   declara en cada reporte, no se omite.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from altur.crossfit import (
    Branch,
    DisagreementFusion,
    FeatureTable,
    logistic_fusion,
    make_nested_fit_predict,
)
from altur.models.base import ModelError
from altur.protocol import nested_cv

BOOTSTRAP_SEED = 20260912


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def brier(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((np.asarray(s, dtype=np.float64) - np.asarray(y)) ** 2))


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    units: Sequence[str],
    *,
    stat: Callable[[np.ndarray, np.ndarray], float] = auc,
    n_boot: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentil 2.5–97.5 remuestreando **unidades**, no filas."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    unidades = np.asarray(units)
    distintas = np.unique(unidades)
    por_unidad = {u: np.flatnonzero(unidades == u) for u in distintas}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        elegidas = rng.choice(distintas, size=distintas.size, replace=True)
        idx = np.concatenate([por_unidad[u] for u in elegidas])
        v = stat(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


@dataclass(slots=True)
class ScenarioResult:
    """Un escenario de ablación, con su incertidumbre pegada al número."""

    name: str
    branches: list[str]
    fusion: str
    n: int
    auc: float
    auc_ci95: tuple[float, float]
    brier: float
    brier_ci95: tuple[float, float]
    threshold_mean: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenarios(branches: Sequence[Branch], backbone: str) -> list[tuple[str, list[Branch], str]]:
    nombres = [b.name for b in branches]
    out: list[tuple[str, list[Branch], str]] = []
    for b in branches:
        out.append((f"solo_{b.name}", [b], "logistic"))
    if len(branches) > 1:
        out.append(("fusion_logistica", list(branches), "logistic"))
        out.append(("regla_de_desacuerdo", list(branches), "disagreement"))
        for b in branches:
            if b.name == backbone:
                continue
            resto = [x for x in branches if x.name != b.name]
            if resto:
                out.append((f"sin_{b.name}", resto, "logistic"))
    if len({n for n, _, _ in out}) != len(out):
        raise ModelError(f"escenarios con nombre repetido a partir de {nombres}")
    return out


def run_ablation(
    ds: Any,
    protocol: Any,
    features: Mapping[str, Mapping[str, float]],
    labels: Mapping[str, int],
    branches: Sequence[Branch],
    *,
    groups: Mapping[str, str] | None,
    calibrator_factory: Callable[[], Any],
    backbone: str = "acoustic",
    inner_folds: int = 3,
    n_boot: int = 2000,
) -> list[ScenarioResult]:
    """Corre cada escenario con los mismos folds y devuelve la tabla comparativa."""
    if not branches:
        raise ModelError("hace falta al menos una rama")
    resultados: list[ScenarioResult] = []

    for nombre, ramas, tipo_fusion in _scenarios(branches, backbone):
        if tipo_fusion == "disagreement":
            def fusion_factory() -> DisagreementFusion:
                return DisagreementFusion(backbone=backbone)
        else:
            fusion_factory = logistic_fusion

        tabla = FeatureTable(features, labels)
        fp = make_nested_fit_predict(
            tabla,
            ramas,
            calibrator_factory=calibrator_factory,
            fusion_factory=fusion_factory,
            inner_folds=inner_folds,
        )
        res = nested_cv(ds, protocol, fp, groups=dict(groups) if groups else None)
        y, s = res.y_true, res.y_score
        unidades = (
            [groups[i] for i in res.ids] if groups else list(res.ids)
        )
        notas = []
        if res.conservative_groups:
            notas.append(
                "Grupos conservadores (1 llamada = 1 grupo, D-A1.2): el intervalo es "
                "condicional a las llamadas observadas y NO cubre dependencia latente "
                "entre llamadas del mismo hablante, voz o cadena."
            )
        resultados.append(
            ScenarioResult(
                name=nombre,
                branches=[b.name for b in ramas],
                fusion=tipo_fusion,
                n=len(y),
                auc=auc(y, s),
                auc_ci95=bootstrap_ci(y, s, unidades, stat=auc, n_boot=n_boot),
                brier=brier(y, s),
                brier_ci95=bootstrap_ci(y, s, unidades, stat=brier, n_boot=n_boot),
                threshold_mean=float(np.mean([f.threshold for f in res.folds])),
                notes=notas,
            )
        )
    return resultados


def format_table(resultados: Sequence[ScenarioResult]) -> str:
    """Tabla legible. El intervalo va pegado al número, nunca en una nota al pie."""
    filas = [f"{'escenario':<22} {'n':>4} {'AUC':>7} {'IC95 AUC':>18} {'Brier':>7} {'umbral':>7}"]
    filas.append("-" * len(filas[0]))
    for r in resultados:
        lo, hi = r.auc_ci95
        filas.append(
            f"{r.name:<22} {r.n:>4} {r.auc:>7.4f} {f'[{lo:.4f}, {hi:.4f}]':>18} "
            f"{r.brier:>7.4f} {r.threshold_mean:>7.3f}"
        )
    return "\n".join(filas)
