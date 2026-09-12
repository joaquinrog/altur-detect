"""Cross-fitting anidado: ramas, fusión, calibración y umbral, todo dentro de `IN_k`.

La corrección más importante del plan (§3) es esta: que los scores de rama sean out-of-fold
**no basta** si la fusión y el calibrador se ajustan sobre esas mismas filas y ahí se evalúan.

El flujo por cada fold externo `k`:

    IN_k  ─┬─ splits internos agrupados ──> scores OOF internos por rama
           │                                        │
           │                                        ├──> fusión  (se ajusta AQUÍ)
           │                                        ├──> calibrador (AQUÍ)
           │                                        └──> umbral (AQUÍ)
           └─ refit de cada rama sobre IN_k completo
                                                    │
    OUT_k ──────────────────────────────────────────┴──> se predice UNA vez, todo congelado

Ninguna fila de `OUT_k` toca fit, fusión, calibración ni umbral. `FeatureTable` puede auditar
cada acceso para que eso sea un test y no una promesa.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from altur.models.base import ModelError, _sigmoid, check_feature_order, check_labels
from altur.protocol import grouped_inner_splits
from altur.registry import fusions

DEFAULT_INNER_FOLDS = 3


@dataclass(frozen=True, slots=True)
class Branch:
    """Una rama del detector: una familia de features y cómo construir su modelo."""

    name: str
    feature_order: tuple[str, ...]
    make_model: Callable[[Sequence[str]], Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelError("una rama necesita nombre")
        object.__setattr__(self, "feature_order", check_feature_order(self.feature_order))

    def model(self) -> Any:
        return self.make_model(self.feature_order)


class FeatureTable:
    """Filas de features por `example_id`, con auditoría opcional de accesos.

    La auditoría existe para el test espía de A3.2: registra qué ids se leyeron y en qué
    momento, de modo que "una fila externa nunca participa en el fit" se comprueba en vez de
    argumentarse.
    """

    def __init__(
        self,
        features: Mapping[str, Mapping[str, float]],
        labels: Mapping[str, int],
        *,
        audit: bool = False,
    ) -> None:
        self._features = dict(features)
        self._labels = dict(labels)
        self.audit = bool(audit)
        self.phase: str = "unset"
        self.touched: list[tuple[str, str]] = []
        faltan = set(self._features) - set(self._labels)
        if faltan:
            raise ModelError(f"{len(faltan)} ids sin etiqueta, p.ej. {sorted(faltan)[:3]}")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._features)

    def _record(self, ids: Sequence[str]) -> None:
        if self.audit:
            self.touched.extend((self.phase, cid) for cid in ids)

    def ids_in_phase(self, phase: str) -> set[str]:
        """Qué filas se leyeron durante una fase. Base del test espía de A3.2."""
        return {cid for ph, cid in self.touched if ph == phase}

    def matrix(self, ids: Sequence[str], feature_order: Sequence[str]) -> np.ndarray:
        self._record(ids)
        order = tuple(feature_order)
        out = np.empty((len(ids), len(order)), dtype=np.float64)
        for i, cid in enumerate(ids):
            row = self._features.get(cid)
            if row is None:
                raise ModelError(f"no hay features para {cid!r}")
            for j, key in enumerate(order):
                if key not in row:
                    raise ModelError(f"a {cid!r} le falta la feature {key!r}")
                out[i, j] = float(row[key])
        if not np.isfinite(out).all():
            raise ModelError("la tabla de features contiene NaN o inf; nan_policy es reject")
        return out

    def y(self, ids: Sequence[str]) -> np.ndarray:
        self._record(ids)
        return check_labels(np.array([self._labels[cid] for cid in ids], dtype=np.int64))


def balanced_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Umbral que maximiza la exactitud balanceada.

    Balanceada y no exactitud a secas porque la prevalencia **no es estable entre splits**
    (D-A1.5: train 59.9 % sintético, val 47.9 %) y el set oculto probablemente tampoco la
    comparte. Un umbral que optimiza exactitud cruda se casa con la tasa base de train.
    """
    s = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    finite = np.isfinite(s)
    if not finite.any():
        return 0.5
    s, labels = s[finite], labels[finite]
    n_pos, n_neg = float(labels.sum()), float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    uniq = np.unique(s)
    cortes = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else uniq
    mejor, mejor_corte = -1.0, 0.5
    for corte in cortes:
        pred = s > corte
        tp = float(labels[pred].sum())
        fp = float(pred.sum() - tp)
        bal = 0.5 * (tp / n_pos + (n_neg - fp) / n_neg)
        if bal > mejor:
            mejor, mejor_corte = bal, float(corte)
    return mejor_corte


class LogisticFusion:
    """Apila los scores de rama con una logística en NumPy puro.

    Se ajusta sobre los scores OOF **internos**, nunca sobre los externos. Con una sola rama
    es una transformación monótona, así que no cambia el ranking ni el AUC — y eso es lo
    correcto: la fusión no debe inventar señal donde hay una sola fuente.
    """

    name = "logistic_fusion@1"

    def __init__(self, l2: float = 1.0, max_iter: int = 200, tol: float = 1e-9) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.branch_names_: tuple[str, ...] = ()

    def fit(self, S: np.ndarray, y: np.ndarray, branch_names: Sequence[str]) -> "LogisticFusion":
        X = np.asarray(S, dtype=np.float64)
        labels = check_labels(y)
        if X.ndim != 2 or X.shape[0] != labels.shape[0]:
            raise ModelError(f"S {X.shape} no cuadra con {labels.shape[0]} etiquetas")
        n_feat = X.shape[1]
        w = np.zeros(n_feat, dtype=np.float64)
        b = 0.0
        for _ in range(self.max_iter):
            p = _sigmoid(X @ w + b)
            r = p - labels
            g_w = X.T @ r + self.l2 * w
            g_b = float(r.sum())
            s = p * (1.0 - p) + 1e-12
            H = (X * s[:, None]).T @ X + self.l2 * np.eye(n_feat)
            h_wb = X.T @ s
            h_bb = float(s.sum())
            # Bloque completo [[H, h_wb], [h_wb.T, h_bb]]
            top = np.hstack([H, h_wb[:, None]])
            bottom = np.hstack([h_wb[None, :], np.array([[h_bb]])])
            full = np.vstack([top, bottom])
            grad = np.concatenate([g_w, [g_b]])
            try:
                step = np.linalg.solve(full, grad)
            except np.linalg.LinAlgError:
                break
            w, b = w - step[:-1], b - float(step[-1])
            if np.max(np.abs(step)) < self.tol:
                break
        if not (np.isfinite(w).all() and np.isfinite(b)):
            raise ModelError("la fusión divergió")
        self.w_, self.b_ = w, float(b)
        self.branch_names_ = tuple(branch_names)
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise ModelError("la fusión no está ajustada")
        X = np.asarray(S, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.w_.shape[0]:
            raise ModelError(f"S {X.shape} no cuadra con {self.w_.shape[0]} ramas")
        return _sigmoid(X @ self.w_ + self.b_)

    def to_dict(self) -> dict[str, Any]:
        if self.w_ is None:
            raise ModelError("la fusión no está ajustada")
        return {
            "name": self.name,
            "branches": list(self.branch_names_),
            "w": [float(v) for v in self.w_],
            "b": float(self.b_),
        }


@fusions.register("logistic", version=1)
def logistic_fusion(**kwargs: Any) -> LogisticFusion:
    """Apilado logístico de scores de rama. NumPy puro, entra al bundle."""
    return LogisticFusion(**kwargs)


@dataclass(slots=True)
class FoldAudit:
    """Qué se ajustó con qué, dentro de un fold externo. Va a `extra` de `FoldResult`."""

    n_train: int
    n_test: int
    inner_folds: int
    branch_inner_auc: dict[str, float] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)
    calibrator: dict[str, Any] = field(default_factory=dict)
    threshold_source: str = "inner_oof_balanced_accuracy"


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    finite = np.isfinite(s)
    y, s = np.asarray(y)[finite], np.asarray(s)[finite]
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


def make_nested_fit_predict(
    table: FeatureTable,
    branches: Sequence[Branch],
    *,
    calibrator_factory: Callable[[], Any],
    fusion_factory: Callable[[], Any] = logistic_fusion,
    inner_folds: int = DEFAULT_INNER_FOLDS,
) -> Callable[[Sequence[str], Sequence[str], np.ndarray], tuple[np.ndarray, float, dict]]:
    """Devuelve un `FitPredict` para `protocol.nested_cv` con todo congelado dentro de IN_k."""
    if not branches:
        raise ModelError("hace falta al menos una rama")
    nombres = [b.name for b in branches]
    if len(set(nombres)) != len(nombres):
        raise ModelError(f"nombres de rama repetidos: {nombres}")

    def fit_predict(
        train_ids: Sequence[str], test_ids: Sequence[str], g_in: np.ndarray
    ) -> tuple[np.ndarray, float, dict]:
        train_ids, test_ids = tuple(train_ids), tuple(test_ids)
        table.phase = "inner"
        y_tr = table.y(train_ids)

        inner = grouped_inner_splits(y_tr, g_in, n_folds=inner_folds)

        # --- 1. Scores OOF INTERNOS por rama. Solo con IN_k. ---
        S_in = np.full((len(train_ids), len(branches)), np.nan, dtype=np.float64)
        for j, branch in enumerate(branches):
            X_tr = table.matrix(train_ids, branch.feature_order)
            for a, b in inner:
                m = branch.model()
                m.fit(X_tr[a], y_tr[a])
                S_in[b, j] = m.p_synthetic(X_tr[b])

        usable = np.isfinite(S_in).all(axis=1)
        if usable.sum() < 4:
            raise ModelError(
                f"solo {int(usable.sum())} filas internas utilizables; los splits internos "
                "no están cubriendo IN_k"
            )

        # --- 2. Fusión, calibrador y umbral: SOLO con los scores internos. ---
        fusion = fusion_factory()
        fusion.fit(S_in[usable], y_tr[usable], nombres)
        fused_in = fusion.transform(S_in[usable])

        calibrator = calibrator_factory()
        calibrator.fit(fused_in, y_tr[usable])
        cal_in = calibrator.transform(fused_in)

        thr = balanced_threshold(cal_in, y_tr[usable])

        auditoria = FoldAudit(
            n_train=len(train_ids),
            n_test=len(test_ids),
            inner_folds=len(inner),
            branch_inner_auc={
                b.name: _auc(y_tr[usable], S_in[usable, j]) for j, b in enumerate(branches)
            },
            fusion=fusion.to_dict(),
            calibrator=calibrator.to_dict(),
        )

        # --- 3. Refit de cada rama sobre IN_k completo y UNA predicción de OUT_k. ---
        # Las fases hacen auditable el candado: "refit" no puede tocar OUT_k, y "predict"
        # solo toca OUT_k. El test espía lo comprueba en vez de creerlo.
        S_out = np.empty((len(test_ids), len(branches)), dtype=np.float64)
        modelos = []
        table.phase = "refit"
        for branch in branches:
            m = branch.model()
            m.fit(table.matrix(train_ids, branch.feature_order), y_tr)
            modelos.append(m)

        table.phase = "predict"
        for j, (branch, m) in enumerate(zip(branches, modelos, strict=True)):
            S_out[:, j] = m.p_synthetic(table.matrix(test_ids, branch.feature_order))

        table.phase = "unset"
        scores = calibrator.transform(fusion.transform(S_out))
        return scores, float(thr), {"audit": auditoria}

    return fit_predict


class DisagreementFusion:
    """Backbone acústico + booster conductual + regla de desacuerdo.

    La decisión de arquitectura del plan: cuando las ramas discrepan fuerte, **gana el
    backbone y baja la confianza**. No se promedia, porque promediar dos ramas que se
    contradicen produce un 0.5 que se lee como "duda calibrada" cuando en realidad es
    "una de las dos está muy equivocada y no sé cuál".

    Dos parámetros, con estatus distinto y declarado:

    - `delta_` — **se aprende** dentro del cross-fitting: el cuantil `q` de los desacuerdos
      observados en los scores OOF internos. Así "discrepan fuerte" significa "más que el
      q % de los casos de este fold", no un número inventado.
    - `shrink` — **se declara**, no se optimiza. Es política de producto: cuánta confianza
      se cede cuando el sistema sabe que sus ramas no se entienden. Optimizarlo sobre AUC
      con n=282 sería ajustar una decisión de producto a ruido muestral.
    """

    name = "disagreement@1"

    def __init__(
        self,
        backbone: str = "acoustic",
        *,
        quantile: float = 0.90,
        shrink: float = 0.5,
        l2: float = 1.0,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ModelError("quantile debe estar en (0, 1)")
        if not 0.0 <= shrink <= 1.0:
            raise ModelError("shrink debe estar en [0, 1]")
        self.backbone = str(backbone)
        self.quantile = float(quantile)
        self.shrink = float(shrink)
        self._inner = LogisticFusion(l2=l2)
        self.delta_: float | None = None
        self.branch_names_: tuple[str, ...] = ()
        self._bb: int | None = None

    def fit(self, S: np.ndarray, y: np.ndarray, branch_names: Sequence[str]) -> "DisagreementFusion":
        nombres = tuple(branch_names)
        if self.backbone not in nombres:
            raise ModelError(f"backbone {self.backbone!r} no está entre las ramas {nombres}")
        if len(nombres) < 2:
            raise ModelError("la regla de desacuerdo necesita al menos dos ramas")
        self.branch_names_ = nombres
        self._bb = nombres.index(self.backbone)
        self._inner.fit(S, y, nombres)
        X = np.asarray(S, dtype=np.float64)
        self.delta_ = float(np.quantile(self._spread(X), self.quantile))
        return self

    def _spread(self, X: np.ndarray) -> np.ndarray:
        """Desacuerdo = distancia máxima entre el backbone y cualquier otra rama."""
        otras = [j for j in range(X.shape[1]) if j != self._bb]
        return np.max(np.abs(X[:, otras] - X[:, [self._bb]]), axis=1)

    def transform(self, S: np.ndarray) -> np.ndarray:
        if self.delta_ is None:
            raise ModelError("la fusión no está ajustada")
        X = np.asarray(S, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.branch_names_):
            raise ModelError(f"S {X.shape} no cuadra con {len(self.branch_names_)} ramas")
        fusionado = self._inner.transform(X)
        discrepan = self._spread(X) > self.delta_
        backbone = X[:, self._bb]
        # Se conserva la DECISIÓN del backbone (de qué lado de 0.5 cae) y se encoge hacia
        # 0.5 la distancia, que es exactamente "el veredicto es suyo, la confianza baja".
        atenuado = 0.5 + self.shrink * (backbone - 0.5)
        return np.where(discrepan, atenuado, fusionado)

    def to_dict(self) -> dict[str, Any]:
        if self.delta_ is None:
            raise ModelError("la fusión no está ajustada")
        return {
            "name": self.name,
            "branches": list(self.branch_names_),
            "backbone": self.backbone,
            "delta": self.delta_,
            "delta_source": f"cuantil {self.quantile} del desacuerdo OOF interno",
            "shrink": self.shrink,
            "shrink_source": "declarado, no optimizado",
            "inner": self._inner.to_dict(),
        }


@fusions.register("disagreement", version=1)
def disagreement_fusion(**kwargs: Any) -> DisagreementFusion:
    """Backbone + booster con regla de desacuerdo. NumPy puro."""
    return DisagreementFusion(**kwargs)
