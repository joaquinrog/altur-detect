"""Grupos, particiones y cross-fitting anidado. El contrato estadistico.

REGLA DE INTEGRACION: prohibido editar en paralelo (AGENTS.md regla 1).

Aqui vive la correccion mas importante del plan (§3): **que los scores de rama sean
out-of-fold no basta** si la fusion y el calibrador se ajustan sobre esas mismas filas y
ahi se evaluan. Y `CalibratedClassifierCV` NO acepta `groups`: usa `StratifiedKFold`
interno, o sea que mete fuga de hablante justo en la calibracion. Por eso la calibracion
lleva aqui su propio splitter por grupo.

Tres piezas:

  1. **Grupos por componentes conectados** — no por una llave suelta. Se congelan en
     `configs/protocol/groups_v1.csv`: si se recalculan por corrida, los resultados de
     tres personas dejan de ser comparables.
  2. **Protocolos** — `official_v1` (juez `val` intacto) y `pooled5_v1` (stretch, contamina).
  3. **Cross-fitting anidado** — todo se ajusta dentro del fold externo y se predice una vez.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .dataset import Dataset

PROTOCOL_VERSION = "groups_v1"
DEFAULT_GROUPS_CSV = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "protocol" / "groups_v1.csv"
)

# 🔴 Llaves que UNEN dos llamadas en el mismo grupo. La lista es corta a proposito.
#
# `vendor` y `accent_country` NO estan aqui, y es la trampa principal de esta funcion:
# unir por vendor colapsaria TODAS las sinteticas en un solo componente, quedandose sin
# particiones utilizables. Son ejes de estratificacion y de evaluacion, no de union.
# Ver plan §4 — la union es por: hablante/voz, llamada donante, linea de guion, derivados.
LINKING_KEYS: tuple[str, ...] = (
    "speaker_or_voice_id",
    "donor_call_id",
    "script_line_id",
    "derived_from",
)

N_FOLDS = 5
SEED = 20260912


class ProtocolError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 1. Grupos por componentes conectados
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # compresion de camino
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Al menor lexicografico, para que el resultado no dependa del orden de llegada.
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """Asignacion llamada -> grupo, mas la evidencia de si vale algo.

    `conservative=True` significa que no hubo evidencia de que los clusters fueran
    hablantes y se cayo al fallback grupo = llamada. **Eso se declara en el reporte**,
    no se esconde: cambia lo que se puede afirmar sobre generalizacion.
    """

    group_of: dict[str, str]
    conservative: bool
    warnings: tuple[str, ...] = ()
    linking_keys_used: tuple[str, ...] = ()

    @property
    def n_groups(self) -> int:
        return len(set(self.group_of.values()))

    def groups_for(self, ids: Sequence[str]) -> np.ndarray:
        return np.array([self.group_of[i] for i in ids])


def build_groups(ds: Dataset, ids: Sequence[str] | None = None) -> GroupAssignment:
    """Componentes conectados sobre las llaves de union disponibles.

    OBS (2026-09-12, medido sobre el release v1.0): el `manifest.csv` oficial trae
    `anon_id`, `label`, `split` y `duration_s` — **y ninguna llave de union**. No hay
    `speaker_or_voice_id`, ni `donor_call_id`, ni linea de guion. Asi que hoy esta funcion
    devuelve necesariamente `conservative=True` y grupo = llamada.

    No es codigo muerto: en cuanto entren clones, donantes o augmentaciones al corpus
    ampliado, esos derivados SI traen `derived_from` y deben caer en la particion de su
    origen. Si esto se escribiera despues, se escribiria tarde y con folds ya corridos.
    """
    ids = tuple(ds.ids if ids is None else ids)
    uf = _UnionFind()
    used: set[str] = set()

    for cid in ids:
        uf.add(cid)
    for cid in ids:
        rec = ds.record(cid)
        for key in LINKING_KEYS:
            val = rec.groups.get(key)
            if val in (None, "", []):
                continue
            used.add(key)
            for v in val if isinstance(val, (list, tuple)) else [val]:
                uf.union(cid, f"{key}={v}")

    # El id del grupo es el menor `anon_id` de sus miembros: determinista y legible.
    members: dict[str, list[str]] = {}
    for cid in ids:
        members.setdefault(uf.find(cid), []).append(cid)
    group_of = {cid: min(ms) for ms in members.values() for cid in ms}

    warnings = _validate_groups(ds, ids, group_of)
    conservative = not used or any(w.startswith("FALLBACK") for w in warnings)
    if conservative:
        group_of = {cid: cid for cid in ids}

    return GroupAssignment(
        group_of=group_of,
        conservative=conservative,
        warnings=tuple(warnings),
        linking_keys_used=tuple(sorted(used)),
    )


def _validate_groups(
    ds: Dataset, ids: Sequence[str], group_of: dict[str, str]
) -> list[str]:
    """Los grupos son ESTIMADOS. Validarlos es obligatorio (plan §4).

    Dos modos de fallo que se ven iguales en la tabla y significan cosas opuestas:
    un cluster que en realidad captura la CLASE, y un cluster que en realidad es un CANAL.
    """
    out: list[str] = []
    by_group: dict[str, list[str]] = {}
    for cid in ids:
        by_group.setdefault(group_of[cid], []).append(cid)

    multi = {g: ms for g, ms in by_group.items() if len(ms) > 1}
    if not multi:
        out.append(
            "FALLBACK: ningun grupo tiene mas de una llamada. Sin llaves de union no hay "
            "evidencia de agrupacion por hablante; grupo = llamada, y se declara."
        )
        return out

    degenerate = [
        g for g, ms in multi.items() if len({ds.record(m).label for m in ms}) == 1
    ]
    if len(degenerate) == len(multi):
        out.append(
            f"CLASE: los {len(multi)} grupos multi-llamada son monoclase. El cluster puede "
            "estar capturando la etiqueta y no al hablante."
        )
    sizes = sorted((len(ms) for ms in by_group.values()), reverse=True)
    if sizes[0] > 0.25 * len(ids):
        out.append(
            f"TAMANO: el grupo mayor tiene {sizes[0]}/{len(ids)} llamadas. Un componente "
            "gigante suele ser una llave de union mal elegida (p.ej. vendor)."
        )
    return out


def freeze_groups(ga: GroupAssignment, path: Path = DEFAULT_GROUPS_CSV) -> Path:
    """Escribe `configs/protocol/groups_v1.csv`. Determinista: mismo input, mismos bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["call_id", "group_id"])
        for cid in sorted(ga.group_of):
            w.writerow([cid, ga.group_of[cid]])
    return path


def load_groups(path: Path = DEFAULT_GROUPS_CSV) -> dict[str, str]:
    if not path.exists():
        raise ProtocolError(
            f"no existe {path}. Corre `python scripts/build_protocol.py` para congelarlo. "
            "Recalcular los grupos por corrida hace incomparables los resultados."
        )
    with path.open(newline="", encoding="utf-8") as f:
        return {r["call_id"]: r["group_id"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 2. Protocolos
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Protocol:
    """Que se entrena, que se evalua, y si contamina al juez."""

    name: str
    fit_ids: tuple[str, ...]
    judge_ids: tuple[str, ...]
    n_folds: int = N_FOLDS
    seed: int = SEED
    val_contaminated: bool = False

    def outer_folds(
        self, ds: Dataset, groups: dict[str, str]
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        return _grouped_folds(ds, self.fit_ids, groups, self.n_folds, self.seed)


def official_v1(ds: Dataset) -> Protocol:
    """`val` (71) intacto como juez final; `StratifiedGroupKFold(5)` dentro de los 282."""
    return Protocol(
        name="official_v1",
        fit_ids=ds.ids_for("train"),
        judge_ids=ds.ids_for("val"),
        val_contaminated=False,
    )


def pooled5_v1(ds: Dataset) -> Protocol:
    """Stretch: `StratifiedGroupKFold(5)` sobre las 353.

    🔴 Aqui el modelo VE audio de `val`. Un detector entrenado asi **no puede evaluarse
    contra `official.val` y afirmar independencia**. La bateria lo rechaza salvo flag
    explicito y la fila sale marcada `val_contaminated=true`.
    """
    return Protocol(
        name="pooled5_v1",
        fit_ids=ds.ids,
        judge_ids=(),
        val_contaminated=True,
    )


PROTOCOLS: dict[str, Callable[[Dataset], Protocol]] = {
    "official_v1": official_v1,
    "pooled5_v1": pooled5_v1,
}


def get_protocol(name: str, ds: Dataset) -> Protocol:
    try:
        return PROTOCOLS[name](ds)
    except KeyError:
        raise ProtocolError(
            f"protocolo desconocido: {name!r}. Hay: {sorted(PROTOCOLS)}"
        ) from None


def _grouped_folds(
    ds: Dataset,
    ids: Sequence[str],
    groups: dict[str, str],
    n_folds: int,
    seed: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    ids = tuple(ids)
    missing = [i for i in ids if i not in groups]
    if missing:
        raise ProtocolError(f"{len(missing)} ids sin grupo congelado, p.ej. {missing[:3]}")
    y = np.array([ds.record(i).label for i in ids])
    g = np.array([groups[i] for i in ids])

    # `shuffle=True` + `random_state` porque sin barajar el resultado depende del orden de
    # llegada, y ese orden es alfabetico por anon_id: un artefacto, no una decision.
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.asarray(ids)
    return [
        (tuple(arr[tr]), tuple(arr[te])) for tr, te in skf.split(np.zeros(len(ids)), y, g)
    ]


def grouped_inner_splits(
    y: np.ndarray, g: np.ndarray, n_folds: int = 3, seed: int = SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splitter por grupo para la CALIBRACION.

    🔴 Existe porque `CalibratedClassifierCV(cv=3)` usa `StratifiedKFold` interno y **no
    acepta `groups`**: calibrar con el va a meter el mismo hablante a los dos lados y a
    producir una curva de calibracion optimista. Se le pasa `cv=grouped_inner_splits(...)`.

    Degrada a `StratifiedKFold` solo si algun fold quedaria monoclase — con grupos grandes
    y n chico puede pasar — y lo dice en voz alta en vez de fallar en silencio.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    try:
        folds = [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y, g)]
    except ValueError:
        folds = []
    if not folds or any(len(np.unique(y[te])) < 2 for _, te in folds):
        import warnings as _w

        _w.warn(
            "calibracion: no se pudo partir por grupo sin folds monoclase; se cae a "
            "StratifiedKFold. La calibracion queda OPTIMISTA y debe declararse.",
            stacklevel=2,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]
    return folds


# ---------------------------------------------------------------------------
# 3. Cross-fitting anidado
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FoldResult:
    fold: int
    test_ids: tuple[str, ...]
    y_true: np.ndarray
    y_score: np.ndarray
    threshold: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NestedResult:
    """Concatenacion de los folds externos. Las metricas se calculan sobre esto."""

    protocol: str
    folds: list[FoldResult]
    conservative_groups: bool
    group_warnings: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(i for f in self.folds for i in f.test_ids)

    @property
    def y_true(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds])

    @property
    def y_score(self) -> np.ndarray:
        return np.concatenate([f.y_score for f in self.folds])


# Firma de lo que el runner enchufa: recibe SOLO ids de entrenamiento y de test, y
# devuelve (scores del test, umbral, extras). Todo ajuste ocurre dentro.
FitPredict = Callable[
    [Sequence[str], Sequence[str], np.ndarray], tuple[np.ndarray, float, dict[str, Any]]
]


def nested_cv(
    ds: Dataset,
    protocol: Protocol,
    fit_predict: FitPredict,
    groups: dict[str, str] | None = None,
    *,
    conservative: bool = True,
    warnings_: tuple[str, ...] = (),
) -> NestedResult:
    """El bucle del plan §3.

    Para cada fold externo k: `IN_k` es todo menos k. Dentro de `IN_k`, y SOLO ahi, se
    augmenta (despues de partir, nunca antes), se ajusta imputacion y seleccion, cada
    modelo de rama con folds internos por grupo, la fusion sobre los scores internos OOF,
    el calibrador sobre la salida de la fusion, y se fija el umbral. Se predice `OUT_k`
    **una sola vez** con todo eso congelado.

    Ninguna decision aprendida —incluido el umbral— toca el fold externo antes de
    predecirlo. Esta funcion no puede impedir que un `fit_predict` mal escrito mire
    `test_ids`; lo que hace es que mirarlos sea visible en el diff.
    """
    groups = load_groups() if groups is None else groups
    results: list[FoldResult] = []
    for k, (train_ids, test_ids) in enumerate(protocol.outer_folds(ds, groups)):
        g_in = np.array([groups[i] for i in train_ids])
        scores, thr, extra = fit_predict(train_ids, test_ids, g_in)
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(test_ids),):
            raise ProtocolError(
                f"fold {k}: fit_predict devolvio {scores.shape}, esperaba ({len(test_ids)},)"
            )
        results.append(
            FoldResult(
                fold=k,
                test_ids=tuple(test_ids),
                y_true=np.array([ds.record(i).label for i in test_ids]),
                y_score=scores,
                threshold=float(thr),
                extra=extra,
            )
        )
    return NestedResult(
        protocol=protocol.name,
        folds=results,
        conservative_groups=conservative,
        group_warnings=warnings_,
    )


def permute_labels_within_groups(
    y: np.ndarray, g: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Permuta etiquetas DENTRO de cada grupo. Base del test de fuga (plan §3).

    Si tras permutar el AUC no cae a ~0.5, algo se esta ajustando fuera de su fold: la
    unica senal que queda es la que el protocolo no debio dejar pasar.

    Con grupos de una sola llamada esto permuta dentro de un grupo de tamano 1, o sea no
    permuta nada. Por eso `leak_test` permuta globalmente cuando los grupos son
    conservadores — y lo dice.
    """
    rng = np.random.default_rng(seed)
    out = y.copy()
    for grp in np.unique(g):
        idx = np.flatnonzero(g == grp)
        out[idx] = rng.permutation(y[idx])
    return out
