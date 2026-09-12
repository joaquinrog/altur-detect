"""Tests de `protocol.py`. Aqui se prueba lo que el plan §3 y §4 afirman.

Casi todos usan un dataset falso con llaves de union inventadas: el dataset OFICIAL no
trae ninguna (OBS 2026-09-12), asi que probar solo con el no ejercitaria los componentes
conectados — que es justo la parte que tiene que funcionar cuando llegue el corpus ampliado.
"""

from __future__ import annotations

import numpy as np
import pytest

from altur.protocol import (
    LINKING_KEYS,
    ProtocolError,
    build_groups,
    freeze_groups,
    get_protocol,
    grouped_inner_splits,
    load_groups,
    nested_cv,
    permute_labels_within_groups,
)
from altur.types import DatasetRecord


class FakeDataset:
    """Lo minimo que `protocol.py` consume de un `Dataset`."""

    def __init__(self, recs: dict[str, tuple[int, str, dict]]) -> None:
        self._r = recs
        self.ids = tuple(sorted(recs))

    def __len__(self) -> int:
        return len(self._r)

    def ids_for(self, split: str) -> tuple[str, ...]:
        return tuple(i for i in self.ids if self._r[i][1] == split)

    def record(self, i: str) -> DatasetRecord:
        label, split, groups = self._r[i]
        return DatasetRecord(example_id=i, label=label, split=split,
                             groups={"call_id": i, **groups})

    def fingerprint(self) -> str:
        return "fake"


def make(n: int = 40, links: bool = False) -> FakeDataset:
    recs = {}
    for k in range(n):
        cid = f"c{k:03d}"
        extra = {"speaker_or_voice_id": f"spk{k // 4}"} if links else {}
        recs[cid] = (k % 2, "train" if k < n - 10 else "val", extra)
    return FakeDataset(recs)


# -- grupos -----------------------------------------------------------------


def test_sin_llaves_cae_al_fallback_conservador() -> None:
    """El caso REAL del dataset oficial: grupo = llamada, y se declara."""
    ga = build_groups(make(links=False))
    assert ga.conservative
    assert ga.n_groups == 40
    assert ga.linking_keys_used == ()
    assert any(w.startswith("FALLBACK") for w in ga.warnings)


def test_componentes_conectados_unen_por_hablante() -> None:
    ga = build_groups(make(links=True))
    assert not ga.conservative
    assert ga.n_groups == 10                      # 40 llamadas / 4 por hablante
    assert ga.group_of["c000"] == ga.group_of["c003"]
    assert ga.group_of["c000"] != ga.group_of["c004"]


def test_union_es_transitiva() -> None:
    """A-B por hablante y B-C por donante debe dejar a A, B y C en el mismo grupo."""
    ds = FakeDataset({
        "a": (0, "train", {"speaker_or_voice_id": "s1"}),
        "b": (1, "train", {"speaker_or_voice_id": "s1", "donor_call_id": "d9"}),
        "c": (1, "train", {"donor_call_id": "d9"}),
        "z": (0, "train", {}),
    })
    g = build_groups(ds).group_of
    assert g["a"] == g["b"] == g["c"] != g["z"]


def test_vendor_no_es_llave_de_union() -> None:
    """Unir por vendor colapsaria todas las sinteticas en un componente. Es la trampa."""
    assert "vendor" not in LINKING_KEYS
    assert "accent_country" not in LINKING_KEYS


def test_grupos_son_deterministas() -> None:
    ds = make(links=True)
    assert build_groups(ds).group_of == build_groups(ds).group_of


def test_congelar_y_releer(tmp_path) -> None:
    ga = build_groups(make(links=True))
    p = freeze_groups(ga, tmp_path / "g.csv")
    assert load_groups(p) == ga.group_of
    assert p.read_bytes() == freeze_groups(ga, tmp_path / "g2.csv").read_bytes()


def test_leer_grupos_que_no_existen(tmp_path) -> None:
    with pytest.raises(ProtocolError, match="Recalcular"):
        load_groups(tmp_path / "nope.csv")


# -- protocolos --------------------------------------------------------------


def test_official_v1_no_toca_val() -> None:
    ds = make()
    proto = get_protocol("official_v1", ds)
    assert set(proto.fit_ids).isdisjoint(proto.judge_ids)
    assert not proto.val_contaminated
    folds = proto.outer_folds(ds, build_groups(ds).group_of)
    for _tr, te in folds:
        assert set(te).isdisjoint(proto.judge_ids)


def test_pooled5_se_marca_contaminado() -> None:
    proto = get_protocol("pooled5_v1", make())
    assert proto.val_contaminated
    assert len(proto.fit_ids) == 40


def test_folds_no_parten_un_grupo() -> None:
    """La propiedad que justifica todo el archivo."""
    ds = make(links=True)
    groups = build_groups(ds).group_of
    proto = get_protocol("official_v1", ds)
    for tr, te in proto.outer_folds(ds, groups):
        assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in te})


def test_folds_cubren_todo_sin_solaparse() -> None:
    ds = make()
    proto = get_protocol("official_v1", ds)
    folds = proto.outer_folds(ds, build_groups(ds).group_of)
    tests = [set(te) for _, te in folds]
    assert set().union(*tests) == set(proto.fit_ids)
    assert sum(map(len, tests)) == len(proto.fit_ids)


def test_id_sin_grupo_falla() -> None:
    ds = make()
    groups = build_groups(ds).group_of
    groups.pop(ds.ids[0])
    with pytest.raises(ProtocolError, match="sin grupo congelado"):
        get_protocol("official_v1", ds).outer_folds(ds, groups)


def test_protocolo_desconocido() -> None:
    with pytest.raises(ProtocolError, match="desconocido"):
        get_protocol("no_existe", make())


# -- calibracion y cross-fitting ---------------------------------------------


def test_calibracion_parte_por_grupo() -> None:
    """🔴 La razon de existir de `grouped_inner_splits`: CalibratedClassifierCV no acepta groups."""
    y = np.array([0, 1] * 20)
    g = np.repeat([f"s{k}" for k in range(10)], 4)
    for tr, te in grouped_inner_splits(y, g, n_folds=3):
        assert set(g[tr]).isdisjoint(set(g[te]))


def test_calibracion_avisa_si_degrada() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    g = np.array(["a"] * 3 + ["b"] * 3)          # cada grupo es monoclase
    with pytest.warns(UserWarning, match="OPTIMISTA"):
        grouped_inner_splits(y, g, n_folds=2)


def test_nested_cv_predice_cada_llamada_una_vez() -> None:
    ds = make()
    groups = build_groups(ds).group_of
    proto = get_protocol("official_v1", ds)
    vistos: list[str] = []

    def fp(train_ids, test_ids, g_in):
        assert set(train_ids).isdisjoint(test_ids)
        vistos.extend(test_ids)
        return np.full(len(test_ids), 0.5), 0.5, {}

    res = nested_cv(ds, proto, fp, groups)
    assert len(vistos) == len(set(vistos)) == len(proto.fit_ids)
    assert len(res.y_score) == len(proto.fit_ids)


def test_nested_cv_rechaza_scores_de_largo_equivocado() -> None:
    ds = make()
    with pytest.raises(ProtocolError, match="esperaba"):
        nested_cv(ds, get_protocol("official_v1", ds),
                  lambda tr, te, g: (np.zeros(3), 0.5, {}), build_groups(ds).group_of)


def test_permutacion_respeta_el_grupo() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    out = permute_labels_within_groups(y, g)
    for grp in np.unique(g):
        m = g == grp
        assert sorted(out[m]) == sorted(y[m])     # misma composicion dentro del grupo
