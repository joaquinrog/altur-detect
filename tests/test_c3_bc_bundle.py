from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "build_bundle_c3_bc.py"
spec = importlib.util.spec_from_file_location("build_bundle_c3_bc", MODULE)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


def _corpus(n_groups: int = 10):
    rng = np.random.default_rng(7)
    order = builder.feature_order_from_lfcc()
    features, labels, groups = {}, {}, {}
    for i in range(372):
        cid = f"row-{i:03d}"
        labels[cid] = int(i >= 128)
        groups[cid] = f"g-{i % n_groups}"
        features[cid] = {name: float(rng.normal() + labels[cid]) for name in order}
    return {"features": features, "labels": labels, "groups": groups, "fingerprint": "test"}


def test_feature_order_excludes_exactly_static_means():
    order = builder.feature_order_from_lfcc()
    assert len(order) == 100
    assert not any("static" in name and name.endswith("_mean") for name in order)


def test_feature_projection_works_for_lfcc_control_namespaces():
    full = tuple(
        f"spectral_factory.lfcc.ch1.{kind}{i}_{stat}"
        for kind in ("static", "delta", "deltadelta")
        for i in range(20)
        for stat in ("mean", "std")
    )

    projected = builder.feature_order_without_static_means(full)

    assert len(projected) == 100
    assert projected[0] == "spectral_factory.lfcc.ch1.static0_std"


def test_fit_is_deterministic_and_has_one_oof_per_row():
    pytest.importorskip("sklearn")
    corpus = _corpus()
    a = builder.fit_c3_corpus(corpus)
    b = builder.fit_c3_corpus(corpus)
    np.testing.assert_array_equal(a["oof"], b["oof"])
    assert len(a["oof"]) == 372
    assert all(f["n_test"] > 0 for f in a["folds"])
    assert {name: block["n"] for name, block in a["blocks"].items()} == {
        "static_std": 20,
        "delta": 40,
        "deltadelta": 40,
    }
    assert 0 <= a["oof_auc"] <= 1
    assert 0 <= a["oof_brier"] <= 1


def test_rejects_monoclass_fold():
    pytest.importorskip("sklearn")
    corpus = _corpus(n_groups=5)
    for key in corpus["groups"]:
        corpus["groups"][key] = "one" if corpus["labels"][key] else key
    with pytest.raises(ValueError, match="cinco folds"):
        builder.fit_c3_corpus(corpus)


def test_destination_is_fail_closed():
    with pytest.raises(ValueError, match="solo puede escribir"):
        builder.build_bundle(_corpus(), out=ROOT / "models" / "current" / "c3")
