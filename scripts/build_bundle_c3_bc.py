"""Construye C3 D-A8.3 sin leer `val` ni escribir en `models/current`.

El corpus se importa de forma diferida desde ``build_c3_bc_corpus``. El contrato mínimo
para el integrador es un mapping (o un objeto con esos atributos) con ``features`` (id ->
mapping de 120 LFCC), ``labels`` (id -> 0/1), ``groups`` (id -> grupo anti-fuga) y, opcionalmente,
``fingerprint``. Los ids solo se usan en memoria y nunca se incluyen en el manifest/reporte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from altur import bundle as bundle_mod
from altur.ablation import auc, brier
from altur.crossfit import balanced_threshold
from altur.models.adapters import logreg
from altur.models.calibration import PlattCalibrator
from altur.provenance import effective_code_digest
from build_bundle import _requirements_lock
from run_experiment import load_spec

SPEC = ROOT / "configs" / "experiments" / "a8_3_c3_bc_v1.yaml"
DESTINATION = ROOT / "models" / "candidates" / "c3_bc_v1"
SEED = 20260912
FOLDS = 5
EXPECTED_ROWS = 372
EXPECTED_HUMAN = 128
EXPECTED_SYNTHETIC = 244


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def _get(corpus: Any, name: str) -> Any:
    if isinstance(corpus, dict):
        return corpus[name]
    return getattr(corpus, name)


def feature_order_without_static_means(full_order: tuple[str, ...]) -> tuple[str, ...]:
    order = tuple(
        name for name in full_order
        if not (".static" in name and name.endswith("_mean"))
    )
    removed = set(full_order) - set(order)
    if len(full_order) != 120 or len(order) != 100 or len(removed) != 20:
        raise RuntimeError("FEATURE_ORDER no produjo exactamente 100 features C3")
    return order


def feature_order_from_lfcc() -> tuple[str, ...]:
    from altur.features.spectral_factory_lfcc import FEATURE_ORDER

    return feature_order_without_static_means(FEATURE_ORDER)


def _validate_corpus(
    corpus: Any, order: tuple[str, ...]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, str]:
    features, labels, groups = _get(corpus, "features"), _get(corpus, "labels"), _get(corpus, "groups")
    ids = tuple(sorted(features))
    if len(ids) != EXPECTED_ROWS:
        raise ValueError(f"C3 requiere {EXPECTED_ROWS} filas, llegaron {len(ids)}")
    if set(ids) != set(labels) or set(ids) != set(groups):
        raise ValueError("features, labels y groups deben cubrir exactamente las mismas filas")
    y = np.asarray([labels[i] for i in ids], dtype=np.int64)
    g = np.asarray([str(groups[i]) for i in ids])
    if (
        set(np.unique(y)) != {0, 1}
        or int((y == 0).sum()) != EXPECTED_HUMAN
        or int((y == 1).sum()) != EXPECTED_SYNTHETIC
    ):
        raise ValueError("C3 requiere exactamente 128 humanas y 244 sintéticas")
    rows = {i: features[i] for i in ids}
    X = np.asarray(
        [[float(row[name]) for name in order] for row in (rows[i] for i in ids)],
        dtype=np.float64,
    )
    if not np.isfinite(X).all() or len(set(g)) < FOLDS:
        raise ValueError("features no finitas o menos de cinco grupos")
    fingerprint = str(getattr(corpus, "fingerprint", "") if not isinstance(corpus, dict) else corpus.get("fingerprint", ""))
    if not fingerprint:
        payload = json.dumps({"labels": y.tolist(), "groups": g.tolist()}, separators=(",", ":"), sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    return ids, X, y, g, fingerprint


def fit_c3_corpus(corpus: Any, feature_order: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Ajusta C3 y devuelve artefactos en memoria; no crea directorios ni archivos."""
    from sklearn.model_selection import StratifiedGroupKFold

    order = feature_order or feature_order_from_lfcc()
    ids, X, y, groups, fingerprint = _validate_corpus(corpus, order)
    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups))
    if len(folds) != FOLDS or any(len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2 for tr, te in folds):
        raise ValueError("cada uno de los cinco folds debe contener ambas clases en train y test")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_sizes = []
    for tr, te in folds:
        model = logreg(order, class_weight="balanced", max_iter=1000)
        model.fit(X[tr], y[tr])
        if np.isfinite(oof[te]).any():
            raise RuntimeError("una fila recibió más de un score OOF")
        oof[te] = model.p_synthetic(X[te])
        fold_sizes.append({
            "n_train": len(tr),
            "n_test": len(te),
            "n_groups_test": len(set(groups[te])),
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("no todas las filas recibieron exactamente un score OOF")
    from sklearn.linear_model import LogisticRegression

    platt_lr = LogisticRegression(max_iter=1000).fit(_logit(oof).reshape(-1, 1), y)
    calibrator = PlattCalibrator.from_dict({"a": float(platt_lr.coef_[0, 0]), "b": float(platt_lr.intercept_[0])})
    calibrated = calibrator.transform(oof)
    threshold = float(balanced_threshold(calibrated, y))
    final = logreg(order, class_weight="balanced", max_iter=1000)
    final.fit(X, y)
    export = final.export_linear(verify_on=X)
    block_indices = {
        "static_std": [i for i, name in enumerate(order) if ".static" in name],
        "delta": [i for i, name in enumerate(order) if ".delta" in name and ".deltadelta" not in name],
        "deltadelta": [i for i, name in enumerate(order) if ".deltadelta" in name],
    }
    blocks = {}
    for block, idx in block_indices.items():
        coef = export.coef[idx]
        blocks[block] = {
            "n": len(idx),
            "coef": [float(v) for v in coef],
            "l2": float(np.linalg.norm(coef)),
        }
    return {
        "ids": ids,
        "export": export,
        "calibrator": calibrator,
        "threshold": threshold,
        "oof": oof,
        "calibrated": calibrated,
        "oof_auc": float(auc(y, calibrated)),
        "oof_brier": float(brier(y, calibrated)),
        "fingerprint": fingerprint,
        "folds": fold_sizes,
        "blocks": blocks,
        "X": X,
        "y": y,
        "groups": groups,
    }


def _load_corpus(spec: dict[str, Any]) -> Any:
    import importlib

    module = importlib.import_module("build_c3_bc_corpus")
    for name in ("load_corpus", "build_corpus", "load_c3_bc_corpus"):
        loader = getattr(module, name, None)
        if loader is not None:
            return loader(spec=spec) if name != "load_c3_bc_corpus" else loader(spec)
    raise RuntimeError(
        "build_c3_bc_corpus debe exponer load_corpus(spec=...) o build_corpus(spec=...)"
    )


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def build_bundle(corpus: Any, *, out: Path = DESTINATION, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    if out.resolve() != DESTINATION.resolve() or "models/current" in out.resolve().as_posix():
        raise ValueError(f"C3 solo puede escribir en {DESTINATION}")
    spec = spec or load_spec(SPEC)
    order = tuple(spec["feature_order"])
    if order != feature_order_from_lfcc() or spec["transforms"] != [] or spec["split"] != "train":
        raise ValueError("la spec C3 no coincide con el contrato congelado")
    result = fit_c3_corpus(corpus, order)
    manifest = bundle_mod.BundleManifest(
        detector_name="spectral_factory_lfcc_c3_bc_logreg@1", code_digest=effective_code_digest(ROOT),
        data_fingerprint=result["fingerprint"], protocol_id="c3_bc_grouped5@1", run_id="c3_bc_v1",
        feature_order=list(order), extractor_refs=[spec["extractor"]], segmenter_ref=spec["segmenter"],
        transform_refs=[], calibrator_ref="platt@1", threshold=result["threshold"], min_seconds_declared=20.0,
        metrics={
            "n_train": EXPECTED_ROWS,
            "class_counts": {"human": EXPECTED_HUMAN, "synthetic": EXPECTED_SYNTHETIC},
            "seed": SEED,
            "folds": result["folds"],
            "oof": {
                "n": EXPECTED_ROWS,
                "exactly_one_per_row": True,
                "score_sha256": hashlib.sha256(result["oof"].tobytes()).hexdigest(),
                "auc": result["oof_auc"],
                "brier": result["oof_brier"],
            },
            "platt": {
                "a": result["calibrator"].a_,
                "b": result["calibrator"].b_,
                "source": "OOF logits only",
            },
            "threshold_source": "balanced_threshold over calibrated OOF scores",
            "coefficient_blocks": result["blocks"],
            "corpus_fingerprint": result["fingerprint"],
        },
        limitations=["OOF es diagnóstico interno, no evidencia de generalización a voces nuevas.",
                     "No se usa val; B/B2 quedan quemados para ajuste según D-A8.3.",
                     "Los IDs, voice IDs y rutas privadas no se incluyen en el bundle."],
    )
    path = bundle_mod.build_bundle(out, export=result["export"], manifest=manifest,
                                   calibrator=result["calibrator"], requirements=_requirements_lock())
    problems = bundle_mod.verify(out)
    if problems:
        raise RuntimeError(f"bundle C3 no verifica: {problems}")
    return {
        "bundle": out.name,
        "manifest_sha256": bundle_mod.sha256_file(path),
        "threshold": result["threshold"],
        "platt": {"a": result["calibrator"].a_, "b": result["calibrator"].b_},
        "n_features": len(order),
        "folds": result["folds"],
        "coefficient_blocks": result["blocks"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    spec = load_spec(SPEC)
    report = build_bundle(_load_corpus(spec), out=args.out, spec=spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
