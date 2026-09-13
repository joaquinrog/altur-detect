"""Congela C2 de D-A7.3 en un bundle: LFCC del freeze `cd028c7`, portado a NumPy.

La receta es la de D-A7.3, la que se midió en `val`, sin cambios:

- features: `spectral_factory.lfcc@1` con el VAD `spectral_factory.energy_adaptive@1`;
- modelo: estandarizado + LogisticRegression(class_weight="balanced", max_iter=1000) sobre las
  282 llamadas de `train`;
- calibrador: Platt ajustado con LogisticRegression (C=1) sobre el logit de scores OOF, con
  StratifiedKFold(5, shuffle, seed 0) y las llamadas ordenadas por id;
- umbral: `balanced_threshold` sobre esos scores calibrados.

No lee `val`. Con `--reference` compara lo que sale contra el export de D-A7.3 para demostrar
que lo servido es lo medido.

    python scripts/build_bundle_lfcc.py --out models/spectral_factory_lfcc_v1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from altur import bundle as bundle_mod
from altur.ablation import auc, bootstrap_ci, brier
from altur.crossfit import (
    Branch,
    FeatureTable,
    balanced_threshold,
    make_nested_fit_predict,
)
from altur.dataset import Dataset
from altur.models.adapters import logreg
from altur.models.calibration import PlattCalibrator
from altur.protocol import load_groups, nested_cv, official_v1, pooled5_v1
from altur.provenance import effective_code_digest
from build_bundle import _commit, _requirements_lock, feature_table
from run_experiment import load_spec

SPEC = ROOT / "configs" / "experiments" / "a7_spectral_factory_lfcc_v1.yaml"
VAL_LOOKS = ROOT / "experiments" / "val_looks.csv"
LR_PARAMS = {"class_weight": "balanced", "max_iter": 1000}
SEED = 0
FOLDS = 5
LOGIT_EPS = 1e-6  # el de D-A7.3 al ajustar Platt


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(q / (1 - q))


def fit_d_a7_3(X: np.ndarray, y: np.ndarray, feature_order: tuple[str, ...]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    oof = np.empty(len(y))
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y):
        m = logreg(feature_order, **LR_PARAMS)
        m.fit(X[tr], y[tr])
        oof[te] = m.p_synthetic(X[te])

    platt = LogisticRegression(max_iter=1000).fit(_logit(oof).reshape(-1, 1), y)
    calibrator = PlattCalibrator.from_dict(
        {"a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0])}
    )
    threshold = balanced_threshold(calibrator.transform(oof), y)

    final = logreg(feature_order, **LR_PARAMS)
    final.fit(X, y)
    return final.export_linear(verify_on=X), calibrator, float(threshold), oof


def compare_reference(path: Path, export, calibrator, threshold: float) -> dict[str, float]:
    ref = json.loads(path.read_text(encoding="utf-8"))
    short = [name.removeprefix("spectral_factory.lfcc.") for name in export.feature_order]
    if short != ref["feature_names"]:
        raise SystemExit("el orden de features no coincide con el export de referencia")
    return {
        "max_abs_diff_mean": float(np.max(np.abs(export.mean - np.array(ref["scaler_mean"])))),
        "max_rel_diff_scale": float(np.max(np.abs(export.scale / np.array(ref["scaler_scale"]) - 1))),
        "max_abs_diff_coef": float(np.max(np.abs(export.coef - np.array(ref["coef"])))),
        "abs_diff_intercept": abs(export.intercept - ref["intercept"]),
        "abs_diff_platt_a": abs(calibrator.a_ - ref["platt"]["A"]),
        "abs_diff_platt_b": abs(calibrator.b_ - ref["platt"]["B"]),
        "abs_diff_threshold": abs(threshold - ref["threshold"]),
    }


def record_val_look(out_name: str) -> None:
    """D-A7.4 mira `val` para ajustar, que es lo más fuerte que se le puede hacer. Queda escrito."""
    with VAL_LOOKS.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"s13_D-A7.4_build_{out_name}",
            "joaquin (sesión 13)",
            ("preregistro D-A7.4: val ENTRA AL AJUSTE (353 llamadas). No es una mirada de "
             "evaluación: a partir de aquí no queda holdout honesto para este bundle"),
        ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "spectral_factory_lfcc_v1")
    parser.add_argument("--reference", type=Path, help="export JSON de D-A7.3 para comparar")
    parser.add_argument(
        "--use-val", action="store_true",
        help="D-A7.4: ajusta con train+val (353). Quema el holdout y deja fila en val_looks.csv",
    )
    args = parser.parse_args(argv)

    spec = load_spec(SPEC)
    feature_order = tuple(spec["feature_order"])
    commit = _commit()
    code_digest = effective_code_digest(ROOT)

    ds = Dataset()
    protocol = pooled5_v1(ds) if args.use_val else official_v1(ds)
    features, labels, run_id = feature_table(spec, ds, code_digest, commit)
    if args.use_val:
        record_val_look(args.out.name)
        # `feature_table` extrae un split por corrida y `ids_for` solo conoce train y val, así que
        # se corre dos veces y se unen. La de train sale de caché; solo se extraen las 71 de val.
        val_features, val_labels, val_run_id = feature_table(
            spec | {"split": "val", "allow_val_fit": True}, ds, code_digest, commit,
        )
        features |= val_features
        labels |= val_labels
        run_id = f"{run_id}+{val_run_id}"
    ids = tuple(sorted(features))
    if set(ids) != set(protocol.fit_ids):
        raise SystemExit(f"las features no cubren exactamente {protocol.name}")
    table = FeatureTable(features, labels)
    X, y = table.matrix(ids, feature_order), table.y(ids)

    # Métricas del arnés: cross-fitting anidado sobre los folds congelados, igual que A4.
    branch = Branch("lfcc", feature_order, lambda order: logreg(order, **LR_PARAMS))
    nested = nested_cv(
        ds, protocol,
        make_nested_fit_predict(table, [branch], calibrator_factory=PlattCalibrator, inner_folds=3),
        load_groups(),
    )
    oof_auc = auc(nested.y_true, nested.y_score)
    auc_lo, auc_hi = bootstrap_ci(nested.y_true, nested.y_score, nested.ids)

    export, calibrator, threshold, _oof = fit_d_a7_3(X, y, feature_order)

    manifest = bundle_mod.BundleManifest(
        detector_name="spectral_factory_lfcc_logreg@1",
        code_digest=code_digest,
        data_fingerprint=ds.fingerprint(),
        protocol_id=protocol.name,
        run_id=run_id,
        feature_order=list(feature_order),
        extractor_refs=[spec["extractor"]],
        segmenter_ref=spec["segmenter"],
        transform_refs=list(spec["transforms"]),
        calibrator_ref="platt@1",
        threshold=threshold,
        min_seconds_declared=20.0,
        metrics={
            "n_train": len(ids),
            "protocol": protocol.name,
            "unit": "call",
            "seed": SEED,
            "recipe": "D-A7.4 (train+val)" if args.use_val else "D-A7.3",
            "oof_auc": round(float(oof_auc), 4),
            "oof_auc_ci95": [round(float(auc_lo), 4), round(float(auc_hi), 4)],
            "oof_brier": round(float(brier(nested.y_true, nested.y_score)), 4),
            "val_contaminated": protocol.val_contaminated,
            "val_d_a7_3": {
                "n": 71, "balanced_accuracy": 0.973, "tpr_synthetic": 1.0, "tnr_human": 0.946,
                "auc": 1.0, "brier": 0.027,
            },
            "val_used_for_selection": True,
            "val_looks_at_selection": 7,
        },
        limitations=[
            (
                "OBS: val (0.973) se uso para ELEGIR entre tres candidatos y ya se habia mirado "
                "cuatro veces (D-A7.3). La cifra es optimista; la que cuenta es la del set oculto."
            ),
            (
                "OBS: el OOF por llamada sobre train satura (AUC ~1.0) y no anticipa el costo de "
                "generalizar a hablantes nuevos."
            ),
            (
                "OBS: en val falla dos humanas con p(sintetica) de 0.95 a 0.997: errores "
                "confiados, no de umbral."
            ),
            (
                "OBS: A3.6 sobre LFCC sale POSITIVO en los dos controles (D-A7.6). El silencio del "
                "caller solo separa con AUC OOF 0.9994 [0.998, 1.000] y el canal del agente, mismo "
                "TTS en ambas clases, con 0.7397 [0.683, 0.801]; los dos mas fuertes que en el "
                "baseline acustico. Gran parte de la separacion es cadena de grabacion, no voz."
            ),
            (
                "UNK: si el set oculto conserva la cadena de grabacion por clase de train/val, C2 "
                "deberia sostenerse; si la cadena cambia o se iguala entre clases, puede caer. No "
                "hay evidencia de que transfiera a una cadena normalizada."
            ),
            (
                "UNK: la curva de truncacion no se midio para LFCC; min_seconds_declared hereda "
                "los 20 s del baseline acustico (D-A3.8)."
            ),
            *(
                [
                    ("🔴 OBS: este bundle se ajusto con train+val (353 llamadas, D-A7.4). NO TIENE "
                     "EVALUACION HONESTA: no queda ningun holdout con el que comparar contra el "
                     "bundle train-only. Sus metricas OOF se calculan sobre datos que el modelo "
                     "vio en el ajuste final y NO son evidencia de superioridad. La razon para "
                     "servirlo es a priori: 71 llamadas mas, con hablantes que no estaban en "
                     "train, y el set oculto se puntua con hablantes nuevos."),
                    ("UNK: si este bundle es mejor o peor que spectral_factory_lfcc_v1 "
                     "(train-only). Solo el resultado del set oculto lo dira, y para entonces ya "
                     "no se puede cambiar. El bundle train-only queda como rollback."),
                ]
                if args.use_val else []
            ),
        ],
    )

    path = bundle_mod.build_bundle(
        args.out, export=export, manifest=manifest, calibrator=calibrator,
        requirements=_requirements_lock(),
    )
    problems = bundle_mod.verify(args.out)
    if problems:
        raise SystemExit(f"el bundle recien escrito no verifica: {problems}")

    report = {
        "bundle": args.out.name,
        "manifest_sha256": bundle_mod.sha256_file(path),
        "detector": manifest.detector_name,
        "n_features": len(feature_order),
        "threshold": round(threshold, 6),
        "platt": {"a": calibrator.a_, "b": calibrator.b_},
        "oof_auc_nested": round(float(oof_auc), 4),
        "oof_auc_ci95": [round(float(auc_lo), 4), round(float(auc_hi), 4)],
    }
    if args.reference:
        report["vs_reference"] = compare_reference(args.reference, export, calibrator, threshold)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
