"""Entrena un candidato product-safe y lo congela en un bundle verificable.

Esto es A4.1. El candidato NO aspira a ganar por AUC — el modelo lo construye la otra
mitad del equipo (`docs/MODEL_HANDOFF.md`). Este bundle es **la prueba de que el arnés
funciona de punta a punta**: entrenar con sklearn, exportar a NumPy puro, sellar con
hashes, cargar sin sklearn y predecir dentro de tolerancia.

El flujo copia deliberadamente el de `crossfit.make_nested_fit_predict`, con una sola
diferencia: aquí `IN_k` es todo `train`. Calibrador y umbral se ajustan sobre scores OOF
**internos**, nunca sobre las mismas filas con las que se ajustó el modelo final.

    python scripts/build_bundle.py --out models/acoustic_ch0_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from altur import bundle as bundle_mod
from altur.ablation import auc, bootstrap_ci, brier
from altur.crossfit import FeatureTable, balanced_threshold
from altur.dataset import Dataset
from altur.models.adapters import logreg
from altur.models.calibration import PlattCalibrator
from altur.protocol import (
    grouped_inner_splits,
    load_groups,
    nested_cv,
    official_v1,
)
from altur.provenance import effective_code_digest
from altur.runner import run_experiment
from run_experiment import load_spec

INNER_FOLDS = 5
SPEC = ROOT / "configs" / "experiments" / "a2_acoustic_ch0_v1.yaml"

# Las cinco dependencias del camino de inferencia (`pyproject.toml`). El lock viaja DENTRO
# del bundle para que el contenedor se pueda reconstruir sin adivinar versiones.
INFERENCE_DEPS = ("numpy", "fastapi", "uvicorn", "pydantic", "python-multipart")


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("hace falta un commit de git para una corrida reproducible") from None


def _requirements_lock() -> str:
    """Versiones instaladas de las dependencias de inferencia, y nada más."""
    import importlib.metadata as md

    lines = [
        "# Dependencias del camino de INFERENCIA, congeladas por scripts/build_bundle.py.",
        "# sklearn NO está aquí a propósito: el bundle es NumPy puro (D-A3.3).",
        f"# python {platform.python_version()}",
    ]
    for name in INFERENCE_DEPS:
        try:
            lines.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            lines.append(f"# {name}: no instalado en el entorno de build")
    return "\n".join(lines) + "\n"


def feature_table(spec: dict[str, Any], ds: Dataset, code_digest: str, commit: str):
    """Corre el extractor (con caché) y devuelve features + etiquetas por `example_id`."""
    private = ROOT / ".private"
    result = run_experiment(
        spec,
        dataset=ds,
        cache_root=private / "cache",
        artifact_root=private / "artifacts",
        ledger_root=private / "experiments" / "runs",
        code_digest=code_digest,
        commit=commit,
        environment={
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "package_version": "0.1.0",
        },
        code_version="0.1.0",
    )
    art_dir = private / "artifacts" / result.run_id
    features: dict[str, dict[str, float]] = {}
    labels: dict[str, int] = {}
    for example_id in ds.ids_for(spec["split"]):
        path = art_dir / f"{hashlib.sha256(example_id.encode()).hexdigest()}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        features[example_id] = payload["features"]
        labels[example_id] = int(payload["label"])
    return features, labels, result.run_id


def fit_final(table: FeatureTable, ids: tuple[str, ...], feature_order: tuple[str, ...],
              groups: dict[str, str]):
    """Modelo final + calibrador + umbral, con el mismo candado que el cross-fitting.

    El calibrador y el umbral se ajustan sobre scores OOF **internos**. Ajustarlos sobre
    las predicciones del modelo ya entrenado en esas mismas filas daría un calibrador
    optimista: el modelo las ha visto, así que sus scores ahí no se parecen a los de
    producción.
    """
    X = table.matrix(ids, feature_order)
    y = table.y(ids)
    g = np.array([groups[i] for i in ids])

    inner = grouped_inner_splits(y, g, n_folds=INNER_FOLDS)
    oof = np.full(len(ids), np.nan, dtype=np.float64)
    for tr, te in inner:
        m = logreg(feature_order)
        m.fit(X[tr], y[tr])
        oof[te] = m.p_synthetic(X[te])
    usable = np.isfinite(oof)
    if usable.sum() < 4:
        raise SystemExit("los splits internos no cubren train")

    calibrator = PlattCalibrator().fit(oof[usable], y[usable])
    threshold = balanced_threshold(calibrator.transform(oof[usable]), y[usable])

    final = logreg(feature_order)
    final.fit(X, y)
    # verify_on con TODAS las filas: el export se compara contra el estimador real y falla
    # si max|delta| > 1e-9. Sin esto, la convención de signo de sklearn se asume en
    # silencio y un bundle invertido pasa todos los hashes (D-A3.3).
    export = final.export_linear(verify_on=X)
    return export, calibrator, float(threshold), oof, usable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "acoustic_ch0_v1")
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    feature_order = tuple(spec["feature_order"])
    commit = _commit()
    code_digest = effective_code_digest(ROOT)

    ds = Dataset()
    groups = load_groups()
    features, labels, run_id = feature_table(spec, ds, code_digest, commit)
    table = FeatureTable(features, labels)
    protocol = official_v1(ds)
    train_ids = protocol.fit_ids

    # --- 1. Métricas honestas: cross-fitting anidado sobre los folds congelados. --------
    from altur.crossfit import Branch, make_nested_fit_predict

    branch = Branch("acoustic", feature_order, lambda order: logreg(order))
    nested = nested_cv(
        ds,
        protocol,
        make_nested_fit_predict(
            table, [branch], calibrator_factory=PlattCalibrator, inner_folds=3
        ),
        groups,
    )
    oof_auc = auc(nested.y_true, nested.y_score)
    oof_brier = brier(nested.y_true, nested.y_score)
    auc_lo, auc_hi = bootstrap_ci(nested.y_true, nested.y_score, nested.ids)

    # --- 2. El artefacto que se congela. ----------------------------------------------
    export, calibrator, threshold, _oof, _usable = fit_final(
        table, train_ids, feature_order, groups
    )

    manifest = bundle_mod.BundleManifest(
        detector_name=f"acoustic_ch0_logreg@{spec['schema_version']}",
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
            "n_train": len(train_ids),
            "protocol": protocol.name,
            "unit": "call",
            "seed": spec["random_seed"],
            "oof_auc": round(float(oof_auc), 4),
            "oof_auc_ci95": [round(float(auc_lo), 4), round(float(auc_hi), 4)],
            "oof_brier": round(float(oof_brier), 4),
            "val_contaminated": protocol.val_contaminated,
            "val_looks": 0,
        },
        limitations=[
            (
                "OBS: el control de confound sale POSITIVO en los dos ejes (D-A3.6). El canal "
                "del agente, mismo TTS en ambas clases, separa con AUC 0.6409 [0.573, 0.707]; "
                "el silencio solo da 0.9662 con AUC de escala 0.0157. Parte grande de este "
                "numero mide la cadena de produccion, no la voz."
            ),
            (
                "OBS: a 5 s de audio el AUC OOF es 0.4012 con IC95 [0.328, 0.460] -- invertido, "
                "no indeciso -- y la confianza media es la mas alta de la curva. Por eso la "
                "parada temprana esta PROHIBIDA (D-A3.8) y min_seconds_declared es 20 s."
            ),
            (
                "OBS: el eje conductual no aporta discriminacion (+0.0015, dentro del IC95, "
                "D-A3.5). Este bundle es solo acustico a proposito."
            ),
            (
                "UNK: la unidad independiente disponible es la llamada. La dependencia por "
                "hablante o por voz es desconocida (D-A1.2), asi que el IC95 no la cubre."
            ),
            (
                "UNK: no hay evidencia de que este numero transfiera a un set con la cadena de "
                "grabacion normalizada. El colapso cross-dataset es el modo de falla "
                "documentado del campo."
            ),
        ],
    )

    path = bundle_mod.build_bundle(
        args.out,
        export=export,
        manifest=manifest,
        calibrator=calibrator,
        requirements=_requirements_lock(),
    )

    problems = bundle_mod.verify(args.out)
    if problems:
        raise SystemExit(f"el bundle recien escrito no verifica: {problems}")

    print(json.dumps({
        "bundle": args.out.name,
        "manifest_sha256": bundle_mod.sha256_file(path),
        "detector": manifest.detector_name,
        "n_features": len(feature_order),
        "threshold": round(threshold, 6),
        "oof_auc": round(float(oof_auc), 4),
        "oof_auc_ci95": [round(float(auc_lo), 4), round(float(auc_hi), 4)],
        "oof_brier": round(float(oof_brier), 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
