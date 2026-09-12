"""Aggregate explicitly selected private A2 runs into a public CP0 report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import levene, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class ReportError(ValueError):
    """A selected private artifact set cannot support the declared CP0 analysis."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid private JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"private JSON must be an object: {path.name}")
    return value


def _safe_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ReportError("run IDs must be single path components")
    return run_id


def load_runs(
    run_ids: tuple[str, ...], *, ledger_root: Path, artifact_root: Path, expected_n: int = 282
) -> list[dict[str, Any]]:
    """Load only the requested private ledgers and artifact directories."""
    if not run_ids:
        raise ReportError("at least one explicit run ID is required")
    if len(set(run_ids)) != len(run_ids):
        raise ReportError("duplicate run IDs are not allowed")
    runs: list[dict[str, Any]] = []
    reference_ids: set[str] | None = None
    reference_digest: str | None = None
    for run_id in run_ids:
        run_id = _safe_run_id(run_id)
        ledger = _read_json(ledger_root / f"{run_id}.json")
        spec = ledger.get("spec")
        results = ledger.get("results")
        if not isinstance(spec, dict) or not isinstance(results, dict):
            raise ReportError("private ledger requires spec and results")
        if spec.get("split") != "train" or results.get("split") != "train":
            raise ReportError("split must be train in both private ledger fields")
        if results.get("n") != expected_n:
            raise ReportError(f"run has n={results.get('n')!r}; expected n={expected_n}")
        code_digest = ledger.get("code_digest")
        if not isinstance(code_digest, str) or not code_digest:
            raise ReportError("private ledger requires a code_digest")
        if reference_digest is None:
            reference_digest = code_digest
        elif code_digest != reference_digest:
            raise ReportError("selected runs must use an identical code_digest")
        artifact_dir = artifact_root / run_id
        paths = sorted(artifact_dir.glob("*.json"))
        if len(paths) != expected_n:
            raise ReportError(f"artifact count {len(paths)} does not match expected n={expected_n}")
        rows: dict[str, dict[str, Any]] = {}
        for path in paths:
            row = _read_json(path)
            example_id = row.get("example_id")
            label = row.get("label")
            features = row.get("features")
            if not isinstance(example_id, str) or not example_id:
                raise ReportError("artifact is missing example_id")
            if example_id in rows:
                raise ReportError("duplicate example_id in private artifacts")
            if label not in (0, 1):
                raise ReportError("labels must be binary 0 or 1")
            if not isinstance(features, dict) or not features:
                raise ReportError("artifact features must be a non-empty object")
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in features.values()):
                raise ReportError("artifact features must be finite numeric values")
            rows[example_id] = {"label": label, "features": {k: float(v) for k, v in features.items()}}
        ids = set(rows)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ReportError("selected runs must have an identical example_id set")
        runs.append({"run_id": run_id, "spec": spec, "rows": rows, "code_digest": code_digest})
    return runs


def _condition(run: dict[str, Any]) -> tuple[str, str, str]:
    spec = run["spec"]
    transforms = tuple(spec.get("transforms", ()))
    if len(transforms) > 1:
        raise ReportError("CP0 accepts at most one holdout transform per run")
    condition = "clean" if not transforms else transforms[0].split("@", 1)[0]
    extractor = spec.get("extractor")
    if not isinstance(extractor, str):
        raise ReportError("extractor must be a string")
    channel = "ch1" if ".ch1" in extractor else "ch0"
    region = "speech" if extractor.endswith(".speech@1") else "silence" if extractor.endswith(".silence@1") else "all"
    return condition, region, channel


def _summary(values: np.ndarray) -> dict[str, float]:
    q1, q3 = np.percentile(values, [25, 75])
    return {"mean": float(values.mean()), "median": float(np.median(values)), "iqr": float(q3 - q1)}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _bh(items: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(items), key=lambda pair: pair[1]["p"])
    previous = 1.0
    total = len(ordered)
    for rank, (index, item) in reversed(list(enumerate(ordered, start=1))):
        adjusted = min(previous, item["p"] * total / rank)
        items[index]["p_bh"] = float(adjusted)
        previous = adjusted


def describe_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    descriptions: list[dict[str, Any]] = []
    for run in runs:
        condition, region, channel = _condition(run)
        rows = run["rows"]
        feature_names = tuple(next(iter(rows.values()))["features"])
        if any(tuple(row["features"]) != feature_names for row in rows.values()):
            raise ReportError("feature order differs within a run")
        result = {"condition": condition, "region": region, "channel": channel, "features": []}
        labels = np.array([row["label"] for row in rows.values()])
        for name in feature_names:
            values = np.array([row["features"][name] for row in rows.values()])
            human, synthetic = values[labels == 0], values[labels == 1]
            location = mannwhitneyu(human, synthetic, alternative="two-sided")
            scale = levene(human, synthetic, center="median")
            distances = np.abs(values - np.median(values))
            auc_distance = float(roc_auc_score(labels, distances))
            entry = {
                "feature": name,
                "n_by_class": {"human": len(human), "synthetic": len(synthetic)},
                "human": _summary(human),
                "synthetic": _summary(synthetic),
                "mann_whitney_u": float(location.statistic),
                "mann_whitney_p": float(location.pvalue),
                "brown_forsythe_statistic": _finite(scale.statistic),
                "brown_forsythe_p": _finite(scale.pvalue),
                "auc_distance_to_pooled_median": auc_distance,
            }
            result["features"].append(entry)
            tests.append(
                {"p": entry["mann_whitney_p"], "entry": entry, "field": "mann_whitney_p_bh"}
            )
            if entry["brown_forsythe_p"] is not None:
                tests.append(
                    {
                        "p": entry["brown_forsythe_p"],
                        "entry": entry,
                        "field": "brown_forsythe_p_bh",
                    }
                )
        descriptions.append(result)
    adjusted = [{"p": item["p"]} for item in tests]
    _bh(adjusted)
    for source, correction in zip(tests, adjusted, strict=True):
        source["entry"][source["field"]] = correction["p_bh"]
    return {
        "schema_version": 1,
        "family": "all run-feature location and scale tests",
        "rms_policy": "RMS-normalized features exclude raw RMS; raw RMS remains diagnostic-only.",
        "runs": descriptions,
    }


def oof_auc(run: dict[str, Any], folds: list[dict[str, list[str]]]) -> dict[str, Any]:
    rows = run["rows"]
    names = tuple(next(iter(rows.values()))["features"])
    y_all: list[int] = []
    scores: list[float] = []
    seen_test: set[str] = set()
    for fold in folds:
        train_ids, test_ids = fold["train"], fold["test"]
        train_set, test_set = set(train_ids), set(test_ids)
        if (
            len(train_set) != len(train_ids)
            or len(test_set) != len(test_ids)
            or train_set & test_set
            or seen_test & test_set
            or train_set | test_set != set(rows)
        ):
            raise ReportError("official_v1 fold IDs do not match selected private artifacts")
        seen_test.update(test_set)
        x_train = np.array([[rows[i]["features"][name] for name in names] for i in train_ids])
        y_train = np.array([rows[i]["label"] for i in train_ids])
        x_test = np.array([[rows[i]["features"][name] for name in names] for i in test_ids])
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(scaler.fit_transform(x_train), y_train)
        scores.extend(model.predict_proba(scaler.transform(x_test))[:, 1])
        y_all.extend(rows[i]["label"] for i in test_ids)
    if seen_test != set(rows):
        raise ReportError("official_v1 test folds must cover each artifact exactly once")
    auc = float(roc_auc_score(y_all, scores)) if len(set(y_all)) == 2 else None
    return {"auc_oof_p_synthetic": auc, "n_oof": len(y_all), "protocol": "official_v1"}


def _official_folds(path: Path) -> list[dict[str, list[str]]]:
    data = _read_json(path)
    try:
        folds = data["protocols"]["official_v1"]["folds"]
    except KeyError as exc:
        raise ReportError("missing frozen official_v1 folds") from exc
    if not isinstance(folds, list):
        raise ReportError("frozen official_v1 folds must be a list")
    return [{"train": fold["train"], "test": fold["test"]} for fold in folds]


def build_report(runs: list[dict[str, Any]], *, folds_path: Path) -> dict[str, Any]:
    coordinates = [_condition(run) for run in runs]
    if len(set(coordinates)) != len(coordinates):
        raise ReportError("selected runs must have unique condition, region, and channel")
    report = describe_runs(runs)
    folds = _official_folds(folds_path)
    aucs = []
    for run in runs:
        condition, region, channel = _condition(run)
        aucs.append({"condition": condition, "region": region, "channel": channel, **oof_auc(run, folds)})
    report["oof"] = aucs
    report["auc_survival"] = [
        {
            **entry,
            "survival_vs_clean": None if clean["auc_oof_p_synthetic"] in (None, 0.5) or entry["auc_oof_p_synthetic"] is None else (entry["auc_oof_p_synthetic"] - 0.5) / (clean["auc_oof_p_synthetic"] - 0.5),
        }
        for entry in aucs
        for clean in aucs
        if clean["condition"] == "clean" and (clean["region"], clean["channel"]) == (entry["region"], entry["channel"])
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, dest="run_ids")
    parser.add_argument(
        "--ledger-root", type=Path, default=Path(".private/experiments/runs")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(".private/artifacts"))
    parser.add_argument("--folds", type=Path, default=Path("configs/protocol/folds_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/a2_cp0_v1.json"))
    args = parser.parse_args(argv)
    report = build_report(
        load_runs(
            tuple(args.run_ids),
            ledger_root=args.ledger_root,
            artifact_root=args.artifact_root,
        ),
        folds_path=args.folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
