import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "report_a2_cp0.py"


def _module():
    spec = importlib.util.spec_from_file_location("report_a2_cp0", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _private_run(root, run_id, rows, *, split="train", n=None):
    ledger = {
        "code_digest": "sha256:v1:test",
        "spec": {
            "extractor": "a2.acoustic.ch0.speech@1",
            "segmenter": "energy@1",
            "transforms": [],
            "split": split,
        },
        "results": {"n": len(rows) if n is None else n, "split": split},
    }
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / f"{run_id}.json").write_text(json.dumps(ledger))
    artifact_dir = root / "artifacts" / run_id
    artifact_dir.mkdir(parents=True)
    for index, row in enumerate(rows):
        (artifact_dir / f"row-{index}.json").write_text(json.dumps(row))


def _rows(prefix="fixture"):
    return [
        {"example_id": f"{prefix}-a", "label": 0, "features": {"f": 0.1}},
        {"example_id": f"{prefix}-b", "label": 0, "features": {"f": 0.2}},
        {"example_id": f"{prefix}-c", "label": 1, "features": {"f": 0.8}},
        {"example_id": f"{prefix}-d", "label": 1, "features": {"f": 0.9}},
    ]


@pytest.mark.parametrize(
    ("rows", "kwargs", "message"),
    [
        (_rows("other"), {}, "identical example_id set"),
        (_rows()[:-1] + [_rows()[0]], {}, "duplicate example_id"),
        (_rows(), {"split": "val"}, "split must be train"),
        (_rows(), {"n": 3}, "expected n"),
    ],
)
def test_report_rejects_invalid_private_run_sets(tmp_path, rows, kwargs, message):
    module = _module()
    _private_run(tmp_path, "run-a", _rows())
    _private_run(tmp_path, "run-b", rows, **kwargs)

    with pytest.raises(module.ReportError, match=message):
        module.load_runs(
            ("run-a", "run-b"),
            ledger_root=tmp_path / "runs",
            artifact_root=tmp_path / "artifacts",
            expected_n=4,
        )


def test_report_output_is_aggregated_without_identifiers(tmp_path):
    module = _module()
    _private_run(tmp_path, "run-a", _rows())
    _private_run(tmp_path, "run-b", _rows())

    runs = module.load_runs(
        ("run-a", "run-b"),
        ledger_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
        expected_n=4,
    )
    report = module.describe_runs(runs)
    encoded = json.dumps(report)

    assert "fixture-a" not in encoded
    assert "example_id" not in encoded
    assert report["family"] == "all run-feature location and scale tests"


def test_oof_fits_only_each_train_fold(tmp_path, monkeypatch):
    module = _module()
    rows = _rows()
    _private_run(tmp_path, "run-a", rows)
    run = module.load_runs(
        ("run-a",),
        ledger_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
        expected_n=4,
    )[0]
    folds = [
        {"train": ["fixture-a", "fixture-c"], "test": ["fixture-b", "fixture-d"]},
        {"train": ["fixture-b", "fixture-d"], "test": ["fixture-a", "fixture-c"]},
    ]
    fit_sizes = []
    original_fit = module.LogisticRegression.fit

    def spy_fit(self, x, y, *args, **kwargs):
        fit_sizes.append(len(y))
        return original_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(module.LogisticRegression, "fit", spy_fit)
    result = module.oof_auc(run, folds)

    assert fit_sizes == [2, 2]
    assert result["n_oof"] == 4
