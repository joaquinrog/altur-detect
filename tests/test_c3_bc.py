from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_c3_bc", ROOT / "scripts" / "evaluate_c3_bc.py")
c3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(c3)


def test_c3_feature_subset_has_exactly_100_and_no_static_mean():
    assert len(c3.C3_FEATURE_ORDER) == 100
    assert not any(name.endswith("_mean") and ".static" in name for name in c3.C3_FEATURE_ORDER)
    assert c3.C3_FEATURE_ORDER == tuple(
        name for name in c3.FULL_FEATURE_ORDER if not ".static" in name or not name.endswith("_mean")
    )


def test_flat_selection_counts_and_groups():
    rows = [
        {"id": "b1", "variant": "plana", "label": 1, "group_id": "voice-1", "source": "bloque_b"},
        {"id": "g1", "variant": "g711", "label": 1, "group_id": "voice-1", "source": "bloque_b"},
        {"id": "b2", "variant": "plana", "label": 0, "group_id": "person-1", "source": "bloque_b"},
    ]
    selected = c3.select_flat_rows(rows)
    assert [row["id"] for row in selected] == ["b1", "b2"]
    with pytest.raises(c3.C3EvaluationError, match="conteo"):
        c3.validate_corpus_rows(selected, expected_counts={"total": 3})


def test_diagnostic_is_distinct_from_confirmatory_and_does_not_claim_b3():
    diagnostic = c3.phase_for_rows([{"source": "bloque_b", "variant": "plana"}])
    assert diagnostic == "diagnostic_b_b2_burned"
    with pytest.raises(c3.C3EvaluationError, match="B3"):
        c3.phase_for_rows([{"source": "b3", "variant": "plana"}])


def test_confirmatory_requires_explicit_manifest_and_paths(tmp_path):
    with pytest.raises(c3.C3EvaluationError, match="manifest.*paths|paths.*manifest"):
        c3.require_confirmatory_inputs(None, None)
    with pytest.raises(c3.C3EvaluationError, match="models/current"):
        c3.reject_production_path(tmp_path / "models" / "current")


def test_paired_metrics_and_exact_mcnemar():
    rows = [
        {"wav_id": "a", "label": 1, "condition": "g711", "c2_score": .2, "c3_score": .8},
        {"wav_id": "b", "label": 1, "condition": "g711", "c2_score": .8, "c3_score": .9},
        {"wav_id": "c", "label": 0, "condition": "g711", "c2_score": .2, "c3_score": .1},
        {"wav_id": "d", "label": 0, "condition": "g711", "c2_score": .8, "c3_score": .2},
    ]
    result = c3.evaluate_paired(rows, c2_threshold=.5, c3_threshold=.5)
    assert result["conditions"]["g711"]["n"] == 4
    assert result["conditions"]["g711"]["difference"]["tpr"]["value"] == pytest.approx(.5)
    assert result["mcnemar"]["discordant_total"] == 2
    assert 0 <= result["conditions"]["g711"]["c3"]["brier"] <= 1


def test_each_model_uses_its_frozen_threshold_and_reports_latency_p95():
    rows = [
        {
            "wav_id": f"v{i}", "label": 1, "condition": "g711",
            "c2_score": score, "c3_score": score,
            "c2_latency_s": i + 1, "c3_latency_s": i + 2,
        }
        for i, score in enumerate((.2, .4, .6, .8))
    ]

    result = c3.evaluate_paired(rows, c2_threshold=.7, c3_threshold=.3)
    condition = result["conditions"]["g711"]

    assert condition["c2"]["tpr"]["value"] == pytest.approx(.25)
    assert condition["c3"]["tpr"]["value"] == pytest.approx(.75)
    assert condition["mcnemar"]["discordant_total"] == 2
    assert condition["latency_s"]["c3_latency_s"]["p95"] > 4


def test_paired_interval_is_for_mean_difference_not_individual_outcomes():
    y = [1] * 100
    c2 = [False] * 40 + [True] * 60
    c3_pred = [True] * 60 + [False] * 40

    interval = c3._paired_bootstrap(y, c2, c3_pred, positive=True)

    assert interval[0] > -1
    assert interval[1] < 1


def test_no_production_write_path():
    with pytest.raises(c3.C3EvaluationError, match="models/current"):
        c3.ensure_report_path(ROOT / "models" / "current" / "report.json")


def test_diagnostic_scoring_uses_same_wav_for_both_models(tmp_path, monkeypatch):
    wav = tmp_path / "one.wav"
    wav.write_bytes(b"same-audio")
    monkeypatch.setattr(
        c3,
        "decode",
        lambda raw, **_: SimpleNamespace(example=raw),
        raising=False,
    )

    class FakeDetector:
        def __init__(self, score):
            self.score = score

        def predict(self, example):
            assert example == b"same-audio"
            return SimpleNamespace(diagnostics={"p_synthetic": self.score})

    rows = c3.score_records(
        [{"wav_id": "w", "label": 1, "condition": "B/plana", "audio_path": wav}],
        FakeDetector(.2),
        FakeDetector(.8),
    )

    assert rows[0]["c2_score"] == .2
    assert rows[0]["c3_score"] == .8
    assert rows[0]["c2_latency_s"] >= 0
    assert "audio_path" not in rows[0]


def test_serving_projects_extractor_superset_to_bundle_feature_order():
    from altur.models.base import LinearExport
    from altur.registry import extractors
    from altur.serving import BundledDetector
    from altur.types import AudioExample

    ref = "test.c3.superset@1"
    if ref not in extractors:
        @extractors.register(
            "test.c3.superset", version=1, channels=(0,), needs_seg=False,
            license="MIT", product_safe=True,
        )
        def extract(_example):
            return {"test.c3.superset.used": 2.0, "test.c3.superset.extra": 99.0}, {}

    detector = BundledDetector(
        name="test-c3",
        export=LinearExport(
            feature_order=("test.c3.superset.used",),
            mean=np.zeros(1), scale=np.ones(1), coef=np.ones(1), intercept=0.0,
            name="test-c3",
        ),
        extractor_refs=[ref],
        segmenter_ref=None,
    )

    prediction = detector.predict(AudioExample(np.ones(8000), None))

    assert prediction.diagnostics["p_synthetic"] > .5
