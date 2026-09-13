import json

import numpy as np
import pytest

from altur.registry import extractors, transforms, turn_sources
from altur.runner import RunnerError, run_experiment, validate_experiment_spec
from altur.types import AudioExample, DatasetRecord, Segmentation

_SPY_CALLS = []


class FakeDataset:
    ids = ("private-a", "private-b")

    def ids_for(self, split):
        assert split == "train"
        return self.ids

    def load(self, example_id, *, with_turns=False):
        assert example_id in self.ids
        assert not with_turns
        return AudioExample(np.ones(80), np.ones(80))

    def record(self, example_id):
        return DatasetRecord(example_id, label=1, split="train", groups={"secret": example_id})

    def audio_sha256(self, example_id):
        return ("a" if example_id.endswith("a") else "b") * 64

    def fingerprint(self):
        return "dataset-fingerprint"


def _register_spies():
    global _SPY_CALLS
    _SPY_CALLS = []
    if "test.seg@1" not in turn_sources:
        @turn_sources.register("test.seg", version=1, is_oracle=False)
        def segment(ex):
            assert type(ex) is AudioExample
            return Segmentation((), "test.seg@1", {})

    if "test.spy@1" not in extractors:
        @extractors.register(
            "test.spy", version=1, channels=(0,), needs_seg=True,
            license="BSD-3-Clause", product_safe=True,
        )
        def extract(ex):
            assert type(ex) is AudioExample
            assert not any(hasattr(ex, field) for field in ("example_id", "label", "split", "groups", "provenance"))
            _SPY_CALLS.append(ex)
            return {"test.spy.value": 2.0}, {"ok": True}
    return _SPY_CALLS


def spec(**overrides):
    value = {
        "schema_version": 1,
        "extractor": "test.spy@1",
        "segmenter": "test.seg@1",
        "feature_order": ["test.spy.value"],
        "dtype": "float64",
        "nan_policy": "reject",
        "split": "train",
        "random_seed": 7,
        "transforms": [],
    }
    value.update(overrides)
    return value


def test_runner_is_leak_free_cached_deterministic_and_summary_has_no_ids(tmp_path):
    calls = _register_spies()
    kwargs = {
        "dataset": FakeDataset(),
        "cache_root": tmp_path / "cache",
        "artifact_root": tmp_path / "artifacts",
        "ledger_root": tmp_path / "ledger",
        "code_digest": "sha256:v1:code",
        "commit": "abc123",
        "environment": {"python": "test"},
        "code_version": "0.1.0",
    }
    first = run_experiment(spec(), **kwargs)
    second = run_experiment(spec(), **kwargs)

    assert len(calls) == 2
    assert first.run_id == second.run_id
    assert first.summary == second.summary
    assert "private-a" not in json.dumps(first.summary)
    assert "private-b" not in json.dumps(first.summary)
    assert first.ledger_path.read_bytes() == second.ledger_path.read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"extractor": "test.spy"},
        {"segmenter": "test.seg"},
        {"feature_order": ["wrong.order"]},
    ],
)
def test_runner_rejects_floating_refs_and_schema_mismatch(tmp_path, change):
    _register_spies()
    with pytest.raises(RunnerError):
        run_experiment(
            spec(**change), dataset=FakeDataset(), cache_root=tmp_path / "cache",
            artifact_root=tmp_path / "artifacts", ledger_root=tmp_path / "ledger",
            code_digest="sha256:v1:code", commit="abc123", environment={},
            code_version="0.1.0",
        )


@pytest.mark.parametrize("split", ["val", "all", "", None])
def test_runner_rejects_non_train_split_before_dataset_load(tmp_path, split):
    _register_spies()

    class NoLoadDataset(FakeDataset):
        def load(self, example_id, *, with_turns=False):
            raise AssertionError("dataset.load must not be called")

    with pytest.raises(RunnerError, match="split must be exactly 'train'"):
        run_experiment(
            spec(split=split), dataset=NoLoadDataset(), cache_root=tmp_path / "cache",
            artifact_root=tmp_path / "artifacts", ledger_root=tmp_path / "ledger",
            code_digest="sha256:v1:code", commit="abc123", environment={}, code_version="0.1.0",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"split": "val"},                              # sin la bandera, `val` sigue cerrado
        {"split": "val", "allow_val_fit": False},      # la bandera en falso no abre nada
        {"split": "all", "allow_val_fit": True},       # la bandera no habilita otros splits
    ],
)
def test_allow_val_fit_only_opens_val_and_only_when_true(overrides):
    """La puerta de D-A7.4 es estrecha a propósito: `val`, y solo con `allow_val_fit: true`."""
    with pytest.raises(RunnerError, match="split must be exactly 'train'"):
        validate_experiment_spec(spec(**overrides))


def test_allow_val_fit_true_lets_val_through():
    """Con la bandera explícita, `val` pasa. Es lo que usa scripts/build_bundle_lfcc.py --use-val."""
    validate_experiment_spec(spec(split="val", allow_val_fit=True))


def _run_kwargs(tmp_path):
    return {
        "dataset": FakeDataset(),
        "cache_root": tmp_path / "cache",
        "artifact_root": tmp_path / "artifacts",
        "ledger_root": tmp_path / "ledger",
        "code_digest": "sha256:v1:code",
        "commit": "abc123",
        "environment": {},
        "code_version": "0.1.0",
    }


def test_runner_applies_ordered_holdout_transforms_before_segmentation(tmp_path):
    calls = _register_spies()
    observed = []
    if "test.add_one@1" not in transforms:
        @transforms.register(
            "test.add_one", version=1, use="holdout", changes_length=False,
            shifts_timing=False, params={"amount": 1.0},
        )
        def add_one(ex, rng):
            observed.append("add")
            return AudioExample(ex.ch0 + 1, ex.ch1, sr=ex.sr, seg=ex.seg)
    else:
        transforms.resolve("test.add_one@1").__globals__["observed"] = observed

    if "test.double@1" not in transforms:
        @transforms.register(
            "test.double", version=1, use="holdout", changes_length=False,
            shifts_timing=False, params={"factor": 2.0},
        )
        def double(ex, rng):
            observed.append("double")
            return AudioExample(ex.ch0 * 2, ex.ch1, sr=ex.sr, seg=ex.seg)
    else:
        transforms.resolve("test.double@1").__globals__["observed"] = observed

    if "test.transform_seg@1" not in turn_sources:
        @turn_sources.register("test.transform_seg", version=1, is_oracle=False)
        def transform_seg(ex):
            observed.append("segment")
            np.testing.assert_array_equal(ex.ch0, np.full(80, 4.0))
            return Segmentation((), "test.transform_seg@1", {})
    else:
        turn_sources.resolve("test.transform_seg@1").__globals__["observed"] = observed

    run_experiment(
        spec(
            segmenter="test.transform_seg@1",
            transforms=["test.add_one@1", "test.double@1"],
        ),
        **_run_kwargs(tmp_path),
    )

    assert observed == ["add", "double", "segment"] * 2
    assert all(call.seg is not None for call in calls)


@pytest.mark.parametrize("refs", [["lowpass_3400"], ["test.augment@1"]])
def test_runner_rejects_floating_or_non_holdout_transforms_before_dataset_load(tmp_path, refs):
    _register_spies()
    if "test.augment@1" not in transforms:
        @transforms.register(
            "test.augment", version=1, use="augment", changes_length=False,
            shifts_timing=False, params={},
        )
        def augment(ex, rng):
            return ex

    class NoLoadDataset(FakeDataset):
        def load(self, example_id, *, with_turns=False):
            raise AssertionError("dataset.load must not be called")

    kwargs = _run_kwargs(tmp_path)
    kwargs["dataset"] = NoLoadDataset()
    with pytest.raises(RunnerError, match="explicit|holdout"):
        run_experiment(spec(transforms=refs), **kwargs)


def test_transform_registry_parameters_invalidate_runner_cache(tmp_path):
    calls = _register_spies()
    if "test.cache_transform@1" not in transforms:
        @transforms.register(
            "test.cache_transform", version=1, use="holdout", changes_length=False,
            shifts_timing=False, params={"amount": 1.0},
        )
        def cache_transform(ex, rng):
            return ex

    entry = transforms.get("test.cache_transform@1")
    original = dict(entry.meta["params"])
    try:
        selected = spec(transforms=["test.cache_transform@1"])
        run_experiment(selected, **_run_kwargs(tmp_path))
        run_experiment(selected, **_run_kwargs(tmp_path))
        assert len(calls) == 2
        entry.meta["params"]["amount"] = 2.0
        run_experiment(selected, **_run_kwargs(tmp_path))
        assert len(calls) == 4
    finally:
        entry.meta["params"].clear()
        entry.meta["params"].update(original)


def test_transform_rng_is_per_example_deterministic_and_metadata_free(tmp_path):
    _register_spies()
    observed: list[float] = []
    if "test.randomized@1" not in transforms:
        @transforms.register(
            "test.randomized", version=1, use="holdout", changes_length=False,
            shifts_timing=False, params={"draws": 1},
        )
        def randomized(ex, rng):
            assert type(ex) is AudioExample
            assert not any(
                hasattr(ex, field)
                for field in ("example_id", "label", "split", "groups", "provenance")
            )
            draw = float(rng.random())
            observed.append(draw)
            return AudioExample(ex.ch0 + draw, ex.ch1, sr=ex.sr, seg=ex.seg)
    else:
        transforms.resolve("test.randomized@1").__globals__["observed"] = observed

    class OneExample(FakeDataset):
        ids = ("private-a",)

    class OtherExample(FakeDataset):
        ids = ("private-b",)

    kwargs = _run_kwargs(tmp_path / "first")
    kwargs["dataset"] = OneExample()
    selected = spec(transforms=["test.randomized@1"])
    run_experiment(selected, **kwargs)
    run_experiment(selected, **{**kwargs, "cache_root": tmp_path / "second" / "cache"})
    run_experiment(selected, **{**kwargs, "dataset": OtherExample(), "cache_root": tmp_path / "third" / "cache"})

    assert observed[0] == observed[1]
    assert observed[0] != observed[2]
