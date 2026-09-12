import json
import os
import threading

import pytest

from altur.cache import FeatureCache, cache_key


def cache_inputs(**overrides):
    values = {
        "audio_bytes": b"effective wav bytes",
        "segmenter": {"id": "webrtc", "version": 1, "params": {"mode": 2}},
        "transforms": [
            {"id": "mulaw_roundtrip", "version": 1, "params": {"p": 0.5}},
            {"id": "rawboost_lite", "version": 2, "params": {"gain": 0.1}},
        ],
        "random_seed": 7,
        "extractor": {
            "id": "acoustic.spectral",
            "version": 1,
            "params": {},
            "schema": {"names": ["centroid", "flatness"]},
        },
        "feature_order": ["centroid", "flatness"],
        "nan_policy": "reject",
        "dtype": "float64",
        "code_digest": "sha256:v1:effective-code",
    }
    values.update(overrides)
    return values


def test_complete_key_changes_for_every_required_component():
    baseline = cache_key(**cache_inputs())
    variations = [
        {"audio_bytes": b"different effective bytes"},
        {"segmenter": {"id": "webrtc", "version": 2, "params": {"mode": 2}}},
        {"segmenter": {"id": "webrtc", "version": 1, "params": {"mode": 3}}},
        {"transforms": [{"id": "mulaw_roundtrip", "version": 2, "params": {"p": 0.5}}]},
        {"transforms": [{"id": "mulaw_roundtrip", "version": 1, "params": {"p": 0.7}}]},
        {"transforms": list(reversed(cache_inputs()["transforms"]))},
        {"random_seed": 8},
        {"random_seed": None, "random_bytes": b"realized samples"},
        {
            "extractor": {
                "id": "acoustic.spectral",
                "version": 2,
                "params": {},
                "schema": {"names": ["centroid", "flatness"]},
            }
        },
        {
            "extractor": {
                "id": "acoustic.spectral",
                "version": 1,
                "params": {"window_ms": 25},
                "schema": {"names": ["centroid", "flatness"]},
            }
        },
        {
            "extractor": {
                "id": "acoustic.spectral",
                "version": 1,
                "params": {},
                "schema": {"names": ["centroid"]},
            }
        },
        {"feature_order": ["flatness", "centroid"]},
        {"nan_policy": "impute_zero"},
        {"dtype": "float32"},
        {"code_digest": "sha256:v1:changed-code"},
    ]
    assert all(cache_key(**cache_inputs(**change)) != baseline for change in variations)


def test_incomplete_key_can_reuse_stale_value_but_complete_key_cannot():
    def incomplete_key(inputs):
        return inputs["audio_bytes"], inputs["extractor"]["id"]

    old = cache_inputs(
        extractor={"id": "x", "version": 1, "params": {}, "schema": {}}, random_seed=1
    )
    new = cache_inputs(
        extractor={"id": "x", "version": 2, "params": {}, "schema": {}}, random_seed=99
    )
    stale_cache = {incomplete_key(old): "old features"}

    assert stale_cache[incomplete_key(new)] == "old features"
    assert cache_key(**old) != cache_key(**new)


def test_key_is_versioned_and_deterministic_for_mapping_order():
    left = cache_key(**cache_inputs(segmenter={"params": {"b": 2, "a": 1}, "version": 1, "id": "x"}))
    right = cache_key(**cache_inputs(segmenter={"id": "x", "version": 1, "params": {"a": 1, "b": 2}}))
    assert left == right
    assert left.startswith("sha256:v1:")


@pytest.mark.parametrize(
    "field,value",
    [("audio_bytes", "not bytes"), ("feature_order", {"a", "b"}), ("nan_policy", object())],
)
def test_rejects_ambiguous_inputs(field, value):
    with pytest.raises((TypeError, ValueError)):
        cache_key(**cache_inputs(**{field: value}))


@pytest.mark.parametrize(
    "randomness",
    [{"random_seed": None}, {"random_seed": 1, "random_bytes": b"realization"}],
)
def test_requires_exactly_one_randomness_identity(randomness):
    with pytest.raises(ValueError, match="exactly one"):
        cache_key(**cache_inputs(**randomness))


def test_persistent_cache_round_trip_and_strict_schema(tmp_path):
    cache = FeatureCache(tmp_path)
    key = cache_key(**cache_inputs())
    cache.write(
        key,
        features={"centroid": 10.0, "flatness": 0.25},
        diagnostics={"failed_frames": 0, "timing_ms": 1.5},
        feature_order=("centroid", "flatness"),
        dtype="float64",
        nan_policy="reject",
    )

    assert cache.read(
        key,
        feature_order=("centroid", "flatness"),
        dtype="float64",
        nan_policy="reject",
    ) == ({"centroid": 10.0, "flatness": 0.25}, {"failed_frames": 0, "timing_ms": 1.5})
    assert cache.read(
        key,
        feature_order=("flatness", "centroid"),
        dtype="float64",
        nan_policy="reject",
    ) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_persistent_cache_rejects_non_finite_values(tmp_path, bad):
    cache = FeatureCache(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        cache.write(
            "sha256:v1:" + "a" * 64,
            features={"x.value": bad},
            diagnostics={},
            feature_order=("x.value",),
            dtype="float64",
            nan_policy="reject",
        )


def test_corrupt_or_partial_cache_entry_is_a_miss(tmp_path):
    cache = FeatureCache(tmp_path)
    key = "sha256:v1:" + "b" * 64
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "key": key}), encoding="utf-8")
    assert cache.read(key, feature_order=("x.value",), dtype="float64", nan_policy="reject") is None


def test_failed_cache_publish_never_exposes_partial_entry(tmp_path, monkeypatch):
    cache = FeatureCache(tmp_path)
    key = "sha256:v1:" + "c" * 64

    def fail_publish(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        cache.write(
            key,
            features={"x.value": 1.0},
            diagnostics={},
            feature_order=("x.value",),
            dtype="float64",
            nan_policy="reject",
        )
    assert cache.read(key, feature_order=("x.value",), dtype="float64", nan_policy="reject") is None


def test_concurrent_identical_cache_writers_publish_one_complete_entry(tmp_path):
    cache = FeatureCache(tmp_path)
    key = "sha256:v1:" + "d" * 64
    errors = []

    def write():
        try:
            cache.write(
                key,
                features={"x.value": 1.0},
                diagnostics={"ok": True},
                feature_order=("x.value",),
                dtype="float64",
                nan_policy="reject",
            )
        except (OSError, TypeError, ValueError) as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert cache.read(
        key, feature_order=("x.value",), dtype="float64", nan_policy="reject"
    ) == ({"x.value": 1.0}, {"ok": True})
