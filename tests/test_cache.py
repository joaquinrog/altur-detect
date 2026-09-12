import pytest

from altur.cache import cache_key


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
