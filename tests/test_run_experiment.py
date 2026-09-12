import importlib.util
from pathlib import Path

import pytest

from altur.runner import RunnerError, validate_experiment_spec

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_experiment.py"


def _script_module():
    spec = importlib.util.spec_from_file_location("run_experiment_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_spec_rejects_duplicate_keys_and_requires_mapping(tmp_path, suffix):
    module = _script_module()
    path = tmp_path / f"bad{suffix}"
    path.write_text('{"split":"train","split":"val"}' if suffix == ".json" else "split: train\nsplit: val\n")

    with pytest.raises(RunnerError, match="duplicate key"):
        module.load_spec(path)


def test_cli_rejects_val_before_constructing_dataset(tmp_path, monkeypatch):
    module = _script_module()
    path = tmp_path / "val.json"
    path.write_text(
        '{"schema_version":1,"extractor":"behavioral@1","segmenter":"energy@1",'
        '"feature_order":["behavioral.lat_med","behavioral.caller_bargein_rate"],'
        '"dtype":"float64","nan_policy":"reject","split":"val","random_seed":7,'
        '"transforms":[]}'
    )
    monkeypatch.setattr(module, "Dataset", lambda *_args, **_kwargs: pytest.fail("Dataset constructed"))

    with pytest.raises(RunnerError, match="split must be exactly 'train'"):
        module.main([str(path)])


def test_a2_specs_are_validated_and_have_exact_feature_order():
    from altur.features.acoustic_minimal import FEATURE_ORDER, _names

    module = _script_module()
    configs = ROOT / "configs" / "experiments"
    expected = {
        "a2_acoustic_ch0_v1.yaml": ("a2.acoustic.ch0@1", list(FEATURE_ORDER)),
        "a2_acoustic_ch1_v1.yaml": ("a2.acoustic.ch1@1", list(_names("a2.acoustic.ch1"))),
        "a2_behavioral_v1.yaml": (
            "behavioral@1",
            ["behavioral.lat_med", "behavioral.caller_bargein_rate"],
        ),
    }

    for filename, (extractor, feature_order) in expected.items():
        config = module.load_spec(configs / filename)
        validate_experiment_spec(config)
        assert config["extractor"] == extractor
        assert config["segmenter"] == "energy@1"
        assert config["feature_order"] == feature_order
