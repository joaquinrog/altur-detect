import json

import pytest

from altur.provenance import effective_code_digest, write_checkpoint


def _write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_effective_digest_uses_dirty_sources_specs_and_configs(tmp_path):
    source = _write(tmp_path, "src/altur/example.py", "VALUE = 1\n")
    spec = _write(tmp_path, "configs/experiments/baseline.yaml", "seed: 7\n")
    config = _write(tmp_path, "pyproject.toml", "[project]\nname = 'example'\n")
    baseline = effective_code_digest(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    source_changed = effective_code_digest(tmp_path)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    spec.write_text("seed: 8\n", encoding="utf-8")
    spec_changed = effective_code_digest(tmp_path)
    spec.write_text("seed: 7\n", encoding="utf-8")
    config.write_text("[project]\nname = 'changed'\n", encoding="utf-8")

    assert len({baseline, source_changed, spec_changed, effective_code_digest(tmp_path)}) == 4
    assert baseline.startswith("sha256:v1:")


def test_effective_digest_is_deterministic_and_excludes_private_or_derived_roots(tmp_path):
    _write(tmp_path, "scripts/run.py", "print('run')\n")
    _write(tmp_path, "configs/experiments/spec.yaml", "name: baseline\n")
    baseline = effective_code_digest(tmp_path)

    for relative in (
        "data/manifest.csv",
        "cache/features.json",
        "models/model.joblib",
        "docs/private.md",
        "experiments/runs/run.json",
        "configs/protocol/groups_v1.csv",
        "configs/protocol/folds_v1.json",
    ):
        _write(tmp_path, relative, "private or derived\n")

    assert effective_code_digest(tmp_path) == baseline
    assert effective_code_digest(tmp_path) == effective_code_digest(tmp_path)


def test_untracked_source_is_part_of_effective_digest(tmp_path):
    _write(tmp_path, "src/altur/tracked.py", "TRACKED = True\n")
    before = effective_code_digest(tmp_path)
    _write(tmp_path, "src/altur/new_dirty_file.py", "UNTRACKED = True\n")
    assert effective_code_digest(tmp_path) != before


def test_checkpoint_index_is_hashed_grouped_and_immutable(tmp_path):
    artifact = _write(tmp_path, "private/result.bin", "secret payload")
    index_root = tmp_path / ".checkpoints"

    record_path = write_checkpoint(
        "L0",
        "baseline",
        {"protocol": artifact},
        code_digest="sha256:v1:effective",
        root=index_root,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record_path == index_root / "L0" / "baseline.json"
    assert record["gate"] == "L0"
    assert record["checkpoint"] == "baseline"
    assert record["code_digest"] == "sha256:v1:effective"
    assert record["artifacts"][0]["name"] == "protocol"
    assert record["artifacts"][0]["sha256"].startswith("sha256:")
    assert record["artifacts"][0]["bytes"] == len(b"secret payload")
    assert "secret payload" not in record_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_checkpoint(
            "L0",
            "baseline",
            {"protocol": artifact},
            code_digest="sha256:v1:changed",
            root=index_root,
        )


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b"])
def test_checkpoint_rejects_unsafe_gate_and_checkpoint_names(tmp_path, value):
    artifact = _write(tmp_path, "artifact.bin", "payload")
    kwargs = {
        "artifacts": {"artifact": artifact},
        "code_digest": "sha256:v1:effective",
        "root": tmp_path / ".checkpoints",
    }
    with pytest.raises(ValueError):
        write_checkpoint(value, "baseline", **kwargs)
    with pytest.raises(ValueError):
        write_checkpoint("L0", value, **kwargs)
