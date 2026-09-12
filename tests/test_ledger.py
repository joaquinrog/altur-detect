import json
import os
import threading

import pytest

from altur.ledger import LedgerError, write_aggregate, write_run


def payload(i=1):
    return {
        "commit": "abc123",
        "data_hash": "sha256:data",
        "environment": {"python": "3.12"},
        "code_version": "0.1.0",
        "results": {"auc": 0.91, "run_number": i},
    }


def test_writes_one_immutable_json_per_run_and_keeps_extra_payload(tmp_path):
    path = write_run("run-001", payload(), root=tmp_path)
    assert path == tmp_path / "run-001.json"
    assert json.loads(path.read_text()) == payload()
    with pytest.raises(FileExistsError):
        write_run("run-001", payload(2), root=tmp_path)
    assert json.loads(path.read_text())["results"]["run_number"] == 1


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ".", "", "a\\b", "aggregate"])
def test_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises((LedgerError, ValueError)):
        write_run(run_id, payload(), root=tmp_path)


def test_requires_reproducibility_fields(tmp_path):
    for field in ("commit", "data_hash", "environment", "code_version"):
        incomplete = payload()
        del incomplete[field]
        with pytest.raises(LedgerError):
            write_run("run", incomplete, root=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit", ""),
        ("data_hash", None),
        ("environment", "python=3.12"),
        ("code_version", ""),
    ],
)
def test_validates_reproducibility_fields(tmp_path, field, value):
    invalid = payload()
    invalid[field] = value
    with pytest.raises(LedgerError):
        write_run("run", invalid, root=tmp_path)


def test_concurrent_creation_has_one_winner(tmp_path):
    outcomes = []

    def create():
        try:
            write_run("same", payload(), root=tmp_path)
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=create) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("created") == 1
    assert outcomes.count("exists") == 7


def test_failed_atomic_publish_never_exposes_partial_run(tmp_path, monkeypatch):
    def fail_publish(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        write_run("interrupted", payload(), root=tmp_path)
    assert not (tmp_path / "interrupted.json").exists()


def test_aggregate_is_sorted_deterministic_and_not_a_source(tmp_path):
    write_run("z-run", payload(2), root=tmp_path)
    write_run("a-run", payload(1), root=tmp_path)
    aggregate = write_aggregate(root=tmp_path)
    first = aggregate.read_bytes()
    assert json.loads(first)["runs"][0]["run_id"] == "a-run"
    assert aggregate.name == "aggregate.json"
    write_run("m-run", payload(3), root=tmp_path)
    write_aggregate(root=tmp_path)
    assert first != aggregate.read_bytes()
    assert all(item["run_id"] != "aggregate" for item in json.loads(aggregate.read_text())["runs"])
