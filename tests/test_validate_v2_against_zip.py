from __future__ import annotations

import importlib.util
import io
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_v2_against_zip.py"
PREFIX = "robustness_phase1/robustness/data/validation_corrupted"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("validate_v2_against_zip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav(ch0: np.ndarray, ch1: np.ndarray) -> bytes:
    pcm = np.stack([ch0, ch1], axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _call(i: int) -> str:
    # Id con la forma real armado en tiempo de ejecución: ningún literal vive en el código.
    return "call_" + format(i, "012x")


def _build(tmp_path: Path, mod) -> tuple[Path, Path]:
    from altur.io import decode
    from altur.registry import transforms

    rng = np.random.default_rng(0)
    rows = ["anon_id,label,split,duration_s"]
    zpath = tmp_path / "robustness.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(4):
            split = "val" if i == 3 else "train"
            rows.append(f"{_call(i)},human,{split},6")
            t = np.arange(6 * 8000) / 8000
            clean = _wav((0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t) + 0.01 * rng.normal(size=t.size)).astype(np.float32),
                         (0.1 * np.sin(2 * np.pi * 900 * t)).astype(np.float32))
            zf.writestr(f"{PREFIX}/clean/{_call(i)}.wav", clean)
            ex = decode(clean).example
            low = transforms.resolve("lowpass_3400@1")(ex, None)
            zf.writestr(f"{PREFIX}/lowpass_3400hz/{_call(i)}.wav", _wav(low.ch0, low.ch1))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    return zpath, manifest


def test_find_prefix_detecta_la_carpeta_clean(mod):
    assert mod.find_prefix(["x/", f"{PREFIX}/clean/{_call(1)}.wav"]) == PREFIX
    with pytest.raises(ValueError):
        mod.find_prefix(["otra/cosa.wav"])


def test_train_ids_nunca_incluye_val(mod, tmp_path):
    _, manifest = _build(tmp_path, mod)
    ids = mod.train_ids(manifest)
    assert _call(3) not in ids and len(ids) == 3


def test_compare_de_una_senal_consigo_misma_es_neutro(mod):
    x = np.sin(np.arange(16000) / 7).astype(np.float32)
    m = mod.compare(x, x)
    assert m["len_ratio"] == 1 and m["rms_ratio"] == pytest.approx(1) and m["lsd_db"] == pytest.approx(0, abs=1e-9)


def test_run_coincide_consigo_mismo_salta_val_y_no_emite_ids(mod, tmp_path, capsys):
    zpath, manifest = _build(tmp_path, mod)
    rows = {r["condition"]: r for r in mod.run(zpath, manifest, limit=10, seed=7)}
    assert rows["clean"]["n"] == 3 and rows["clean"]["lsd_db"] == pytest.approx(0, abs=1e-6)
    assert rows["lowpass_3400hz"]["n"] == 3 and rows["lowpass_3400hz"]["lsd_db"] < 0.5
    assert rows["opus_16kbps"]["n"] == 0 and rows["opus_16kbps"]["lsd_db"] is None
    assert mod.main(["--zip", str(zpath), "--manifest", str(manifest), "--limit", "10"]) == 0
    out = capsys.readouterr().out
    assert "call_" not in out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
def test_la_cadena_v2_resuelve_todas_las_refs(mod):
    from altur.registry import transforms

    for refs in mod.load_conditions().values():
        for ref in refs:
            assert callable(transforms.resolve(ref))
