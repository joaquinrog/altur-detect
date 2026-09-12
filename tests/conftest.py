"""Fixtures.

⚠️ El WAV dorado lo GENERAMOS nosotros — tonos y ruido. El obvio sería una llamada real del
dataset, y eso no se puede commitear ni en privado, porque el repo se vuelve público al
final y los términos de Altur prohíben redistribuir.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from altur.io import write_wav

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "synthetic_golden.wav"


def _synth(seconds: float = 6.0, sr: int = 8000, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Dos canales con estructura de turnos alternados. No es voz: es un sustituto determinista."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    floor0 = rng.normal(0, 0.002, n)
    floor1 = rng.normal(0, 0.002, n)
    ch0 = floor0.copy()
    ch1 = floor1.copy()
    # ch0 habla en ventanas impares, ch1 en pares: turnos que no se solapan.
    for k in range(int(seconds)):
        sl = slice(k * sr, (k + 1) * sr)
        env = np.hanning(sr)
        if k % 2:
            ch0[sl] += 0.30 * env * np.sin(2 * np.pi * 180 * t[sl])
        else:
            ch1[sl] += 0.22 * env * np.sin(2 * np.pi * 320 * t[sl])
    return ch0.astype(np.float32), ch1.astype(np.float32)


@pytest.fixture(scope="session")
def golden_path() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not GOLDEN.exists():
        ch0, ch1 = _synth()
        write_wav(str(GOLDEN), ch0, ch1)
    return GOLDEN


@pytest.fixture(scope="session")
def golden_bytes(golden_path: Path) -> bytes:
    return golden_path.read_bytes()


@pytest.fixture(scope="session")
def golden_b64(golden_bytes: bytes) -> str:
    return base64.b64encode(golden_bytes).decode()


@pytest.fixture(scope="session")
def mono_bytes(tmp_path_factory) -> bytes:
    p = tmp_path_factory.mktemp("audio") / "mono.wav"
    ch0, _ = _synth()
    write_wav(str(p), ch0)
    return p.read_bytes()


@pytest.fixture(scope="session")
def wideband_bytes(tmp_path_factory) -> bytes:
    """16 kHz: fuera de contrato. Prueba el remuestreo defensivo."""
    p = tmp_path_factory.mktemp("audio") / "wb.wav"
    sr = 16000
    t = np.arange(4 * sr) / sr
    write_wav(str(p), 0.3 * np.sin(2 * np.pi * 200 * t),
              0.2 * np.sin(2 * np.pi * 400 * t), sr=sr)
    return p.read_bytes()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from altur.api import app
    with TestClient(app) as c:
        yield c
