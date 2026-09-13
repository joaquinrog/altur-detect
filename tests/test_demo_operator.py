from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo_operator.py"
SPEC = importlib.util.spec_from_file_location("demo_operator", SCRIPT)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo)


def test_select_lan_ip_ignores_loopback_and_container_networks():
    output = """\
lo               UNKNOWN        127.0.0.1/8 ::1/128
docker0          DOWN           172.17.0.1/16
wlp0s20f3        UP             10.22.219.231/20 fe80::1/64
"""

    assert demo.select_lan_ip(output) == ("wlp0s20f3", "10.22.219.231")


def test_select_lan_ip_can_use_a_wired_interface():
    output = "enp3s0 UP 192.168.4.20/24\nwlp0s20f3 DOWN\n"

    assert demo.select_lan_ip(output) == ("enp3s0", "192.168.4.20")


def test_format_call_reports_verdict_confidence_and_total():
    call = {
        "ts": 1,
        "ref": "a1b2c3",
        "status": 200,
        "is_synthetic": True,
        "confidence": 0.973,
        "server_ms": 143.2,
    }

    text = demo.format_call(call)

    assert "SINTÉTICA" in text
    assert "97.3%" in text
    assert "0.143 s" in text
    assert "a1b2c3" in text


def test_format_call_makes_errors_visible():
    assert "ERROR HTTP 400" in demo.format_call({"status": 400, "ref": "bad"})


def test_write_state_is_private(tmp_path):
    path = tmp_path / "state.json"

    demo.write_state(path, {"token": "secret", "port": 8000})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert demo.read_state(path)["token"] == "secret"


def test_call_key_changes_only_when_the_latest_call_changes():
    call = {"ts": 1, "ref": "abc", "status": 200, "server_ms": 20}

    assert demo.call_key(call) == demo.call_key(dict(call))
    assert demo.call_key(call) != demo.call_key({**call, "ref": "def"})


def test_readiness_transition_alerts_when_detector_falls():
    assert demo.readiness_transition(True, False) == "[ALERTA] Detector no listo."
    assert demo.readiness_transition(False, False) is None
    assert demo.readiness_transition(False, True) == "[OK] Detector listo."
