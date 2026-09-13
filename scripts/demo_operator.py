#!/usr/bin/env python3
"""Opera y vigila el Monitor durante un ensayo local o el turno real.

La herramienta no envia llamadas, no usa ``val`` y no despliega. Bini o Altur son
quienes llaman a ``/detect``; este proceso prepara y observa el servidor de Joaquin.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

DEFAULT_IMAGE = "altur-detect:c2-mon-ui-s15"
DEFAULT_CONTAINER = "altur-demo"
DEFAULT_STATE = Path("/tmp/altur-demo-operator.json")


def select_lan_ip(output: str) -> tuple[str, str] | None:
    """Elige una IPv4 privada util, prefiriendo WiFi y despues Ethernet."""
    candidates: list[tuple[int, str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] == "lo" or fields[1] == "DOWN":
            continue
        interface = fields[0]
        if interface.startswith(("docker", "br-", "veth")):
            continue
        for field in fields[2:]:
            raw = field.split("/", 1)[0]
            try:
                address = ipaddress.ip_address(raw)
            except ValueError:
                continue
            if address.version == 4 and address.is_private and not address.is_link_local:
                priority = 0 if interface.startswith("wl") else 1
                candidates.append((priority, interface, raw))
    if not candidates:
        return None
    _, interface, address = min(candidates)
    return interface, address


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def call_key(call: dict[str, Any]) -> str:
    fields = ("ts", "ref", "status", "is_synthetic", "confidence", "server_ms", "ms")
    return json.dumps([call.get(field) for field in fields], separators=(",", ":"))


def readiness_transition(previous: bool | None, current: bool) -> str | None:
    if previous == current:
        return None
    return "[OK] Detector listo." if current else "[ALERTA] Detector no listo."


def total_ms(call: dict[str, Any]) -> float | None:
    if isinstance(call.get("server_ms"), (int, float)):
        return float(call["server_ms"])
    parts = [call.get("upload_ms"), call.get("ms")]
    values = [float(value) for value in parts if isinstance(value, (int, float))]
    return sum(values) if values else None


def format_call(call: dict[str, Any]) -> str:
    status = call.get("status")
    ref = str(call.get("ref") or "sin-ref")
    if status != 200:
        return f"{ref} | ERROR HTTP {status if status is not None else 'desconocido'}"
    verdict = "SINTÉTICA" if call.get("is_synthetic") is True else "HUMANA"
    confidence = call.get("confidence")
    confidence_text = (
        f"{float(confidence) * 100:.1f}%" if isinstance(confidence, (int, float)) else "n/d"
    )
    elapsed = total_ms(call)
    elapsed_text = f"{elapsed / 1000:.3f} s" if elapsed is not None else "n/d"
    return f"{ref} | {verdict} | confianza {confidence_text} | total {elapsed_text}"


def docker(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["sg", "docker", "-c", shlex.join(["docker", *args])]
    return subprocess.run(command, check=check, capture_output=True, text=True)


def request_json(url: str, token: str | None = None, timeout: float = 3) -> dict[str, Any]:
    if token:
        separator = "&" if "?" in url else "?"
        url += separator + urllib.parse.urlencode({"k": token})
    request = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def discover_lan_ip() -> tuple[str, str] | None:
    process = subprocess.run(
        ["ip", "-4", "-br", "addr"], check=True, capture_output=True, text=True
    )
    return select_lan_ip(process.stdout)


def port_available(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def wait_ready(base_url: str, seconds: int = 90) -> dict[str, Any] | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            data = request_json(f"{base_url}/health/ready", timeout=2)
            if data.get("status") == "ready":
                return data
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1)
    return None


def resolve_target(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.base_url:
        token = args.token or os.environ.get("ALTUR_MONITOR_TOKEN")
        return args.base_url.rstrip("/"), token
    if not args.state.exists():
        raise RuntimeError(f"no hay sesion local en {args.state}; ejecuta `start` primero")
    state = read_state(args.state)
    return str(state["base_url"]), str(state["token"])


def start(args: argparse.Namespace) -> int:
    if args.state.exists():
        print(f"Ya existe una sesion en {args.state}. Ejecuta `stop` antes de reemplazarla.", file=sys.stderr)
        return 2
    if not port_available(args.port):
        print(f"El puerto {args.port} esta ocupado. No se detuvo ningun proceso.", file=sys.stderr)
        return 2
    image = docker(["image", "inspect", args.image, "--format", "{{.Id}}"], check=False)
    if image.returncode != 0:
        print(f"No existe la imagen local {args.image}.", file=sys.stderr)
        return 2

    token = secrets.token_hex(16)
    run = docker([
        "run", "-d", "--rm", "--name", args.container,
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-p", f"0.0.0.0:{args.port}:8000",
        "-e", "ALTUR_MONITOR=1", "-e", f"ALTUR_MONITOR_TOKEN={token}",
        args.image,
    ], check=False)
    if run.returncode != 0:
        print(run.stderr.strip() or "Docker no pudo arrancar el contenedor.", file=sys.stderr)
        return 1

    base_url = f"http://127.0.0.1:{args.port}"
    ready = wait_ready(base_url)
    if ready is None:
        logs = docker(["logs", "--tail", "80", args.container], check=False)
        docker(["rm", "-f", args.container], check=False)
        print("El detector no llego a ready.\n" + logs.stdout[-3000:], file=sys.stderr)
        return 1

    lan = discover_lan_ip()
    state = {
        "container": args.container,
        "image": args.image,
        "image_id": image.stdout.strip(),
        "port": args.port,
        "token": token,
        "base_url": base_url,
        "lan_interface": lan[0] if lan else None,
        "lan_ip": lan[1] if lan else None,
        "started_at": time.time(),
    }
    write_state(args.state, state)
    monitor_url = f"{base_url}/monitor?{urllib.parse.urlencode({'k': token})}"
    print("\nLISTO PARA ENSAYO")
    print(f"Detector: {ready.get('detector', 'desconocido')}")
    print(f"Imagen:   {args.image} ({image.stdout.strip()[:19]}...)")
    print(f"Monitor:  {monitor_url}")
    if lan:
        print(f"Bini:     http://{lan[1]}:{args.port}/detect  ({lan[0]})")
        print(f"Health:   http://{lan[1]}:{args.port}/health/ready")
    else:
        print("Bini:     no se encontro una IPv4 LAN; revisa `ip -4 -br addr`")
    print("Vigilar:  python scripts/demo_operator.py watch")
    print("Cerrar:   python scripts/demo_operator.py stop")
    if not args.no_open:
        webbrowser.open(monitor_url)
    return 0


def show_status(args: argparse.Namespace) -> int:
    try:
        base_url, token = resolve_target(args)
        ready = request_json(f"{base_url}/health/ready")
        calls = request_json(f"{base_url}/monitor/calls?limit=100", token)
    except (RuntimeError, OSError, ValueError, urllib.error.URLError) as error:
        print(f"NO DISPONIBLE: {error}", file=sys.stderr)
        return 1
    summary = calls.get("summary") or {}
    print(f"READY: {ready.get('detector', 'desconocido')}")
    print(
        f"LLAMADAS: {summary.get('calls', 0)} | OK: {summary.get('ok', 0)} | "
        f"ERRORES: {summary.get('errors', 0)}"
    )
    recent = calls.get("calls") or []
    if recent:
        print("ULTIMA: " + format_call(recent[0]))
    return 0


def watch(args: argparse.Namespace) -> int:
    try:
        base_url, token = resolve_target(args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"Monitoreando {base_url}. Ctrl+C detiene el monitor, no el servidor.")
    started = time.monotonic()
    last_key: str | None = None
    last_connection: bool | None = None
    last_ready: bool | None = None
    last_errors = 0
    next_heartbeat = started
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            try:
                data = request_json(f"{base_url}/monitor/calls?limit=100", token)
                calls = data.get("calls") or []
                summary = data.get("summary") or {}
                if last_connection is not True:
                    print("[OK] Monitor conectado.")
                last_connection = True
                ready = data.get("ready") is True
                message = readiness_transition(last_ready, ready)
                if message:
                    print(message)
                last_ready = ready
                errors = int(summary.get("errors") or 0)
                if errors > last_errors:
                    print(f"[ALERTA] Nuevos errores: {errors - last_errors}; acumulados: {errors}")
                last_errors = errors
                if calls:
                    key = call_key(calls[0])
                    if key != last_key:
                        elapsed = total_ms(calls[0])
                        prefix = "[ALERTA >30s]" if elapsed is not None and elapsed > 30_000 else "[LLAMADA]"
                        print(prefix + " " + format_call(calls[0]))
                        last_key = key
                now = time.monotonic()
                if now >= next_heartbeat:
                    print(
                        f"[VIVO] total={summary.get('calls', 0)} ok={summary.get('ok', 0)} "
                        f"errores={summary.get('errors', 0)}"
                    )
                    next_heartbeat = now + args.heartbeat
            except (OSError, ValueError, urllib.error.URLError) as error:
                if last_connection is not False:
                    print(f"[ALERTA] Monitor sin respuesta: {error}")
                last_connection = False
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoreo detenido. El servidor sigue activo.")
    return 0


def stop(args: argparse.Namespace) -> int:
    if not args.state.exists():
        print("No hay sesion local registrada; nada que detener.")
        return 0
    state = read_state(args.state)
    container = str(state.get("container") or DEFAULT_CONTAINER)
    result = docker(["rm", "-f", container], check=False)
    args.state.unlink(missing_ok=True)
    if result.returncode != 0 and "No such container" not in result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
        return 1
    print(f"Sesion cerrada: {container}. Token efimero eliminado.")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--state", type=Path, default=DEFAULT_STATE, help=argparse.SUPPRESS)
    commands = root.add_subparsers(dest="command", required=True)

    start_parser = commands.add_parser("start", help="levanta la demo local y abre el Monitor")
    start_parser.add_argument("--image", default=DEFAULT_IMAGE)
    start_parser.add_argument("--container", default=DEFAULT_CONTAINER)
    start_parser.add_argument("--port", type=int, default=8000)
    start_parser.add_argument("--no-open", action="store_true")
    start_parser.set_defaults(func=start)

    for name, help_text, function in (
        ("status", "muestra readiness, conteos y ultima llamada", show_status),
        ("watch", "vigila llamadas, errores y caidas en vivo", watch),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--base-url", help="servidor remoto; si se omite usa la sesion local")
        command.add_argument("--token", help="preferir ALTUR_MONITOR_TOKEN para no exponerlo en ps")
        if name == "watch":
            command.add_argument("--interval", type=float, default=1.0)
            command.add_argument("--heartbeat", type=float, default=15.0)
            command.add_argument("--duration", type=float, default=0, help="segundos; 0 hasta Ctrl+C")
        command.set_defaults(func=function)

    stop_parser = commands.add_parser("stop", help="apaga la demo local y elimina el token")
    stop_parser.set_defaults(func=stop)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
