"""The sandbox port forward (tools/environments/streams.py) reports connections that never reached their
target: the sandbox browser's liveness is re-checked only after one (tools/browser_session_sandbox.py)."""

import socket
import subprocess
import sys
import threading
import time

import pytest

from tools.environments import streams


@pytest.fixture
def local_relay(monkeypatch):
    """The relay runs on this host instead of inside a sandbox: same script, same pipes."""
    def open_stream(env, argv, **kw):
        return subprocess.Popen([sys.executable, *argv[1:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    monkeypatch.setattr(streams, "open_stream", open_stream)
    opened = []
    yield opened
    for env, port in opened:
        streams._forwards.pop((id(env), "127.0.0.1", port)).close()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _one_shot_server(reply: bytes) -> int:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        with conn, srv:
            conn.sendall(reply)
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def _relay(local_port: int) -> bytes:
    with socket.create_connection(("127.0.0.1", local_port), timeout=10) as conn:
        conn.settimeout(10)
        conn.shutdown(socket.SHUT_WR)  # the client is done sending; the relay ends with the target's reply
        chunks = []
        while chunk := conn.recv(65536):
            chunks.append(chunk)
        return b"".join(chunks)


def test_only_a_connection_that_never_reaches_its_target_counts_as_a_forward_failure(local_relay):
    env = object()
    live = _one_shot_server(b"devtools")
    local_relay.append((env, live))
    assert _relay(streams.forward_port(env, live)) == b"devtools"

    dead = _closed_port()  # Chromium died: nothing listens on its DevTools port any more
    local_relay.append((env, dead))
    assert _relay(streams.forward_port(env, dead)) == b""

    deadline = time.monotonic() + 10
    while streams.forward_failures(env, dead) == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert streams.forward_failures(env, dead) == 1
    assert streams.forward_failures(env, live) == 0
