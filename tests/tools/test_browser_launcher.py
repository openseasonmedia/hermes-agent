"""Behaviour contracts of the Python Chromium launcher (tools/browser_launcher.py).

A fake ``chromium`` (a Python script) stands in for the real binary: it writes ``DevToolsActivePort``
into its ``--user-data-dir`` exactly like Chromium and records its argv, so every path through the
launcher runs for real (spawn, readiness poll, registry, records, parent-first stop, orphan reap).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tools import browser_launcher as launcher

_FAKE_CHROMIUM = textwrap.dedent('''\
    #!{python}
    import json, os, signal, subprocess, sys, time
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    os.makedirs(udd, exist_ok=True)
    log = os.path.join(udd, "order.log")
    with open(os.path.join(udd, "argv.json"), "w") as f:
        json.dump(sys.argv, f)
    # A helper child like Chromium's zygote: records when IT is told to stop.
    child = subprocess.Popen([sys.executable, "-c",
        "import os,signal,sys,time\\n"
        "def h(*_):\\n"
        "    open(sys.argv[1],'a').write('child\\\\n'); sys.exit(0)\\n"
        "signal.signal(signal.SIGTERM, h)\\n"
        "parent = os.getppid()\\n"
        "while os.getppid() == parent: time.sleep(0.1)\\n"
        "time.sleep(5)\\n", log])  # outlives a SIGKILLed parent briefly, then exits (no leak)
    def on_term(*_):
        open(log, "a").write("parent\\n")
        sys.exit(0)  # leaves the child running, as a browser mid-shutdown would
    signal.signal(signal.SIGTERM, on_term)
    with open(os.path.join(udd, "DevToolsActivePort"), "w") as f:
        f.write("45678\\n/devtools/browser/fake-id\\n")
    time.sleep(600)
''')


@pytest.fixture
def fake_chromium(tmp_path, monkeypatch):
    script = tmp_path / "chromium"
    script.write_text(_FAKE_CHROMIUM.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setattr(launcher, "_resolve_executable", lambda: str(script))
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr("tools.browser_session_cloud.chromium_args", list)
    monkeypatch.setattr("tools.browser_session_cloud.is_headed_mode", lambda: False)
    yield script
    launcher.stop_all_browsers()


def _records() -> Path:
    """The test profile's launch records dir (created private, as the launcher creates it)."""
    from hermes_constants import get_hermes_home
    path = Path(get_hermes_home()) / "cache" / "browser" / "chromium-records"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _argv(browser) -> list[str]:
    return json.loads(Path(browser.user_data_dir, "argv.json").read_text())


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_launch_reads_devtools_port_and_runs_headless_on_an_ephemeral_home_profile(fake_chromium):
    browser, err = launcher.launch_chromium("task-a", env=dict(os.environ))
    assert err is None
    assert browser.cdp_url == "ws://127.0.0.1:45678/devtools/browser/fake-id"
    assert browser.http_url == "http://127.0.0.1:45678"
    assert browser.owned and browser.ephemeral
    from hermes_constants import get_hermes_home
    assert Path(browser.user_data_dir).parent == Path(get_hermes_home()) / "cache" / "browser" / "chromium"
    argv = _argv(browser)
    assert "--remote-debugging-port=0" in argv
    assert f"--user-data-dir={browser.user_data_dir}" in argv
    assert "--headless=new" in argv
    assert argv[-1] == "about:blank"


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_launch_record_lives_in_the_profiles_private_state_dir(fake_chromium):
    """Launcher state lives in a private per-profile dir: the record under ``<HERMES_HOME>/cache/browser/
    chromium-records`` (0700) and the throwaway profile under the 0700 ephemeral root beside it."""
    from hermes_constants import get_hermes_home
    browser, err = launcher.launch_chromium("recorded", env=dict(os.environ))
    assert err is None
    state = Path(get_hermes_home()) / "cache" / "browser"
    assert stat.S_IMODE((state / "chromium-records").stat().st_mode) == 0o700
    assert stat.S_IMODE((state / "chromium").stat().st_mode) == 0o700
    [record] = (state / "chromium-records").glob("*.json")
    assert json.loads(record.read_text())["pid"] == browser.pid
    launcher.stop_browser("recorded")
    assert not list((state / "chromium-records").glob("*.json"))


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_one_browser_per_key_and_separate_browsers_for_separate_keys(fake_chromium):
    a1, _ = launcher.launch_chromium("named-a", env=dict(os.environ))
    a2, _ = launcher.launch_chromium("named-a", env=dict(os.environ))
    b, _ = launcher.launch_chromium("named-b", env=dict(os.environ))
    assert a1.pid == a2.pid
    assert b.pid != a1.pid
    assert b.user_data_dir != a1.user_data_dir


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_two_processes_on_one_profile_never_share_an_ephemeral_profile(fake_chromium):
    """CLI + gateway on one profile launch the same key (``browser-exec-default``, a named session): each
    process gets its own throwaway profile, and a launch never wipes the other process's live one."""
    from hermes_constants import hermes_home_key
    code = ("import sys; from tools.browser_launcher import _default_user_data_dir; "
            "print(_default_user_data_dir(sys.argv[1], sys.argv[2]))")
    other = Path(subprocess.run([sys.executable, "-c", code, "browser-exec-default", hermes_home_key()],
                                cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
                                capture_output=True, text=True, check=True).stdout.strip())
    (other / "Default").mkdir(parents=True)
    (other / "Default" / "Cookies").write_text("the other process's session")

    mine, err = launcher.launch_chromium("browser-exec-default", env=dict(os.environ))
    assert err is None
    assert Path(mine.user_data_dir) != other
    assert (other / "Default" / "Cookies").read_text() == "the other process's session"


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if the exit teardown failed
def test_a_bare_consumer_exiting_normally_stops_its_browser(fake_chromium):
    """A script / eval that imports only the session layer and exits normally: the atexit teardown must stop
    the browser (the teardown's process-tree helper is loaded before exit, not first imported during it)."""
    script = ("import os, sys\n"
              "import tools.browser_session  # registers the exit teardown\n"
              "from tools import browser_launcher\n"
              "browser_launcher.needs_chromium_sandbox_bypass = lambda: False\n"
              "browser, err = browser_launcher.launch_chromium('bare', executable=sys.argv[1], headed=False,\n"
              "                                                env=dict(os.environ), timeout=30)\n"
              "print(browser.pid if browser else err, flush=True)\n")
    done = subprocess.run([sys.executable, "-c", script, str(fake_chromium)], check=False,
                          cwd=Path(__file__).resolve().parents[2],
                          env=dict(os.environ), capture_output=True, text=True, timeout=120)
    pid = int(done.stdout.strip().splitlines()[0])
    deadline = time.monotonic() + 10
    while launcher._pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    survived = launcher._pid_alive(pid)
    if survived and str(fake_chromium) in Path(f"/proc/{pid}/cmdline").read_text(errors="replace"):
        os.kill(pid, signal.SIGKILL)
    assert not survived, done.stderr[-2000:]


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_stop_signals_the_browser_before_its_helpers_and_removes_the_ephemeral_profile(fake_chromium):
    browser, _ = launcher.launch_chromium("stop-me", env=dict(os.environ))
    order_log = Path(browser.user_data_dir, "order.log")
    snapshot = order_log.parent.parent / "order-snapshot.log"
    # The profile dir is removed on stop; mirror the log before it goes.
    real_rmtree = launcher.shutil.rmtree

    def keep_log(path, *a, **k):
        if Path(path) == Path(browser.user_data_dir) and order_log.exists():
            snapshot.write_text(order_log.read_text())
        return real_rmtree(path, *a, **k)

    launcher.shutil.rmtree = keep_log
    try:
        launcher.stop_browser("stop-me")
    finally:
        launcher.shutil.rmtree = real_rmtree
    assert snapshot.read_text().splitlines() == ["parent", "child"]
    assert not Path(browser.user_data_dir).exists()
    assert launcher.get_browser("stop-me") is None


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_a_dead_browser_is_dropped_so_the_next_launch_starts_a_new_one(fake_chromium):
    first, _ = launcher.launch_chromium("crashy", env=dict(os.environ))
    os.kill(first.pid, signal.SIGKILL)
    launcher._procs["crashy"].wait(timeout=5)
    assert launcher.get_browser("crashy") is None
    second, err = launcher.launch_chromium("crashy", env=dict(os.environ))
    assert err is None and second.pid != first.pid


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_launch_failure_reports_the_exit_and_leaves_nothing_registered(tmp_path, monkeypatch, fake_chromium):
    broken = tmp_path / "broken-chromium"
    broken.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('No usable sandbox!')\nsys.exit(7)\n")
    broken.chmod(0o755)
    browser, err = launcher.launch_chromium("broken", executable=str(broken), env=dict(os.environ), timeout=10)
    assert browser is None
    assert "exited with code 7" in err and "--no-sandbox" in err
    assert launcher.get_browser("broken") is None


@pytest.mark.platforms("posix")
def test_a_browser_that_exited_during_startup_is_not_signalled(tmp_path, caplog, fake_chromium):
    """A Chromium that died during startup is only reaped: signalling its pid would be refused as recycled
    and logged as a WARNING that blames the wrong thing."""
    import logging
    broken = tmp_path / "exits-at-once"
    broken.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(21)\n")
    broken.chmod(0o755)
    caplog.set_level(logging.WARNING)
    browser, err = launcher.launch_chromium("dies", executable=str(broken), env=dict(os.environ), timeout=10)
    assert browser is None and "exited with code 21" in err
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def _write_orphan_record(records: Path, pid: int, udd: str) -> Path:
    dead_owner = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_owner.wait()
    path = records / f"orphan-{pid}.json"
    path.write_text(json.dumps({"owner_pid": dead_owner.pid, "owner_start_time": None, "pid": pid,
                                "start_time": launcher._start_time(pid), "user_data_dir": udd,
                                "home_key": "x", "ephemeral": False}))
    return path


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_orphan_reap_refuses_a_pid_whose_cmdline_lacks_the_recorded_user_data_dir(tmp_path, fake_chromium):
    stranger = subprocess.Popen(["sleep", "3001"])
    try:
        record = _write_orphan_record(_records(), stranger.pid, str(tmp_path / "some-profile"))
        assert launcher.reap_orphaned_browsers() == 0
        assert stranger.poll() is None  # never signalled
        assert not record.exists()  # the stale record is dropped
    finally:
        stranger.kill()
        stranger.wait()


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_orphan_reap_stops_a_crashed_owners_browser(tmp_path, fake_chromium):
    udd = tmp_path / "orphan-profile"
    proc = subprocess.Popen([str(fake_chromium), f"--user-data-dir={udd}"], start_new_session=True)
    deadline = time.monotonic() + 10
    while not (udd / "DevToolsActivePort").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    _write_orphan_record(_records(), proc.pid, str(udd))
    assert launcher.reap_orphaned_browsers() == 1
    proc.wait(timeout=10)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the fake browser's reparented helper is ours to stop
def test_records_of_a_live_owner_are_left_alone(tmp_path, fake_chromium):
    launcher.launch_chromium("mine", env=dict(os.environ))
    assert launcher.reap_orphaned_browsers() == 0
    assert launcher.get_browser("mine") is not None
    assert list(_records().glob("*.json"))


# A Hermes process on a persistent profile: launches (or attaches to) its Chromium, prints the pid, waits
# for a line on stdin, then exits normally (the session layer's exit teardown stops its browsers).
_SHARED_PROFILE_USER = textwrap.dedent("""\
    import os, sys
    import tools.browser_session  # registers the exit teardown
    from tools import browser_launcher as launcher
    launcher.needs_chromium_sandbox_bypass = lambda: False
    exe, udd, running = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if running:  # Chromium's singleton would hand a second launch to the running instance
        launcher.running_instance = lambda d: (45678, "/devtools/browser/fake-id", running) if d == udd else None
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned}" if browser else err, flush=True)
    sys.stdin.readline()
""")


def _shared_profile_user(fake_chromium, udd, running: int = 0) -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen([sys.executable, "-c", _SHARED_PROFILE_USER, str(fake_chromium), str(udd), str(running)],
                            cwd=Path(__file__).resolve().parents[2], env=dict(os.environ), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline().split()
    assert line and line[0].isdigit(), proc.stderr.read()[-2000:] if proc.poll() is not None else line
    assert (line[1] == "True") is not bool(running)
    return proc, int(line[0])


def _gone(pid: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while launcher._pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not launcher._pid_alive(pid)


def _kill_fake(pid: int, fake_chromium) -> None:
    with contextlib.suppress(OSError):
        if str(fake_chromium) in Path(f"/proc/{pid}/cmdline").read_text(errors="replace"):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_shared_profile_browser_outlives_its_launcher_while_another_process_holds_it(fake_chromium, tmp_path):
    """Two Hermes processes on one persistent profile: the one that launched its Chromium lets go (its normal
    exit) while the other still drives it. The browser stays up for the other one, and that last holder's
    own normal exit stops it, leaving no launcher state behind."""
    udd, key = tmp_path / "shared-profile", f"profile:{tmp_path / 'shared-profile'}"
    mine, err = launcher.launch_chromium(key, executable=str(fake_chromium), user_data_dir=str(udd), ephemeral=False,
                                         headed=False, env=dict(os.environ), timeout=30)
    assert err is None and mine.owned
    holder, _ = _shared_profile_user(fake_chromium, udd, running=mine.pid)
    try:
        launcher.stop_browser(key)
        assert launcher._pid_alive(mine.pid), "the launcher let go of a browser another process still drives"

        holder.communicate("\n", timeout=60)  # the last holder exits normally
        assert _gone(mine.pid)
        assert not [p.name for p in _records().iterdir() if not p.name.startswith(".")]
    finally:
        holder.kill()
        _kill_fake(mine.pid, fake_chromium)


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # the fake browser is ours to stop if a leg failed
def test_a_crashed_launchers_shared_browser_is_reaped_only_after_its_last_holder_is_gone(fake_chromium, tmp_path):
    """The launcher crashed while another live Hermes process holds its browser: no reaper kills it. Once
    that holder crashed too (proven by pid + start time), the next reap stops it."""
    udd = tmp_path / "shared-profile"
    owner, pid = _shared_profile_user(fake_chromium, udd)
    holder, _ = _shared_profile_user(fake_chromium, udd, running=pid)
    try:
        owner.kill()
        owner.wait(timeout=10)
        assert launcher.reap_orphaned_browsers() == 0
        assert launcher._pid_alive(pid), "reaped while a live process still holds it"

        holder.kill()
        holder.wait(timeout=10)
        assert launcher.reap_orphaned_browsers() == 1
        assert _gone(pid)
    finally:
        owner.kill()
        holder.kill()
        _kill_fake(pid, fake_chromium)


@pytest.mark.parametrize("headed,display,expect_headless", [
    (False, ":1", True),
    (True, ":1", False),
    (True, "", True),  # Linux without a display: a headed launch would die
])
@pytest.mark.platforms("linux")
def test_headed_only_when_configured_and_a_display_exists(headed, display, expect_headless):
    env = {"DISPLAY": display} if display else {}
    assert launcher._resolve_headed(headed, env) is (not expect_headless)


@pytest.mark.platforms("posix")
def test_packaged_only_flags_never_reach_an_explicit_binary(monkeypatch):
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: True)
    monkeypatch.setattr("tools.browser_session_cloud.chromium_args", lambda: ["--lang=de"])
    packaged = launcher.build_chromium_argv("/c", "/p", packaged=True, headed=False, sandbox_bypass=True)
    dock = launcher.build_chromium_argv("/c", "/p", packaged=False, headed=True, sandbox_bypass=True)
    real = launcher.build_chromium_argv("/c", "/p", packaged=False, headed=True, sandbox_bypass=False)
    assert "--password-store=basic" in packaged and "--password-store=basic" not in real
    assert "--headless=new" in packaged and "--headless=new" not in real
    assert set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS) <= set(packaged) and set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS) <= set(dock)
    assert not set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS) & set(real)  # the user's own browser keeps its sandbox
    for argv in (packaged, dock, real):
        assert "--lang=de" in argv and argv[-1] == "about:blank"
