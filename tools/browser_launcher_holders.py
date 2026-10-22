"""Cross-process holders of a shared-profile Chromium (``tools/browser_launcher.py``).

A persistent profile (``browser.user_data_dir``, the Bot Desktop profile, the real-profile copy) has ONE
Chromium, launched by whichever Hermes process got there first; every other process attaches to it. Each
attacher leaves a lease beside the launch record in the profile's private records dir. The process holding
the record (the launcher, or whoever it was handed to) stops the browser on exit only when no other live
Hermes process holds a lease; otherwise it hands the record to one of them, so the LAST holder stops it. A
holder that crashed is dropped once its pid and start time no longer match, never on the pid alone.

Every decision runs under the records dir's lock: a lease written while another process decides cannot be
missed. Siblings late-import the facade (its ``_pid_alive`` / ``_start_time`` are the patched seams).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_LOCK_NAME = ".holders.lock"


@contextlib.contextmanager
def records_lock(records: Path) -> Iterator[None]:
    """Exclusive, cross-process lock on ``records`` (the profile's private launcher state dir). Never nested:
    two locks on one file in one process are two open file descriptions, and the second would wait forever."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(records / _LOCK_NAME, flags, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # retries for ~10 s, then raises
                    break
                except OSError:
                    continue
        else:
            import fcntl  # windows-footgun: ok — the nt branch above
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            with contextlib.suppress(OSError):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def _lease_path(record: Path, holder_pid: int) -> Path:
    return record.with_name(f"{record.stem}@{holder_pid}.lease")


def _load(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        return data
    path.unlink(missing_ok=True)
    return None


def process_alive(pid, start_time) -> bool:
    """``pid`` is still the process that wrote a record or lease: alive, and with the recorded start time
    (a recycled pid has another one)."""
    from tools import browser_launcher as launcher
    if not isinstance(pid, int) or not launcher._pid_alive(pid):
        return False
    return start_time is None or launcher._start_time(pid) == start_time


def write_lease(record: Path, browser) -> None:
    """This process holds ``browser`` (caller holds :func:`records_lock`)."""
    from tools import browser_launcher as launcher
    from utils import atomic_json_write
    me = os.getpid()
    atomic_json_write(_lease_path(record, me), {
        "holder_pid": me, "holder_start_time": launcher._start_time(me), "pid": browser.pid,
        "start_time": browser.start_time, "user_data_dir": browser.user_data_dir}, indent=None)


def drop_lease(record: Path) -> None:
    _lease_path(record, os.getpid()).unlink(missing_ok=True)


def other_live_holders(record: Path, browser_start_time) -> list[dict]:
    """Leases of OTHER live processes on the browser ``record`` names (caller holds :func:`records_lock`).
    A lease of a dead holder, or one naming an earlier browser on the same pid, is removed."""
    holders = []
    for path in sorted(record.parent.glob(f"{record.stem}@*.lease")):
        lease = _load(path)
        if lease is None or lease.get("holder_pid") == os.getpid():
            continue
        if lease.get("start_time") != browser_start_time or not process_alive(
                lease.get("holder_pid"), lease.get("holder_start_time")):
            path.unlink(missing_ok=True)
            continue
        holders.append(lease)
    return holders


def drop_dead_leases(records: Path) -> None:
    """Leases whose holder is gone (caller holds :func:`records_lock`): a lease outlives its launch record when
    the browser died while its holder crashed."""
    for path in records.glob("*.lease"):
        lease = _load(path)
        if lease is not None and not process_alive(lease.get("holder_pid"), lease.get("holder_start_time")):
            path.unlink(missing_ok=True)


def hand_over(record: Path, data: dict, holder: dict) -> None:
    """Make ``holder`` the record's owner: every reaper now sees a live owner, and the holder stops the
    browser when it is the last one out."""
    from utils import atomic_json_write
    atomic_json_write(record, {**data, "owner_pid": holder["holder_pid"],
                               "owner_start_time": holder.get("holder_start_time")}, indent=None)
    logger.info("browser: Chromium pid %s is still used by Hermes pid %s; handed it over instead of stopping it",
                data.get("pid"), holder["holder_pid"])
