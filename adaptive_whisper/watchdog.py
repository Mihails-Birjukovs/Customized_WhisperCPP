"""Bounded native-process progress watchdog, standard-library only.

This tracks *observed progress callbacks*, not inferred transcription
correctness or acoustic content. No shell. Separate raw stdout/stderr files
avoid deadlocks and retain byte-exact logs for private audit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock, Thread
import os
import re
import subprocess
import time
from typing import Any


_PROGRESS = re.compile(rb"progress\s*=\s*(\d{1,3})\s*%", re.IGNORECASE)


@dataclass(frozen=True)
class WatchdogPolicy:
    startup_timeout_s: float = 90.0
    idle_progress_timeout_s: float = 90.0
    poll_s: float = 0.1
    terminate_grace_s: float = 2.0

    def __post_init__(self) -> None:
        if min(self.startup_timeout_s, self.idle_progress_timeout_s,
               self.poll_s, self.terminate_grace_s) <= 0:
            raise ValueError("Watchdog timeouts must be strictly positive")


def _bounded_end(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """Terminate only the direct native process, then escalate to kill."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=grace_s)
    except ProcessLookupError:
        return


def run_watched(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str,str],
    stdout_raw: Path,
    stderr_raw: Path,
    hard_timeout_s: float,
    policy: WatchdogPolicy,
) -> dict[str,Any]:
    """Run native executable, preserving raw bytes, with optional progress aborts.

    Aborts after too long with no first progress callback, too long since the
    last *advanced* progress percentage, or an absolute hard timeout. Output
    arriving without advanced progress is not counted as decoding progress.
    A process that naturally exits before first progress is not called stalled.
    """
    if hard_timeout_s <= 0 or not command:
        raise ValueError("Invalid command or hard timeout")
    if stdout_raw.exists() or stderr_raw.exists():
        raise FileExistsError("Native raw log must not be overwritten")
    if stdout_raw.parent != stderr_raw.parent or not stdout_raw.parent.is_dir():
        raise ValueError("Native log paths need an existing common directory")
    state_lock = Lock()
    progress: list[dict[str,float|int]] = []
    last_advanced_at: float | None = None
    highest_pct: int | None = None
    t0 = time.monotonic()
    flags: dict[str,Any] = {}
    if os.name == "nt":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = True
    proc = subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            bufsize=0,**flags)

    def drain(stream: Any, destination:Path, track:bool) -> None:
        nonlocal last_advanced_at, highest_pct
        carry = b""
        with destination.open("wb") as sink:
            while True:
                chunk=os.read(stream.fileno(),4096)
                if not chunk:
                    break
                sink.write(chunk)
                if not track:
                    continue
                joined=carry+chunk
                with state_lock:
                    for match in _PROGRESS.finditer(joined):
                        pct=int(match.group(1))
                        if 0 <= pct <= 100 and (highest_pct is None or pct>highest_pct):
                            highest_pct=pct
                            last_advanced_at=time.monotonic()
                            progress.append({"pct":pct,
                                "elapsed_s":round(last_advanced_at-t0,3)})
                carry=joined[-96:]
        stream.close()

    stdout_thread=Thread(target=drain,args=(proc.stdout,stdout_raw,False),daemon=True)
    stderr_thread=Thread(target=drain,args=(proc.stderr,stderr_raw,True),daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    reason=None
    try:
        while proc.poll() is None:
            now=time.monotonic()
            with state_lock:
                last=last_advanced_at
            if now-t0 >= hard_timeout_s:
                reason="hard_wall_timeout"
                break
            if last is None and now-t0 >= policy.startup_timeout_s:
                reason="startup_no_progress"
                break
            if last is not None and now-last >= policy.idle_progress_timeout_s:
                reason="progress_idle_timeout"
                break
            time.sleep(min(policy.poll_s,max(0.001,hard_timeout_s-(now-t0))))
        if reason is not None:
            _bounded_end(proc,policy.terminate_grace_s)
        else:
            proc.wait(timeout=policy.terminate_grace_s)
    finally:
        if proc.poll() is None:
            _bounded_end(proc,policy.terminate_grace_s)
        stdout_thread.join(timeout=policy.terminate_grace_s)
        stderr_thread.join(timeout=policy.terminate_grace_s)
    elapsed=round(time.monotonic()-t0,3)
    with state_lock:
        events=list(progress)
        last_progress=highest_pct
        last_progress_at=last_advanced_at
    return {
       "exit_code":None if reason else proc.returncode,
       "native_returncode":proc.returncode,
       "timed_out":bool(reason),
       "watchdog_termination_reason":reason,
       "elapsed_seconds":elapsed,
       "highest_progress_pct":last_progress,
       "observed_first_progress":bool(events),
       "progress_events":events,
       "progress_idle_at_exit_s":None if last_progress_at is None
                                  else round(time.monotonic()-last_progress_at,3),
       "stdout_raw_bytes":stdout_raw.stat().st_size if stdout_raw.is_file() else None,
       "stderr_raw_bytes":stderr_raw.stat().st_size if stderr_raw.is_file() else None,
       "exact_native_stall_location_known":False,
    }
