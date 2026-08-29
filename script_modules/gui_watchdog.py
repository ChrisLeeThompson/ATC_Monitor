"""
Main-thread responsiveness watchdog (v3.3.7).

The 2026-08-19 evening crashes were GUI hangs: the main thread stopped
processing posted events within ~10 ms of a Start click while the worker
thread stayed healthy, and Windows eventually terminated the ghosted app
(TerminateProcess) -- which leaves no faulthandler dump and no WER minidump.
The only instrument that can attribute this failure class is one that
captures the stack during the hang. That is this watchdog:

- The GUI thread beats a timestamp once per second (QTimer -> beat()).
- A daemon thread polls the timestamp; if the beat goes stale past the
  stall threshold it writes "WATCHDOG ... unresponsive" plus a full
  all-thread traceback (faulthandler.dump_traceback) through the same
  kept-open crash file faulthandler is armed on -- once per stall episode,
  re-arming if the GUI recovers.

Known blind spot: a main thread hung while holding the GIL also starves
this pure-Python watchdog thread. The observed incidents had the GIL
circulating (the logging thread kept writing), so this design captures
them; the GIL-holding class is covered separately by the
faulthandler.dump_traceback_later fallback re-armed from the GUI heartbeat
(atc_monitor._on_watchdog_heartbeat, since v3.3.12) -- it dumps from a C
thread that needs no GIL.

Suspend guard: time.monotonic() on Windows includes time spent suspended,
so a laptop sleep (or a whole-process freeze) would fake a stall. If the
poll loop itself skipped far more than one interval, the whole process was
frozen -- re-baseline and skip instead of dumping. The skip is logged: the
same gap signature also appears when this thread was starved by a
GIL-holding main-thread stall, and a silent re-baseline would hide that
episode entirely (the dump_traceback_later fallback captures its stack).

Every write path is guarded: the watchdog must never raise and never take
the app down.
"""
import faulthandler
import threading
import time

_SUSPEND_GUARD_FACTOR = 5


class GuiWatchdog:

    def __init__(self, fp, *, stall_after: float = 10.0,
                 poll_interval: float = 2.0,
                 clock=time.monotonic,
                 dump=faulthandler.dump_traceback):
        self._fp = fp
        self._stall_after = stall_after
        self._poll = poll_interval
        self._clock = clock
        self._dump = dump
        self._last_beat = clock()
        self._last_check = clock()
        self._stalled_since = None
        self._stop_evt = threading.Event()
        self._thread = None

    def beat(self):
        """GUI-thread heartbeat: one GIL-atomic store, no I/O, no locks."""
        self._last_beat = self._clock()

    def start(self):
        """Start the daemon poll thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="GuiWatchdog", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout: float = 3.0):
        """Stop the poll thread promptly (belt-and-braces; daemon exit is safe)."""
        self._stop_evt.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(join_timeout)

    def _run(self):
        while not self._stop_evt.wait(self._poll):
            self.check_once()

    def _write(self, text: str):
        # Guarded individually: a broken fp must not abort the episode's
        # remaining steps (the faulthandler dump writes via the raw fd and
        # may still succeed when the Python-level write cannot).
        try:
            if self._fp is not None:
                self._fp.write(text)
                self._fp.flush()
        except Exception:
            pass

    def check_once(self, now: float | None = None):
        """
        One poll of the stall state machine. Public so tests can drive it
        with a fake clock; must never raise (it runs unattended for the
        whole app lifetime).
        """
        try:
            if now is None:
                now = self._clock()

            # Suspend guard: if this loop itself skipped several intervals,
            # the whole process was frozen (system sleep) -- the GUI never
            # had a chance to beat. Re-baseline instead of false-alarming.
            gap = now - self._last_check
            self._last_check = now
            if gap > self._poll * _SUSPEND_GUARD_FACTOR:
                self._last_beat = now
                self._write(
                    f"WATCHDOG {time.strftime('%H:%M:%S')} poll gap "
                    f"{gap:.1f}s - re-baselined (system suspend, or this "
                    f"thread was starved by a GIL-holding stall)\n"
                )
                return

            age = now - self._last_beat
            if self._stalled_since is None:
                if age > self._stall_after:
                    self._stalled_since = self._last_beat
                    self._write(
                        f"WATCHDOG {time.strftime('%H:%M:%S')} main thread "
                        f"unresponsive for {age:.1f}s\n"
                    )
                    try:
                        self._dump(file=self._fp, all_threads=True)
                    except Exception:
                        pass
            elif age <= self._stall_after:
                stalled_for = self._last_beat - self._stalled_since
                self._stalled_since = None
                self._write(
                    f"WATCHDOG {time.strftime('%H:%M:%S')} main thread "
                    f"recovered after {stalled_for:.1f}s\n"
                )
        except Exception:
            pass
