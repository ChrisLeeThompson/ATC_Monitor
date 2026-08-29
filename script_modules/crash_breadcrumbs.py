"""
Synchronous run-lifecycle breadcrumbs for the crash record (v3.3.6).

The app log is queued (QueueHandler -> QueueListener), so the final seconds of
records die with the process on a hard crash -- the 2026-08-19 on-tool crash
left a 1-37 s blind spot between "Patterning started" and process death. These
breadcrumbs write one short line per coarse run-lifecycle milestone straight
through the same kept-open crash file faulthandler uses, so even a crash that
loses every queued record still bounds the death to a phase.

Rules of the road:
- drop() must never raise and must stay cheap: one write + flush, guarded by a
  blanket try/except. It is called from both the GUI and the worker thread; a
  single buffered .write of one small string is GIL-atomic enough for this
  diagnostic duty (faulthandler itself writes via the raw fd independently).
- No-op until atc_monitor.setup_logging wires the fp via set_fp().
- Breadcrumb lines ("BC hh:mm:ss ...") count as non-banner content for the
  launch-time fault-log sweep, so they auto-mirror to the script directory.
"""
import time

_fp = None


def set_fp(fp):
    """Wire (or clear, with None) the crash-record file object."""
    global _fp
    _fp = fp


def drop(text: str):
    """Write one breadcrumb line to the crash record; silent no-op on any failure."""
    try:
        if _fp is not None:
            _fp.write(f"BC {time.strftime('%H:%M:%S')} {text}\n")
            _fp.flush()
    except Exception:
        pass
