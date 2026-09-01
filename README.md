# ATC Monitor

ATC Monitor reads the microscope's real-time monitor (RTM) image stream through the Thermo Scientific
AutoScript SDK, tracks how each pattern's milled area is changing, and stops patterning
automatically once every pattern has met its completion criteria for a set number of consecutive
confirmation rounds. It is designed to run unattended alongside AutoTEM Cryo.

## Requirements

- Python 3.11
- Thermo Scientific AutoScript 4.13 or later, on a machine that can reach the microscope. This is
  a proprietary package and is not installed from PyPI; its Enthought environment already bundles
  every dependency listed in `requirements.txt`.
- The Python packages in `requirements.txt`. PySide6 is pinned to 6.7.1, which is the validated
  version; Qt object-lifetime behavior is version-sensitive here.

## Running

```
python atc_monitor.py
```

Select the FIB quadrant in Microscope Control first, then click Start. The application connects to
the microscope, waits quietly until the ion beam is on and patterning begins, and starts analyzing
once it sees an active patterning session.

## How completion is decided

Each pattern is cropped from the RTM image and measured over a rolling batch of frames. Three
criteria are evaluated, and each can be enabled or disabled independently from the results panel:

- **Mean Slope** — the rate of change of the mean pixel value has flattened.
- **Match Score** — consecutive images have stopped changing structurally.
- **Percent Pixels** (or **Foreground Energy**, depending on the binarization method) — the
  foreground metric has fallen to or below the Maximum Pixels threshold.

All enabled criteria must pass together for a set number of consecutive confirmation rounds
(three by default) before a pattern is called complete. When every pattern is complete, the
application stops patterning and saves its results.

The foreground criterion also has a stall latch, selected by the Foreground Completion Mode
setting. Some samples expose static material, such as a grid bar, that holds the foreground metric
above any threshold you could have chosen in advance; milling then runs for a long time. In
"Absolute + stall latch" mode the criterion also completes when the foreground trace has provably
floored — dropped well below its running peak and stayed flat for a sustained window. Completions
that relied on the latch are labeled `(stall)` in the results panel and recorded as
`completed_via_stall` in the run metadata. Choosing "Absolute (threshold)" turns the latch off entirely.

## Contrast and brightness calibration

With Auto CB enabled, the application calibrates detector contrast and brightness once at the start
of a run, aiming for a target median brightness and dynamic range while keeping clipping inside the
limits you set. The measured response of each detector is cached in `logs\cb_plant.json` and reused
to speed up later runs; the file is created automatically, is keyed by system name so several
microscopes can share one deployment, and can be deleted safely.

## Files the application writes

| Path | Contents | Growth |
| --- | --- | --- |
| `logs/atc_monitor.log` | Main application log, one line per event. | Rotating, 5 MB × 5 files. |
| `logs/faulthandler.log` | Crash and diagnostic record, copied here at startup from the local per-user copy. | Rotates to `.1` at 5 MB. |
| `%LOCALAPPDATA%\ATC_Monitor\logs\faulthandler.log` | The live crash record. Always on local disk so a crash is never lost to a network share. | Swept and reset at each launch. |
| `Saved_Data\Run-N_<timestamp>\` | Images, per-batch metrics, plots, and run metadata. Written only when Save Data is checked. | One directory per run; delete when no longer needed. |
| `logs\cb_plant.json` | Learned detector response, keyed by system name. | A few hundred bytes. |

If the script directory is not writable, the application falls back to `%LOCALAPPDATA%` for its
logs and reports where they went in the log's first lines. Log retention is automatic; only
`Saved_Data` grows without limit, because those are your results.

The first line of `logs/atc_monitor.log` records the running version, for example
`LAUNCH app=3.3.9`. Check it after deploying to confirm the build you intended is the one running.

## Supported patterns

One or two rectangle patterns per session. Regular and cleaning cross sections are not supported.
Patterns narrower than the aspect-ratio threshold are ignored, which filters out stress-relief cuts.

## License

The source code is MIT licensed. See [LICENSE](LICENSE).

The images in `script_assets/` are third-party artwork and are **not** covered
by that license. They are included for use within this application only and are
not licensed for redistribution. Replace them with your own artwork before
redistributing this project.
