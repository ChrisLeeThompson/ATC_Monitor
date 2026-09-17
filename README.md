# ATC Monitor

> [!NOTE]
> **Full documentation:** https://chrisleethompson.github.io/scripts/atc_monitor/

A PySide6 desktop utility that reads a Thermo Scientific FIB-SEM's real-time monitor (RTM) image stream through the Thermo Scientific AutoScript SDK, tracks how each milling pattern's area is changing, and stops patterning automatically once every pattern has met its completion criteria. It runs unattended alongside Thermo Scientific AutoTEM Cryo (ATC) during the Rough Milling activity, and can also monitor Rectangle patterns used for manual milling.

This script is experimental. Many of the features are still being explored and tested.

## Documentation

Full documentation: https://chrisleethompson.github.io/scripts/atc_monitor/

## Features

- **Automatic stop** of FIB patterning when one or two Rectangle patterns have completed, moving ATC on to its next activity.
- **Three completion criteria** per pattern, each independently switchable: mean-slope flattening, match-score stability, and a foreground (percent pixels or foreground energy) threshold.
- **Confirmation rounds** so all enabled criteria must pass together for a set number of consecutive batches before a pattern is called complete.
- **Stall latch** for patterns that mill into a grid bar, where the foreground metric never reaches its threshold.
- **Automatic contrast/brightness calibration** of the detector at the start of a run, with the learned response cached per microscope.
- **Save Data** option that writes raw RTM images, processed images, plots, per-batch metrics, and run metadata for later analysis or model training.

## Requirements

- Python 3.11+
- PySide6 6.7.1 (pinned; the worker teardown path was validated against this release)
- shiboken6 6.7.1
- NumPy 2.2.5+, OpenCV 4.8.1+ (`opencv-python`), Pillow 10.1+, scikit-image 0.25.2+, Matplotlib 3.8.1+
- Thermo Scientific AutoScript 4.14+ (required; the script cannot run without it)

All packages above ship with the AutoScript 4.14 Python environment, where the script is developed and tested.

## Installation

1. Download the latest release ZIP from the [Releases page](https://github.com/ChrisLeeThompson/atc_monitor/releases).
2. Extract it and copy the script folder to your desired location. Scripts that control a microscope are best installed on the Support PC (SPC) or Microscope PC (MPC).
3. Run the script with the AutoScript Python environment; no packages need to be installed. Do not pip-install `requirements.txt` into the AutoScript environment; it already provides these packages.

## Running

Run the main module from the script folder:

```
python atc_monitor.py
```

The script also runs from the AutoScript Python interpreter or AutoScript Runner. Click Start in the UI to connect to the AutoScript server and begin monitoring.

## Notes

- Only Rectangle patterns are monitored (Regular and Cleaning Cross-section patterns are rejected), and at most two patterns at a time. Patterns narrower than the aspect-ratio threshold, such as stress-relief cuts, are ignored.
- Thresholds and crop rectangles can be adjusted while monitoring is active; changes apply to the next batch of RTM images.
- The script writes `logs/atc_monitor.log` (rotating), `logs/faulthandler.log`, and `logs/cb_plant.json` in the script folder, and a `Saved_Data/Run-N_<timestamp>/` folder per run when Save Data is checked. If the script folder is not writable, logs fall back to `%LOCALAPPDATA%\ATC_Monitor\logs\`. Only `Saved_Data` grows without limit.
- The first line of `logs/atc_monitor.log` records the running version (for example `LAUNCH app=3.4.4`). Check it after updating.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Christopher Thompson.

The Catbug artwork in `script_assets/` is not covered by the MIT license; see [LICENSE](LICENSE). PySide6 (Qt for Python) is licensed under the LGPLv3 and is used as an unmodified runtime dependency installed from PyPI; it is not distributed with this source.

## Contact

Developed by Chris Thompson with assistance from Anthropic's Claude. Questions and suggestions are welcome: [@ChrisLeeThompson](https://github.com/ChrisLeeThompson) on GitHub.
