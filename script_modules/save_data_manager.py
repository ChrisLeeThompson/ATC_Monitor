"""
Save Data Manager

Manages saving of monitoring run data when the "Save Data" checkbox is checked.

Data is saved under a single ``Saved_Data`` parent in the script root (so all
runs stay together and can be moved as one unit) with the following structure.
Runs created before v3.3.4 lived directly in the script root as
``Saved_Data_Run-{X}_...``; their numbers are still scanned so numbering
continues across the layout change.

    Saved_Data/Run-{X}_{YYYY-MM-DD}_{HH-MM-SS}/
    ├── run_metadata.json
    ├── plots/
    │   ├── pattern_1_mean_pixel.png
    │   ├── pattern_1_match_score.png
    │   ├── pattern_2_mean_pixel.png
    │   └── pattern_2_match_score.png
    ├── pattern_1/
    │   ├── metrics.csv
    │   ├── raw_images/
    │   │   ├── batch_001.png
    │   │   └── ...
    │   ├── grayscale_images/
    │   │   ├── batch_001.png
    │   │   └── ...
    │   ├── binary_images/
    │   │   ├── batch_001.png
    │   │   └── ...
    │   └── match_images/
    │       ├── batch_001.png
    │       └── ...
    └── pattern_2/
        ├── metrics.csv
        ├── raw_images/
        ├── grayscale_images/
        ├── binary_images/
        └── match_images/

Images:
    - Raw: Full uncropped RTM image (last frame of each batch)
    - Grayscale: Batch-filtered processed image (full size)
    - Binary: Thresholded image (full size)
    - Match: The exact image fed to calculate_match_score (crop-sized normalized
      top-hat foreground map, or -- when match-on-foreground is off -- the
      sign-split fixed-scale grayscale change map at twice the crop width:
      positive change in the left half, negative in the right). Persisted
      because the 2026-08-17 stuck-score defect had to be diagnosed by
      reconstructing these from raw frames.
    - All saved as 8-bit or 16-bit PNG (lossless)

Metrics CSV columns (see METRICS_FIELDNAMES — single source of truth):
    batch_number, image_number, raw_width, raw_height, crop_width,
    crop_height, mean_pixel_value, mean_pixel_slope, match_score,
    white_pixel_percentage, smoothed_foreground, running_peak, stall_latched

Run metadata JSON:
    Run info, microscope settings, UI parameters, processing parameters,
    criteria enabled states, per-pattern details and completion times.
"""

import csv
import os
import json
import logging
import datetime
import numpy as np
from pathlib import Path
from PIL import Image


logger = logging.getLogger(__name__)


# Canonical metrics-CSV schema. Single source of truth for both the per-batch
# row builder (save_batch_metrics) and the CSV writer (save_metrics_csv) so the
# two cannot drift out of sync (a mismatch makes DictWriter raise at end of run).
METRICS_FIELDNAMES = [
    'batch_number',
    'image_number',
    'raw_width',
    'raw_height',
    'crop_width',
    'crop_height',
    'mean_pixel_value',
    'mean_pixel_slope',
    'match_score',
    'white_pixel_percentage',
    # Stall-latch trace (v3.3.4): persisted so field replays of the latch need
    # no reconstruction from images. Empty cells before the window fills.
    'smoothed_foreground',
    'running_peak',
    'stall_latched',
]


class SaveDataManager:
    """
    Manages saving of monitoring run data to disk.
    
    Created once per monitoring session when Save Data is enabled.
    The run directory is created immediately on initialization.
    """

    def __init__(self, script_root: str | Path):
        """
        Initialize the save data manager and create the run directory.
        
        :param script_root: Root directory of the script (parent for saved data)
        """
        self._script_root = Path(script_root)
        self._run_start_time = datetime.datetime.now()
        self._run_dir = self._create_run_directory()
        self._plots_dir = self._run_dir / "plots"
        self._plots_dir.mkdir(exist_ok=True)
        
        # Per-pattern directories and batch counters
        self._pattern_dirs: dict[int, dict[str, Path]] = {}
        self._batch_counters: dict[int, int] = {}
        
        # Per-pattern metrics accumulation
        self._metrics: dict[int, list[dict]] = {}
        
        logger.info(f"SaveDataManager initialized: {self._run_dir}")
    
    # ===========================
    # Directory Management
    # ===========================
    
    def _create_run_directory(self) -> Path:
        """
        Create a uniquely numbered run directory with timestamp under the
        ``Saved_Data`` parent.

        Scans existing directories to determine the next run number. Legacy
        pre-v3.3.4 runs (``Saved_Data_Run-{X}_...`` directly in the script
        root, or moved into the parent by hand) are scanned too, so numbering
        continues across the layout change instead of resetting to 1.
        Format: Saved_Data/Run-{X}_{YYYY-MM-DD}_{HH-MM-SS}

        :return: Path to the created run directory
        """
        parent = self._script_root / "Saved_Data"
        parent.mkdir(parents=True, exist_ok=True)
        prefix = "Run-"
        legacy_prefix = "Saved_Data_Run-"

        def _scan_run_numbers(root: Path, pfx: str) -> list[int]:
            found = []
            if root.exists():
                for item in root.iterdir():
                    if item.is_dir() and item.name.startswith(pfx):
                        # Extract run number (between the prefix and next "_")
                        try:
                            after_prefix = item.name[len(pfx):]
                            found.append(int(after_prefix.split("_")[0]))
                        except (ValueError, IndexError):
                            continue
            return found

        # Note: "Saved_Data_Run-" names do not start with "Run-", so the two
        # parent scans never double-count a directory.
        existing_runs = (
            _scan_run_numbers(parent, prefix)
            + _scan_run_numbers(parent, legacy_prefix)
            + _scan_run_numbers(self._script_root, legacy_prefix)
        )

        next_run = max(existing_runs, default=0) + 1
        timestamp = self._run_start_time.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the directory idempotently: if a leftover dir or a concurrent
        # process already claimed this run number, increment and retry rather
        # than crashing (which would silently disable saving for the session).
        while True:
            run_dir_name = f"{prefix}{next_run}_{timestamp}"
            run_dir = parent / run_dir_name
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                next_run += 1

        logger.info(f"Created run directory: Saved_Data/{run_dir_name}")
        return run_dir
    
    def initialize_pattern(self, pattern_idx: int):
        """
        Create subdirectories for a pattern.
        
        Called when a pattern is first detected/validated.
        
        :param pattern_idx: Pattern index (0 or 1)
        """
        pattern_name = f"pattern_{pattern_idx + 1}"
        pattern_dir = self._run_dir / pattern_name
        
        dirs = {
            'root': pattern_dir,
            'raw': pattern_dir / "raw_images",
            'grayscale': pattern_dir / "grayscale_images",
            'binary': pattern_dir / "binary_images",
            'match': pattern_dir / "match_images",
        }
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        self._pattern_dirs[pattern_idx] = dirs
        self._batch_counters[pattern_idx] = 0
        self._metrics[pattern_idx] = []
        
        logger.info(f"Pattern {pattern_idx + 1} directories created: {pattern_dir}")
    
    # ===========================
    # Per-Batch Saving
    # ===========================
    
    def save_batch_images(
        self,
        pattern_idx: int,
        raw_image: np.ndarray,
        grayscale_image: np.ndarray | None = None,
        binary_image: np.ndarray | None = None,
        match_image: np.ndarray | None = None,
        batch_number: int | None = None
    ):
        """
        Save images from a completed batch.

        :param pattern_idx: Pattern index (0 or 1)
        :param raw_image: Full uncropped raw RTM image (last frame of batch)
        :param grayscale_image: Batch-filtered processed image (full size), or None
        :param binary_image: Thresholded binary image (full size), or None
        :param match_image: The crop-sized image fed to calculate_match_score
            this batch (float [0, 1]), or None
        :param batch_number: The worker's authoritative batch number
            (pattern_state.batch_count). Passing it keeps filenames in lockstep
            with the worker even when a save is skipped -- the internal counter
            fallback desyncs permanently on any skipped batch, which silently
            no-ops the criteria-met rename.
        """
        if pattern_idx not in self._pattern_dirs:
            logger.warning(
                f"Pattern {pattern_idx + 1} not initialized - skipping image save"
            )
            return

        if batch_number is not None:
            self._batch_counters[pattern_idx] = int(batch_number)
        else:
            self._batch_counters[pattern_idx] += 1
        batch_num = self._batch_counters[pattern_idx]
        filename = f"batch_{batch_num:03d}.png"
        dirs = self._pattern_dirs[pattern_idx]
        
        # Save raw image (always available)
        self._save_image(raw_image, dirs['raw'] / filename)
        
        # Save grayscale image (available after filtering)
        if grayscale_image is not None:
            self._save_image(grayscale_image, dirs['grayscale'] / filename)
        
        # Save binary image (available after delay disengages)
        if binary_image is not None:
            self._save_image(binary_image, dirs['binary'] / filename)

        # Save the match-scorer input (crop-sized; float x255 -> uint8 is
        # lossless for the fixed-scale counts/255 representation)
        if match_image is not None:
            self._save_image(match_image, dirs['match'] / filename)

        logger.debug(
            f"Pattern {pattern_idx + 1} batch {batch_num} images saved"
        )

    def mark_batch_criteria_met(self, pattern_idx: int, batch_num: int):
        """
        Tag a batch's saved images as the criteria-met batch by renaming them
        from ``batch_NNN.png`` to ``batch_NNN_criteria_met.png``.

        Renames the file in each image directory (raw/grayscale/binary/match)
        that exists; missing files are skipped (e.g. binary is not saved during
        the delay phase). Safe to call once per pattern at completion.

        :param pattern_idx: Pattern index (0 or 1)
        :param batch_num: Batch number on which criteria were met (matches the
            per-pattern counter used to name the PNGs in save_batch_images)
        """
        if pattern_idx not in self._pattern_dirs:
            logger.warning(
                f"Pattern {pattern_idx + 1} not initialized - "
                f"skipping criteria-met rename"
            )
            return

        src_name = f"batch_{batch_num:03d}.png"
        dst_name = f"batch_{batch_num:03d}_criteria_met.png"
        dirs = self._pattern_dirs[pattern_idx]

        renamed = 0
        for key in ('raw', 'grayscale', 'binary', 'match'):
            src = dirs[key] / src_name
            if src.exists():
                os.replace(src, dirs[key] / dst_name)
                renamed += 1

        if renamed == 0:
            logger.warning(
                f"Pattern {pattern_idx + 1} batch {batch_num}: no saved images "
                f"found to tag as criteria-met (batch numbering out of sync?)"
            )
        else:
            logger.info(
                f"Pattern {pattern_idx + 1} batch {batch_num} tagged as "
                f"criteria-met ({renamed} images)"
            )
    
    def save_batch_metrics(
        self,
        pattern_idx: int,
        image_number: int,
        raw_image_size: tuple[int, int],
        crop_image_size: tuple[int, int],
        mean_pixel_value: float,
        mean_pixel_slope: float,
        match_score: float | None,
        white_pixel_percentage: float,
        batch_number: int | None = None,
        smoothed_foreground: float | None = None,
        running_peak: float | None = None,
        stall_latched: bool = False
    ):
        """
        Accumulate metrics for a completed batch.

        Metrics are stored in memory and written to CSV at end of run.

        :param pattern_idx: Pattern index (0 or 1)
        :param image_number: Global image counter value
        :param raw_image_size: (width, height) of the full processed image
        :param crop_image_size: (width, height) of the cropped analysis image
        :param mean_pixel_value: Mean pixel value of cropped filtered image
        :param mean_pixel_slope: Current slope of mean pixel values
        :param match_score: Current pattern match score, or None when unavailable
            this batch (written as an empty CSV cell)
        :param white_pixel_percentage: Current white pixel percentage
        :param batch_number: The worker's authoritative batch number
            (pattern_state.batch_count); metrics rows continue after pattern
            completion while image saves stop, so this must not lean on the
            image-save counter
        :param smoothed_foreground: Stall latch's median-smoothed foreground
            value this round (None -> empty cell)
        :param running_peak: Stall latch's running peak of the smoothed trace
            (None -> empty cell)
        :param stall_latched: Whether the stall latch condition held this round
        """
        if pattern_idx not in self._metrics:
            logger.warning(
                f"Pattern {pattern_idx + 1} not initialized - skipping metrics save"
            )
            return

        # Deliberately does NOT touch _batch_counters: metrics rows continue after
        # a pattern completes while image saves stop, and _batch_counters must keep
        # meaning "last SAVED image batch" so build_pattern_detail's total_batches
        # matches the PNGs actually on disk.
        if batch_number is not None:
            batch_num = int(batch_number)
        else:
            batch_num = self._batch_counters.get(pattern_idx, 0)

        # Keys must match METRICS_FIELDNAMES exactly (the CSV writer enforces it).
        self._metrics[pattern_idx].append({
            'batch_number': batch_num,
            'image_number': image_number,
            'raw_width': raw_image_size[0],
            'raw_height': raw_image_size[1],
            'crop_width': crop_image_size[0],
            'crop_height': crop_image_size[1],
            'mean_pixel_value': mean_pixel_value,
            'mean_pixel_slope': mean_pixel_slope,
            'match_score': '' if match_score is None else match_score,
            'white_pixel_percentage': white_pixel_percentage,
            'smoothed_foreground': ('' if smoothed_foreground is None
                                    else smoothed_foreground),
            'running_peak': '' if running_peak is None else running_peak,
            'stall_latched': int(stall_latched),
        })
    
    # ===========================
    # End-of-Run Saving
    # ===========================
    
    def save_metrics_csv(self, pattern_idx: int):
        """
        Write accumulated metrics to CSV for a pattern.
        
        :param pattern_idx: Pattern index (0 or 1)
        """
        if pattern_idx not in self._metrics or not self._metrics[pattern_idx]:
            logger.info(f"Pattern {pattern_idx + 1}: No metrics to save")
            return
        
        csv_path = self._pattern_dirs[pattern_idx]['root'] / "metrics.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._metrics[pattern_idx])
        
        logger.info(
            f"Pattern {pattern_idx + 1} metrics saved: "
            f"{len(self._metrics[pattern_idx])} rows -> {csv_path.name}"
        )
    
    def save_run_metadata(self, metadata: dict):
        """
        Save run metadata as JSON.
        
        The caller is responsible for assembling the metadata dictionary.
        This method adds the run end timestamp and writes to disk.
        
        :param metadata: Complete metadata dictionary
        """
        # Add end timestamp
        metadata.setdefault('run_info', {})
        metadata['run_info']['timestamp_end'] = (
            datetime.datetime.now().isoformat(timespec='seconds')
        )
        
        json_path = self._run_dir / "run_metadata.json"

        # Write atomically: serialize to a temp file in the same directory,
        # flush+fsync, then os.replace() (atomic on the same filesystem) so a
        # crash mid-write can never leave a truncated/corrupt metadata file.
        tmp_path = json_path.with_name(json_path.name + ".tmp")
        with open(tmp_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, json_path)

        logger.info(f"Run metadata saved: {json_path.name}")
    
    def save_plot_image(self, figure, filename: str):
        """
        Save a matplotlib figure as PNG to the plots directory.
        
        :param figure: matplotlib Figure object
        :param filename: Filename (e.g., 'pattern_1_mean_pixel.png')
        """
        plot_path = self._plots_dir / filename
        figure.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        logger.info(f"Plot saved: {filename}")
    
    def finalize(self, metadata: dict):
        """
        Finalize the run — save all accumulated data.
        
        Call this at the end of a monitoring session. Writes:
        1. Metrics CSV for each pattern
        2. Run metadata JSON
        
        Plot images should be saved separately via save_plot_image()
        before calling finalize.
        
        :param metadata: Complete metadata dictionary
        """
        # Save metrics for each initialized pattern
        for pattern_idx in sorted(self._metrics.keys()):
            self.save_metrics_csv(pattern_idx)
        
        # Save run metadata
        self.save_run_metadata(metadata)
        
        logger.info(f"Run data finalized: {self._run_dir.name}")
    
    # ===========================
    # Metadata Assembly Helpers
    # ===========================
    
    def build_run_metadata(
        self,
        microscope_data: dict | None = None,
        ui_parameters: dict | None = None,
        processing_parameters: dict | None = None,
        criteria_enabled: dict | None = None,
        pattern_details: list[dict] | None = None,
        completion_status: str = "",
        stop_reason: str = "",
        cb_calibration: dict | None = None,
    ) -> dict:
        """
        Assemble the run metadata dictionary.

        :param microscope_data: Microscope settings (ion species, voltage, etc.)
            NOTE: detector contrast/brightness in this block are read BEFORE the
            auto-CB calibration runs; the post-calibration values live in the
            cb_calibration block.
        :param ui_parameters: UI parameter values
        :param processing_parameters: Internal processing parameters
        :param criteria_enabled: Criteria checkbox states
        :param pattern_details: List of per-pattern detail dicts
        :param completion_status: Why the run ended at the run level, e.g.
            "CRITERIA_MET" | "USER_STOPPED" | "RTM_FAILURE" | "ERROR".
        :param stop_reason: Human-readable reason the run ended.
        :param cb_calibration: Auto-CB calibration outcome (status, measurements,
            start/final contrast+brightness), or None when it did not run.
        :return: Complete metadata dictionary ready for save_run_metadata()
        """
        # Extract run number from directory name
        run_number = self._extract_run_number()

        metadata = {
            "run_info": {
                "timestamp_start": self._run_start_time.isoformat(timespec='seconds'),
                "timestamp_end": None,  # Filled by save_run_metadata
                "run_number": run_number,
                "run_directory": self._run_dir.name,
                "completion_status": completion_status,
                "stop_reason": stop_reason,
            },
            "microscope": microscope_data or {},
            "ui_parameters": ui_parameters or {},
            "criteria_enabled": criteria_enabled or {},
            "processing_parameters": processing_parameters or {},
            "cb_calibration": cb_calibration,
            "patterns": pattern_details or [],
        }

        return metadata

    def save_cb_trace_csv(self, rows: list[dict]):
        """
        Write the auto-CB per-measurement trace beside run_metadata.json.

        One row per calibration measurement (n, tag, frames, contrast,
        brightness, median, span, wclip, bclip, cost). No-op when empty.
        """
        if not rows:
            return
        fieldnames = ["n", "tag", "frames", "contrast", "brightness",
                      "median", "span", "wclip", "bclip", "cost"]
        path = self._run_dir / "cb_trace.csv"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"CB calibration trace saved: {path.name} ({len(rows)} rows)")
        except OSError:
            logger.warning("Could not save CB calibration trace", exc_info=True)
    
    def build_pattern_detail(
        self,
        pattern_idx: int,
        pattern_id: str = "",
        pattern_type: str = "",
        width_um: float = 0.0,
        height_um: float = 0.0,
        aspect_ratio: float = 0.0,
        crop_rect: tuple | list | None = None,
        monitoring_start: datetime.datetime | None = None,
        completion_time: datetime.datetime | None = None,
        duration: str = "",
        criteria_met_batch: int | None = None,
        completion_status: str = "",
        crop_rect_history: list | None = None,
        completed_via_stall: bool = False,
    ) -> dict:
        """
        Build a detail dictionary for a single pattern.

        :param pattern_idx: Pattern index (0 or 1)
        :param criteria_met_batch: Batch number at which all criteria were met,
            or None if the pattern never completed. The metric values for that
            batch live in the per-batch metrics CSV, keyed by this number.
        :param completion_status: Whether THIS pattern completed when the run
            ended, e.g. "CRITERIA_MET" | "INCOMPLETE" (independent of the
            run-level status -- one pattern can complete before a user stop).
        :param crop_rect_history: List of {"batch": N, "rect": [x, y, w, h]}
            entries recording every crop rect the pattern ran under ("crop_rect"
            alone silently loses operator changes made mid-run).
        :return: Pattern detail dictionary for inclusion in metadata
        """
        # total_batches = last SAVED image batch (image saves stop at pattern
        # completion, so for a completed pattern this equals criteria_met_batch and
        # always matches the PNG files on disk; post-completion metrics rows are
        # not counted here -- the metrics CSV carries its own batch numbers).
        total_batches = self._batch_counters.get(pattern_idx, 0)

        return {
            "pattern_index": pattern_idx + 1,
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "width_um": width_um,
            "height_um": height_um,
            "aspect_ratio": aspect_ratio,
            "crop_rect": list(crop_rect) if crop_rect else None,
            "crop_rect_history": list(crop_rect_history) if crop_rect_history else [],
            "monitoring_start": (
                monitoring_start.isoformat(timespec='seconds')
                if monitoring_start else None
            ),
            "completion_time": (
                completion_time.isoformat(timespec='seconds')
                if completion_time else None
            ),
            "duration": duration,
            "total_batches": total_batches,
            "criteria_met_batch": criteria_met_batch,
            "completion_status": completion_status,
            # True when the foreground criterion passed via the grid-bar stall
            # latch on the completing round (the absolute threshold never
            # cleared) -- flags the run for operator review of the final image.
            "completed_via_stall": completed_via_stall,
        }
    
    # ===========================
    # Internal Helpers
    # ===========================
    
    @staticmethod
    def _save_image(image: np.ndarray, path: Path):
        """
        Save a numpy array as a PNG image.
        
        Handles conversion from various numpy dtypes to PIL-compatible formats:
        - bool: Converted to uint8 (0/255)
        - float (e.g. from filter_images): Scaled from [0.0, 1.0] to uint8 [0, 255]
        - int64 (e.g. raw RTM data): Clipped to [0, 255] and cast to uint8
        - uint8/uint16: Passed through directly
        
        :param image: 2D numpy array
        :param path: Output file path
        """
        if image.dtype == bool:
            image = (image.astype(np.uint8)) * 255
        elif np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8 and image.dtype != np.uint16:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(path, format='PNG')
    
    def _extract_run_number(self) -> int:
        """Extract the run number from the directory name."""
        name = self._run_dir.name
        try:
            # Format: Run-{X}_{YYYY-MM-DD}_{HH-MM-SS}
            # (legacy pre-v3.3.4: Saved_Data_Run-{X}_... -- both split on "Run-")
            after_prefix = name.split("Run-")[1]
            return int(after_prefix.split("_")[0])
        except (IndexError, ValueError):
            return 0
    
    @property
    def run_directory(self) -> Path:
        """Get the run directory path."""
        return self._run_dir
    
    @property
    def plots_directory(self) -> Path:
        """Get the plots directory path."""
        return self._plots_dir