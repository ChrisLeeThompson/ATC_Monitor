"""
Workflow Worker Module

This module orchestrates the complete FIB patterning monitoring workflow in a
separate thread to avoid blocking the GUI.

Key responsibilities:
- Microscope connection and validation
- Continuous monitoring (runs until Stop button)
- RTM data acquisition loop
- Image processing pipeline
- Pattern matching and evaluation
- GUI updates via signals
- Graceful shutdown handling
"""
import json
import logging
import os
import time
import math
import datetime
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QCoreApplication
from dataclasses import dataclass, field
from script_modules import crash_breadcrumbs
# Qt-free by design (see app_styles.StatusText): shared status-bar wording
# for microscope states reported from more than one site.
from script_modules.app_styles import StatusText
from script_modules.image_pattern_matching import calculate_match_score
from script_modules.atc_monitor_parameters import (
    SlopeMethod, WindowMode, BinarizationMethod, ForegroundCompletionMode
)

# Import processing modules
from script_modules.fib_patterning_monitor import (
    connect_to_microscope,
    disconnect_from_microscope,
    validate_active_device,
    validate_patterns,
    setup_rtm_monitoring,
    acquire_rtm_data,
    check_patterning_state,
    check_fib_beam_on,
    stop_patterning,
    restart_rtm,
    get_system_name,
    get_specimen_current,
    read_detector_cb,
    set_detector_cb,
    read_detector_cb_limits,
    read_scanning_bit_depth,
    RTM_FRESH_DATA,
    RTM_NO_NEW_DATA
)
from script_modules.rtm_data_processing import (
    get_images_from_rtm_data,
    precompute_pattern_metadata,
    compute_cb_balance,
    cb_edge_cost,
    cb_clip_actionable,
    cb_clips_actionable,
    cb_rail_dominated,
    cb_rail_dominated_clips,
    CBEdgeConfig,
    CB_EDGE_TOL_DEFAULT,
    run_cb_correction
)
from script_modules.image_processing import (
    filter_images, threshold_adaptive, calculate_otsu_threshold_values,
    frozen_threshold_for_boundary, blend_images, signed_to_match_image,
    white_tophat_map, tophat_energy, tophat_display, tophat_normalized,
)
from script_modules.image_cropping import crop_to_rect
from script_modules.image_analysis import (
    calculate_mean_pixel_value,
    calculate_slope_of_mean_pixel_values,
    calculate_slope_linear_regression,
    calculate_num_white_pixels
)
from script_modules.results_evaluation import (
    EvaluationState,
    EvaluationCriteria,
    StallConfig,
    apply_stall_carry,
    evaluate_foreground_stall,
    evaluate_pattern_completion,
    check_all_patterns_complete
)
from script_modules.image_pattern_matching import estimate_interframe_gain
from script_modules.worker_parameter_manager import (
    WorkerParameterManager,
    display_name,
    validate_parameter_update
)
from script_modules.atc_monitor_parameters import RTMMode
from script_modules.save_data_manager import SaveDataManager
from pathlib import Path


logger = logging.getLogger(__name__)


# ===========================
# run_metadata.json status values
# ===========================
# Run-level: why the whole run ended (run_info.completion_status).
RUN_STATUS_CRITERIA_MET = "CRITERIA_MET"   # all patterns met criteria (auto-stop)
RUN_STATUS_USER_STOPPED = "USER_STOPPED"   # user clicked Stop (or closed the window)
RUN_STATUS_RTM_FAILURE = "RTM_FAILURE"     # ended by the consecutive-RTM-failure limit
RUN_STATUS_ERROR = "ERROR"                 # ended by an unhandled exception in run()
RUN_STATUS_PATTERNING_STOPPED = "PATTERNING_STOPPED"  # patterning ended on-tool before criteria (operator abort at the microscope)
RUN_STATUS_BEAM_OFF = "BEAM_OFF"           # FIB beam turned off mid-session
# Pattern-level: whether an individual pattern completed (patterns[].completion_status).
PATTERN_STATUS_CRITERIA_MET = "CRITERIA_MET"
PATTERN_STATUS_INCOMPLETE = "INCOMPLETE"


# ===========================
# Data Structures
# ===========================

@dataclass
class WorkerParameters:
    """Parameters for the workflow worker."""
    # Microscope connection
    microscope_host: str = "localhost"

    # Processing parameters
    number_of_images: int = 10
    gaussian_sigma: float = 1.0
    apply_dilation: bool = True
    threshold_num_classes: int = 2

    # Pattern 1 parameters
    pattern_1_crop_rect: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    pattern_1_mean_slope_threshold: float = 0.75
    pattern_1_match_score_threshold: float = 0.075
    pattern_1_max_pixels_threshold: float = 4.5

    # Pattern 2 parameters (if applicable)
    pattern_2_crop_rect: tuple[int, int, int, int] | None = None
    pattern_2_mean_slope_threshold: float = 0.75
    pattern_2_match_score_threshold: float = 0.075
    pattern_2_max_pixels_threshold: float = 4.5

    # Monitoring parameters
    confirmation_rounds: int = 3
    acquisition_delay_seconds: float = 1.0
    aspect_ratio_threshold: float = 0.3
    percent_difference_threshold: float = 20.0

    # Slope calculation parameters
    slope_method: SlopeMethod = SlopeMethod.GRADIENT
    binarization_method: BinarizationMethod = BinarizationMethod.FROZEN_MID
    tophat_radius: int = 5
    match_on_foreground: bool = True  # match score on the normalized top-hat foreground map instead of the grayscale change map (True = the field-validated path; kept in sync with ProcessingParameters)
    foreground_completion_mode: ForegroundCompletionMode = ForegroundCompletionMode.ABSOLUTE_PLUS_STALL  # foreground criterion: plain ABSOLUTE, or ABSOLUTE plus the grid-bar stall latch
    stall_window: int = 16  # stall latch: consecutive rounds the smoothed trace must stay flat
    stall_drop_fraction: float = 0.35  # stall latch: value must be <= this fraction of the running peak
    stall_rel_tolerance: float = 0.10  # stall latch: relative flatness tolerance
    stall_abs_tolerance: float = 0.15  # stall latch: absolute flatness tolerance (foreground units)
    num_points_for_slope: int = 4
    linear_regression_fit_points: int = 3

    # Pattern matching parameters (defaults kept in sync with
    # ProcessingParameters -- 3 is the on-tool-validated tiling)
    min_pattern_splits: int = 3
    target_tile_size: int = 100

    # RTM settings
    rtm_mode: RTMMode = RTMMode.LOW_RESOLUTION

    # Window mode
    window_mode: WindowMode = WindowMode.MANUAL
    analysis_interval_seconds: int = 4
    calibration_sample_count: int = 10
    min_window_size: int = 3
    max_window_size: int = 100

    # Criteria enabled flags (from checkboxes - global, not per-pattern)
    mean_slope_enabled: bool = True
    match_score_enabled: bool = True
    percent_pixels_enabled: bool = True

    # Save data settings
    save_data: bool = False
    script_root: str = ""

    # Auto contrast/brightness calibration (one-shot at patterning start, then held static)
    auto_cb_on_start: bool = True
    cb_recalibrate_every_session: bool = False
    cb_white_level: float = 255.0
    cb_lower_margin: float = 0.15
    cb_upper_margin: float = 0.20
    cb_max_white_clip_fraction: float = 0.02
    cb_max_black_clip_fraction: float = 0.05
    cb_min_bound: float = 0.0
    cb_max_bound: float = 1.0
    cb_max_iterations: int = 12
    cb_settle_seconds: float = 0.2
    cb_frames_per_measurement: int = 2


@dataclass
class PatternState:
    """State tracking for a single pattern during monitoring."""
    pattern_index: int  # 0 or 1
    pattern_id: str
    crop_rect: tuple[int, int, int, int] | None

    # Thresholds
    mean_slope_threshold: float
    match_score_threshold: float
    max_pixels_threshold: float

    # Data storage for metrics
    mean_pixel_values: list[float]
    specimen_currents: list[float]
    specimen_current_batch_numbers: list[int]  # Batch numbers aligned with specimen_currents
    match_scores: list[float]
    match_score_batch_numbers: list[int]  # Batch numbers aligned with match_scores
    white_pixel_percentages: list[float]
    batch_numbers: list[int]  # Batch sequence numbers (1, 2, 3...) for plot x-axis

    # Image accumulation for batch processing
    accumulated_images: list[np.ndarray]  # Batch of raw RTM images
    batch_count: int = 0  # Running count of completed batches (for plot x-axis)
    mean_pixel_at_delay: float | None = None  # Mean pixel when delay ends (for thresholding)

    # Per-pattern delay tracking
    delay_active: bool = True  # Each pattern manages its own delay state
    delay_match_below_count: int = 0  # Times match score was below threshold during delay
    initial_mean_pixel_value: float | None = None  # First mean pixel value for percentage comparison
    stall_history_start: int = 0  # Index into white_pixel_percentages where the stall latch's view begins (advanced on crop change: a new crop is a new energy scale)
    stall_dropped_carry: bool = False  # The trace had already dropped from its peak when a crop change reset the latch's history; lets an already-floored trace re-latch after the change (cleared if the post-crop trace shows renewed activity)
    last_gain_corrected: bool = False  # Previous scored batch applied an inter-frame gain correction; consecutive corrections are refused (a real uniform fade masquerades as sustained gain -- a genuine CB step is one-off)

    # Template matching
    reference_template: np.ndarray | None = None

    # History of crop rects: [{"batch": N, "rect": [x, y, w, h] | None}, ...].
    # Seeded at session start with batch 0; an operator change mid-run appends the
    # batch it takes effect from, so saved metrics can be attributed to the crop
    # that produced them (run_metadata's crop_rect alone loses mid-run changes).
    crop_rect_history: list = field(default_factory=list)

    # Timing
    start_monitoring_time: datetime.datetime | None = None
    completion_time: datetime.datetime | None = None  # Captured once at completion
    completion_logged: bool = False  # Track if we've logged completion
    criteria_met_batch: int | None = None  # Batch number at which criteria were met
    scan_direction: str = ""  # Pattern scan direction (e.g. "BottomToTop"); "" if unknown
    rotation: float = 0.0  # pattern.rotation in radians, CW-positive; 0.0 if unknown

    # Evaluation state
    evaluation_state: EvaluationState | None = None
    criteria: EvaluationCriteria | None = None  # Cached, rebuilt on parameter change


# ===========================
# Worker Thread
# ===========================

class WorkflowWorker(QObject):
    """
    Worker for the FIB patterning monitoring workflow.

    A plain QObject moved onto a dedicated QThread (see MainWindow.on_start_clicked)
    rather than a QThread subclass. This gives the object thread-affinity on the
    worker thread. Control slots (request_stop/pause/resume/update_parameter) are
    wired with DirectConnection, so they execute synchronously on the GUI thread
    and just set thread-safe flags (bools, GIL-atomic) or queue updates into the
    lock-protected WorkerParameterManager. The worker loop polls those flags each
    iteration. This means run() does not pump a worker event loop
    (no QCoreApplication.processEvents) -- removing the interleaving between this
    blocking loop and the GUI's signal handling that was implicated in a crash.

    run() is the long-running entry point (started by QThread.started). It emits
    finished() when done so the thread can quit and be cleaned up.
    """

    # Signals for GUI updates
    status_update = Signal(str)  # Status bar messages (timed, auto-clearing)
    persistent_status_update = Signal(str)  # Status bar messages (persistent until overwritten)
    # Sticky advisories (Auto CB outcomes): shown over the persistent text
    # until cleared or replaced, so ordinary state churn cannot wipe them.
    # An empty emission retires the standing advisory.
    warning_status_update = Signal(str)
    progress_update = Signal(int)  # Progress percentage (0-100)

    # Image display signals
    processed_image_ready = Signal(np.ndarray, int)  # (image, pattern_index)
    binary_image_ready = Signal(np.ndarray, int)  # (binary_image, pattern_index)

    # Pattern scan direction + rotation (emitted once per pattern when states are initialized)
    pattern_scan_direction_ready = Signal(int, str, float)  # (pattern_index, scan_direction, rotation_rad)

    # Plot data signals
    mean_pixel_data_ready = Signal(list, list, int)  # (x_data, y_data, pattern_index)
    specimen_current_data_ready = Signal(list, list, int)  # (x_data, y_data, pattern_index)
    specimen_current_disabled = Signal() # Emitted when system does not support specimen current
    match_score_data_ready = Signal(list, list, int)  # (x_data, y_data, pattern_index)

    # Results signals
    pattern_results_ready = Signal(dict, int)  # (results_dict, pattern_index)

    # Plot visibility control
    show_pattern_plots = Signal(int)  # pattern_index - show plots for this pattern

    # Calibration signal (adjusted_interval_seconds, was_clamped)
    calibration_complete = Signal(int, bool)

    # Monitoring icon state
    set_monitoring_icon_active = Signal(bool)  # True=color, False=grayscale

    # Control signals
    monitoring_started = Signal()
    microscope_connected = Signal()  # Connection established; until this fires
                                     # the GUI must not claim it is monitoring
    connection_status_update = Signal(str)  # Right-side connection indicator
                                            # (StatusText CONNECTING /
                                            # CONNECTED / NOT_CONNECTED)
    monitoring_stopped = Signal(str)  # Message when stopped by user
    monitoring_paused = Signal()  # Monitoring paused
    monitoring_resumed = Signal()  # Monitoring resumed
    session_reset = Signal()  # Emitted when evaluation resets for new patterning session
    save_plots_requested = Signal(str)  # plots_dir path - GUI should save plot PNGs
    error_occurred = Signal(str)  # Error message
    finished = Signal()  # Emitted when run() returns; drives thread teardown

    # Consecutive genuine RTM-communication failures before ending the run.
    # Only true comm exceptions count; benign no-data states (patterning idle,
    # FIB quadrant inactive, RTM off at the scope) do not (they return cleanly).
    RTM_FAILURE_LIMIT = 5

    # Auto-CB measurement configuration (internal). The correction algorithm
    # itself lives in rtm_data_processing.run_cb_correction with its own
    # constants; here only the measurement plumbing is configured.
    CB_CONTRAST_PERCENTILE = 2.0      # p2..p98 robust occupied span / edges

    # Floor for the post-write settle and the stale-frame re-grab wait, used
    # until _calibrate_image_rate has measured the real frame period. Above the
    # ~0.22 s observed on a Hydra so a settle is never shorter than one frame.
    CB_MIN_SETTLE_SECONDS = 0.3

    # Rollback flag: True restores the pre-3.3.1 behavior of restarting the RTM
    # before every calibration measurement (one restart per calibration plus a
    # stale-frame re-grab guard replaced it; flip this if stale frames appear).
    CB_RESTART_RTM_EACH_MEASUREMENT = False

    # Ask the RTM for the next data set during calibration measurements
    # (GetRtmDataSettings.wait_for_next_data) instead of reading whatever is
    # buffered. This is the authoritative frame-advance signal, but the SDK call
    # blocks until new data exists, and the manual pitches the flag at "jobs
    # where one pass takes long" -- so on a slow-pass pattern a measurement
    # could sit waiting, and a Stop cannot interrupt a call already inside the
    # SDK. Flip to False if that shows up on the tool: the rail-aware
    # discriminator carries on alone, and the 2026-08-23 field failure stays
    # fixed without it -- the rail-aware path is covered independently of this
    # flag, so turning it off is a supported fallback rather than a regression.
    CB_WAIT_FOR_NEW_RTM_DATA = True

    def __init__(self, parameters: WorkerParameters):
        super().__init__()
        self.parameters = parameters
        self.microscope = None
        self.pattern_states = []
        self._stop_requested = False
        self._paused = False
        self._patterning_was_running = False  # Track state transitions
        self._stop_reason = "Monitoring stopped"  # Default reason (overwritten by run())
        self._needs_cb_calibration = False  # Armed at transition; runs once RTM data flows
        # True once a calibration attempt ran this Start->Stop run. A fresh
        # worker is created per Start, so instance init is the per-run reset.
        # Gates the default first-session-only cadence at the trigger.
        self._cb_calibrated_this_run = False
        self._cb_summary = None             # calibration outcome for run metadata
        self._cb_trace = []                 # per-measurement telemetry rows
        self._cb_last_pooled_mean = None    # stale-RTM-frame discriminator (fallback)
        self._cb_last_balance = None        # CBBalance behind _cb_last_pooled_mean
        # Tri-state freshness of the last pooled measurement, from the SDK:
        #   True  -- every grab was confirmed a new frame (wait_for_next_data)
        #   False -- the RTM reported no new data (genuinely frozen / job ended)
        #   None  -- unknown; the SDK flag is unavailable, so fall back to the
        #            statistical discriminator
        self._cb_last_frame_fresh = None

        # Idle loop emission guards (emit once, not every iteration)
        self._paused_status_emitted = False
        self._fib_inactive_logged = False
        self._waiting_for_patterning_logged = False
        self._waiting_for_beam_logged = False  # Emit "FIB beam off" wait once
        self._fib_inactive_paused_logged = False  # Emit "FIB quadrant inactive" pause once
        self._validation_failed_logged = False  # Suppresses repeated validation logging
        self._validation_warning_posted = False  # A validation-failure warning stands
        self._state_check_failed_logged = False  # Suppresses repeated comm-error logging

        # RTM-communication failure tracking (genuine comm errors only)
        self._consecutive_rtm_failures = 0
        self._rtm_failure_stop = False  # True when the run ended due to RTM comm loss
        self._error_stop = False  # True when run() ended via an unhandled exception

        # Processing state
        self._pattern_metadata = None

        # Parameter manager for live updates
        self.param_manager = WorkerParameterManager()

        # Image rate calibration (TIME_BASED mode)
        self._image_timestamps: list[float] = []
        self._rate_calibrated: bool = False
        self._images_per_second: float = 0.0

        # Save data manager (created per patterning session if save_data enabled)
        self._save_manager: SaveDataManager | None = None
        self._pattern_infos = None  # Stored for metadata at finalization
        self._microscope_data: dict = {}  # Cached at patterning start

        # System capabilities (determined at connection from system name)
        self._specimen_current_available: bool = True

    def _format_duration(self, duration: datetime.timedelta) -> str:
        """
        Format a timedelta into MM:SS clock format.

        For durations >= 1 hour, uses H:MM:SS format.

        Examples:
        - 00:45
        - 02:30
        - 59:59
        - 1:15:20

        :param duration: Time duration to format
        :return: Clock-formatted duration string
        """
        total_seconds = int(duration.total_seconds())

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @Slot()
    def run(self):
        """Main workflow execution - continuous monitoring until stopped."""
        try:
            # No status emission here: a timed "starting" message covered the
            # persistent "Connecting to microscope..." for its first 5 s --
            # exactly the window in which the connection attempt is what the
            # operator needs to see.
            self.monitoring_started.emit()

            # Step 1: Connect to microscope
            if not self._connect_to_microscope():
                return

            # Step 2: Wait until the FIB beam is on. The beam auto-offs after ~60
            # min idle, and several AutoScript calls (e.g. service.system.name)
            # fault natively when it is off, so nothing heavier runs until the
            # beam is confirmed on. While waiting we sit idle, like "waiting for
            # patterning". Returns False only if Stop was requested while waiting.
            if not self._wait_for_fib_beam():
                return

            # Step 3: Validate FIB is active imaging device (beam is on -> safe)
            if not self._validate_device():
                return

            # Step 4: Read system capabilities (beam is on -> safe)
            self._read_system_capabilities()

            # Step 5: Setup RTM monitoring (includes initial restart)
            if not self._setup_rtm():
                return

            # Step 6: Run continuous monitoring loop
            self._continuous_monitoring_loop()
            # Preserve a specific stop reason set inside the loop (e.g. RTM comm
            # loss); only fall back to the user-stop message otherwise.
            if not self._rtm_failure_stop:
                self._stop_reason = "Monitoring stopped by user"

        except Exception as e:
            logger.error("Workflow error", exc_info=True)
            self.error_occurred.emit(f"Workflow error: {str(e)}")
            self._stop_reason = "Monitoring stopped (error)"
            self._error_stop = True

        finally:
            self._cleanup()
            # Push our affinity back to the main thread while we still own it
            # (moveToThread may only push from the owning thread). Establishes
            # the invariant: thread.isFinished() => worker affinity == main
            # thread, so all later destruction is same-thread and legal.
            try:
                app = QCoreApplication.instance()
                if app is not None:
                    self.moveToThread(app.thread())
            except Exception:
                logger.warning("Could not re-home worker to main thread", exc_info=True)
            # Signal the owning thread that work is done so it can quit and the
            # worker/thread can be deleted (wired in MainWindow.on_start_clicked).
            self.finished.emit()

    def _connect_to_microscope(self) -> bool:
        """Connect to the microscope.

        The connection state rides its own channel into the status bar's
        right-side indicator (permanent, never covered by message traffic):
        with no microscope reachable, a timed message once expired
        mid-attempt and the app claimed "waiting for patterning" with
        nothing to wait on. A failure leaves the operator-worded error as
        the run's stop reason on the left side (the log carries the detail
        via error_occurred) while the indicator falls back to not-connected.
        """
        self.connection_status_update.emit(StatusText.CONNECTING)
        self.progress_update.emit(5)

        microscope, message = connect_to_microscope(self.parameters.microscope_host)

        if microscope is None:
            self._stop_reason = "Connection error (see console)"
            self.error_occurred.emit(f"Connection failed: {message}")
            self.connection_status_update.emit(StatusText.NOT_CONNECTED)
            return False

        self.microscope = microscope
        self.microscope_connected.emit()
        self.connection_status_update.emit(StatusText.CONNECTED)
        return True

    def _wait_for_fib_beam(self) -> bool:
        """
        Block until the FIB (ion) beam is on, or Stop is requested.

        The ion beam auto-offs after ~60 min idle, and several AutoScript calls
        (e.g. service.system.name) fault natively when it is off, so we confirm
        the beam is on before any heavier startup call. While waiting we sit idle,
        like the "waiting for patterning" state.

        :return: True once the beam is on; False if Stop was requested first.
        """
        emitted = False
        while not self._stop_requested:
            # Stop is delivered via DirectConnection (sets self._stop_requested
            # from the GUI thread), so no event-loop pump is needed here.
            beam_on, _msg = check_fib_beam_on(self.microscope)
            if beam_on:
                if emitted:
                    self.status_update.emit("FIB beam on - starting up...")
                    self._waiting_for_beam_logged = False
                return True

            # Beam off or unreadable -> wait quietly (emit once).
            if not emitted:
                self.set_monitoring_icon_active.emit(False)
                self.persistent_status_update.emit(StatusText.FIB_BEAM_OFF)
                emitted = True
            time.sleep(self.parameters.acquisition_delay_seconds)

        return False

    def _read_system_capabilities(self):
        """
        Read system capabilities (system name) after the FIB is confirmed active.

        microscope.service.system.name has been observed to trigger a native
        access violation in the AutoScript transport when the FIB is off/not the
        active device, so this must run only after _validate_device() passes.
        Non-fatal: on any failure we keep the default (specimen current enabled).
        """
        system_name = get_system_name(self.microscope)
        self._specimen_current_available = (system_name != "Arctis")
        if not self._specimen_current_available:
            logger.info(
                f"System '{system_name}' does not support specimen current "
                f"- specimen current plotting disabled"
            )
            self.specimen_current_disabled.emit()

    def _validate_device(self) -> bool:
        """Validate FIB is active imaging device."""
        self.status_update.emit("Validating FIB device...")
        self.progress_update.emit(10)

        is_valid, message = validate_active_device(self.microscope)
        if not is_valid:
            self._stop_reason = f"Device validation failed: {message}"
            self.error_occurred.emit(self._stop_reason)
            return False

        self.status_update.emit("FIB is the active device")
        return True

    def _setup_rtm(self) -> bool:
        """Setup RTM monitoring (includes initial RTM restart)."""
        self.status_update.emit("Setting up RTM monitoring...")
        self.progress_update.emit(15)

        success, message = setup_rtm_monitoring(
            self.microscope,
            mode=self.parameters.rtm_mode.name  # Convert enum to string (e.g., "LOW_RESOLUTION")
        )

        if not success:
            self._stop_reason = f"RTM setup failed: {message}"
            self.error_occurred.emit(self._stop_reason)
            return False

        self.status_update.emit("RTM monitoring configured")
        return True

    def _continuous_monitoring_loop(self):
        """
        Continuous monitoring loop.

        Runs until Stop button is pressed. Handles:
        - Patterning starting and stopping
        - Pattern validation
        - Automatic patterning stop when criteria met
        - RTM restarts at appropriate times
        """
        # Log only: the catbug label owns the monitoring state (surface
        # separation, operator request 2026-08-27) -- the bar is reserved for
        # what the catbug cannot show.
        logger.info("Monitoring ready")
        self.progress_update.emit(20)

        while not self._stop_requested:
            # NOTE: no QCoreApplication.processEvents() here. Control signals
            # (stop/pause/resume/param) are delivered via DirectConnection, so
            # they set their flags / queue updates synchronously from the GUI
            # thread; we just read those flags each iteration. Pumping the worker
            # event loop here previously interleaved this blocking loop with the
            # GUI's signal handling and was implicated in a startup crash.

            # Check if paused
            if self._paused:
                if not self._paused_status_emitted:
                    # Log only (once per pause): the catbug label already
                    # says "Monitoring paused"; this records when the loop
                    # actually honored the request.
                    logger.info("Worker loop paused")
                    self._paused_status_emitted = True
                time.sleep(0.5)  # Check every 500ms
                continue
            self._paused_status_emitted = False

            # Check the FIB beam first. It auto-offs after ~60 min idle, and some
            # AutoScript calls fault natively while it is off, so we must not poll
            # patterning state (or anything heavier) until the beam is on. While
            # off we sit and wait, exactly like the patterning-not-active state.
            beam_on, _beam_msg = check_fib_beam_on(self.microscope)
            if beam_on is None:
                # Comm error reading beam state -- treat like a state-check failure:
                # keep the session alive, warn once, back off, retry.
                if not self._state_check_failed_logged:
                    logger.warning("FIB beam state check failed - retrying")
                    self.persistent_status_update.emit(StatusText.COMM_RETRY)
                    self._state_check_failed_logged = True
                time.sleep(self.parameters.acquisition_delay_seconds)
                continue
            if not beam_on:
                if not self._waiting_for_beam_logged:
                    self.set_monitoring_icon_active.emit(False)
                    self.persistent_status_update.emit(StatusText.FIB_BEAM_OFF)
                    self._waiting_for_beam_logged = True
                # The beam being off ends any active session; clear the transition
                # flag so a fresh session validates when the beam (and patterning)
                # return. Finalize any saved data first: the aborted sessions are
                # the highest-value diagnostics, and the 2026-08-17 hard-failure
                # runs lost their metrics because nothing finalized on this path.
                if self._patterning_was_running:
                    logger.info("FIB beam turned off - ending current session")
                    self._finalize_saved_data(
                        RUN_STATUS_BEAM_OFF, "FIB beam turned off mid-session"
                    )
                    self._patterning_was_running = False
                self._validation_failed_logged = False
                self._fib_inactive_paused_logged = False
                time.sleep(self.parameters.acquisition_delay_seconds)
                continue
            self._waiting_for_beam_logged = False

            # Check if patterning is running (tri-state: True / False / None)
            is_running, message = check_patterning_state(self.microscope)

            # Communication error (None): do not treat as "patterning stopped".
            # That would flip _patterning_was_running and, on the next good poll,
            # wipe the whole session (plots, metrics, reference template,
            # confirmation progress) -- and a session wiped mid-run can miss real
            # completion. Keep the session alive, warn once, back off, and retry.
            if is_running is None:
                if not self._state_check_failed_logged:
                    logger.warning(
                        f"Patterning state check failed ({message}) - "
                        f"keeping session alive and retrying"
                    )
                    self.persistent_status_update.emit(StatusText.COMM_RETRY)
                    self._state_check_failed_logged = True
                time.sleep(self.parameters.acquisition_delay_seconds)
                continue
            self._state_check_failed_logged = False

            if not is_running:
                # A not-running reading is only trustworthy while the FIB is
                # the active imaging device: with another quadrant selected
                # (e.g. the SEM), the state check reads not-running even
                # though the mill continues. Treating that as "patterning
                # stopped" finalized the session, and re-selecting the FIB
                # then started a fresh one -- re-validation, a new run
                # directory, and a second Auto CB mid-mill (field report
                # 2026-08-25). Pause instead: keep the session and its saved
                # data alive, say why, and resume when the FIB is selected
                # again. A device-check comm error (None) holds too -- same
                # rationale as the state-check comm error above.
                fib_active, device_msg = validate_active_device(self.microscope)
                if fib_active is not True:
                    if not self._fib_inactive_paused_logged:
                        # The device message goes to the log: the 2026-08-27
                        # idle flapping (three pauses in a minute with the FIB
                        # apparently selected) could not be diagnosed because
                        # this line hid whether the check read another device
                        # (False, "current: N") or failed outright (None).
                        logger.info(
                            "FIB is not the active quadrant - monitoring "
                            f"paused until it is selected again ({device_msg})"
                        )
                        self.set_monitoring_icon_active.emit(False)
                        self.persistent_status_update.emit(
                            StatusText.FIB_INACTIVE_PAUSED
                        )
                        self._fib_inactive_paused_logged = True
                    time.sleep(self.parameters.acquisition_delay_seconds)
                    continue

                # Patterning not active, with the FIB confirmed active: the
                # reading is trustworthy. A pause that ends here (the mill
                # stopped while another quadrant was selected) resolves as a
                # normal patterning stop.
                if self._fib_inactive_paused_logged:
                    # The pause can resolve on this idle path too. On
                    # 2026-08-27 the device check flapped three times between
                    # runs and this path reset the flag silently -- the
                    # persistent pause text stood through the whole episode
                    # and the operator had to flip SEM->FIB by hand to clear
                    # it. Same brief confirmation as the mid-run resume,
                    # persistent clear first (see the resume branch below).
                    logger.info(
                        "FIB quadrant active again - monitoring resumed"
                    )
                    self.persistent_status_update.emit("")
                    self.status_update.emit(StatusText.FIB_ACTIVE_RESUMED)
                self._fib_inactive_paused_logged = False
                if not self._waiting_for_patterning_logged:
                    self.set_monitoring_icon_active.emit(False)
                    # Log only: the grayscale catbug and its "waiting for
                    # patterning" message own this state (surface
                    # separation, operator request 2026-08-27).
                    logger.info("Waiting for patterning to start")
                    self._waiting_for_patterning_logged = True

                # Clear "patterning was running" flag. Finalize any saved data
                # first (metrics.csv / run_metadata.json / plots): the operator
                # stopping patterning at the microscope before criteria is the
                # classic abort path, and the 2026-08-17 hard-failure runs
                # (11/12 - the gridbar hangs) lost all their metrics because
                # nothing finalized here.
                if self._patterning_was_running:
                    logger.info("Patterning stopped")
                    self._finalize_saved_data(
                        RUN_STATUS_PATTERNING_STOPPED,
                        "Patterning stopped on-tool before criteria were met"
                    )
                    self._patterning_was_running = False

                # Reset validation guard so next patterning session triggers fresh validation
                self._validation_failed_logged = False

                time.sleep(self.parameters.acquisition_delay_seconds)
                continue
            self._waiting_for_patterning_logged = False

            if self._fib_inactive_paused_logged:
                # Back from a quadrant flip. When the session survived the
                # pause it simply continues -- no re-validation, no new run
                # directory, no second Auto CB. (A batch spanning the pause
                # can mix pre- and post-pause frames; that is the same brief
                # mixing any RTM hiccup produces and the evaluation
                # machinery already tolerates it.)
                logger.info("FIB quadrant active again - monitoring resumed")
                # Brief confirmation only (operator request 2026-08-27: the
                # standing "resumed" banner outlived the whole patterning
                # session). Clear the persistent pause text before the timed
                # emit -- a timed message expires by restoring the persistent
                # slot, which still holds the stale pause wording; and emitting
                # the persistent clear second would erase the timed overlay.
                self.persistent_status_update.emit("")
                self.status_update.emit(StatusText.FIB_ACTIVE_RESUMED)
                self.set_monitoring_icon_active.emit(True)
                self._fib_inactive_paused_logged = False

            # Patterning is running!
            # Detect transition from stopped to running
            if not self._patterning_was_running:

                # Guard: if validation already failed for this patterning session
                # (e.g. aspect ratio below threshold), wait quietly until patterning
                # stops and a new session begins rather than re-validating every cycle.
                if self._validation_failed_logged:
                    time.sleep(self.parameters.acquisition_delay_seconds)
                    continue

                logger.info("Patterning started - restarting RTM")
                crash_breadcrumbs.drop("patterning-detected")
                # No bar emission: the colored catbug and its "patterning
                # detected" message own this state (surface separation,
                # operator request 2026-08-27 -- supersedes the v3.3.13
                # persistent baseline that lived here).
                restart_rtm(self.microscope)

                # Reset calibration for new patterning session
                self._image_timestamps.clear()
                self._rate_calibrated = False

                time.sleep(0.5)  # Brief pause after RTM restart

                # Validate patterns once at patterning start
                validation_result = validate_patterns(
                    self.microscope,
                    aspect_ratio_threshold=self.parameters.aspect_ratio_threshold
                )

                if not validation_result.success:
                    self.set_monitoring_icon_active.emit(False)
                    # Warning channel, not persistent: a standing advisory
                    # (settings change, a previous session's Auto CB outcome)
                    # sits on top of the persistent slot, and a failed
                    # validation never reaches the calibration whose start
                    # would clear it -- on 3.4.2 the failure message was
                    # hidden for as long as any advisory stood (field report
                    # 2026-08-27). As a warning it replaces the stale
                    # advisory and clears itself when a later session
                    # validates and calibrates.
                    self.warning_status_update.emit(
                        f"Pattern validation failed: {validation_result.message}"
                    )
                    self._validation_warning_posted = True
                    logger.warning(
                        f"Pattern validation failed: {validation_result.message}"
                    )
                    # Don't set _patterning_was_running - retry next iteration
                    self._validation_failed_logged = True
                    time.sleep(self.parameters.acquisition_delay_seconds)
                    continue

                # Validation passed - initialize pattern states
                if self._validation_warning_posted:
                    # The failure warning must not outlive its truth: a valid
                    # session is starting now (field screenshot 2026-08-27:
                    # a rejected stress-relief cut's aspect-ratio message
                    # stood through the whole run that followed it). Cleared
                    # here rather than left to the calibration-start clear,
                    # which never comes when Auto CB is disabled.
                    self.warning_status_update.emit("")
                    self._validation_warning_posted = False
                self._patterning_was_running = True
                self.set_monitoring_icon_active.emit(True)

                # Clear previous session's plots before starting new session
                self.session_reset.emit()

                self._initialize_pattern_states(validation_result.patterns)

                self._arm_cb_calibration()
                self._cb_summary = None
                self._cb_trace = []

                # Initialize save data manager for this patterning session
                if self.parameters.save_data and self.parameters.script_root:
                    try:
                        self._save_manager = SaveDataManager(
                            self.parameters.script_root
                        )
                        self._pattern_infos = validation_result.patterns
                        for ps in self.pattern_states:
                            self._save_manager.initialize_pattern(ps.pattern_index)
                        self._microscope_data = self._read_microscope_data()
                        logger.info("Save data manager initialized for this session")
                        crash_breadcrumbs.drop("run-dir created")
                    except Exception as e:
                        logger.error(f"Failed to initialize save data: {e}")
                        self._save_manager = None

                # Log validated pattern details
                for i, info in enumerate(validation_result.patterns, 1):
                    logger.info(
                        f"Pattern {i} ({info.pattern_type}): "
                        f"{info.width * 1e6:.2f} µm × {info.height * 1e6:.2f} µm, "
                        f"AR={info.aspect_ratio:.3f}"
                    )

                # Show plots for each pattern
                for ps in self.pattern_states:
                    self.show_pattern_plots.emit(ps.pattern_index)
                    logger.info(f"Showing plots for Pattern {ps.pattern_index + 1}")

            # Process RTM data
            processing_successful = self._process_rtm_data()

            if not processing_successful:
                # End the run if we've had too many consecutive genuine RTM
                # communication failures (the monitor is blind, so auto-stop
                # at criteria can no longer work). Benign no-data states do not
                # increment the counter, so this fires only on a real comm loss.
                # Per design: surface a message and stop monitoring -- do not
                # stop the beam (a separate application owns the mill).
                if self._consecutive_rtm_failures >= self.RTM_FAILURE_LIMIT:
                    self._stop_reason = (
                        f"Monitoring stopped: no RTM data received after "
                        f"{self.RTM_FAILURE_LIMIT} consecutive attempts "
                        f"(check microscope connection)"
                    )
                    logger.error(self._stop_reason)
                    self.persistent_status_update.emit(self._stop_reason)
                    self.set_monitoring_icon_active.emit(False)
                    self._rtm_failure_stop = True
                    break

                time.sleep(self.parameters.acquisition_delay_seconds)
                continue

            # Calibrate image rate (TIME_BASED mode)
            if self.parameters.window_mode == WindowMode.TIME_BASED:
                self._calibrate_image_rate()

            # Check if all patterns are complete
            evaluation_states = [ps.evaluation_state for ps in self.pattern_states]
            if check_all_patterns_complete(evaluation_states):
                # Log completion with durations for each pattern
                logger.info("All patterns complete")

                for ps in self.pattern_states:
                    if ps.start_monitoring_time is not None and ps.completion_time is not None:
                        duration = ps.completion_time - ps.start_monitoring_time
                        duration_str = self._format_duration(duration)

                        logger.info(
                            f"  Pattern {ps.pattern_index + 1}: {duration_str}"
                        )

                # Trailing ellipsis: this reports an action in progress (the
                # stop command is about to be issued), not a finished state.
                self.status_update.emit(
                    "All patterns complete - stopping patterning..."
                )

                # Stop patterning
                success, message = stop_patterning(self.microscope)
                if success:
                    logger.info("Patterning stopped successfully")
                else:
                    logger.warning(f"Failed to stop patterning: {message}")

                # Restart RTM to prepare for next patterning session
                restart_rtm(self.microscope)
                logger.info("RTM restarted - ready for next patterning session")

                # Finalize saved data before resetting
                self._finalize_saved_data(
                    RUN_STATUS_CRITERIA_MET,
                    "All patterns met completion criteria",
                )

                # Reset evaluation states for next patterning run
                self._reset_evaluation_states()

                # Set flag to false so we can detect next patterning start
                self._patterning_was_running = False

                # Continue monitoring (don't exit loop)
                continue

            # Wait before next acquisition
            time.sleep(self.parameters.acquisition_delay_seconds)

        self.progress_update.emit(100)

    def _arm_cb_calibration(self):
        """
        Arm the one-shot CB calibration for a patterning session.

        It does not run here -- at the stopped->running transition the RTM was
        just restarted and has no data yet. It runs from _process_rtm_data once a
        valid RTM frame arrives, still before any batch is accumulated into the
        analysis baseline (mean-pixel reference, match template, frozen Otsu
        threshold), so CB is locked before the baseline is captured. Fail-open.

        The auto_cb_on_start opt-in is deliberately not read here, and that is
        the whole contract of this method. Arming happens before _process_rtm_data
        drains queued parameter updates, so an operator who enables Auto CB
        mid-run still has that change sitting in the queue at this point. Reading
        the flag here armed from the stale value: the trigger never fired, no
        status was emitted, and run_metadata.json still recorded
        auto_cb_on_start: true for a session whose detector was never touched.
        The trigger reads the flag after the drain instead.
        """
        self._needs_cb_calibration = True

    def _at_batch_boundary(self) -> bool:
        """
        True when at least one pattern is starting a fresh batch -- the point at
        which device validation and queued parameter updates are applied.

        Any, not all, and that is the whole point. The two patterns' buffers fill
        independently: a pattern with no RTM image on a frame is skipped by the
        dispatch loop, and _process_pattern clears its own buffer when the image
        dimensions change. Either event offsets the two accumulation cycles by a
        constant, and since each buffer then runs 0,1,...,N-1,0 with period
        number_of_images, they are never simultaneously empty again. Requiring
        simultaneity therefore let one dropped frame close this gate for the rest
        of the patterning session, silently disabling every live parameter update
        and the FIB active-device check (the operator saw the crop box move and
        got no status message). It healed only at the next patterning session,
        when _reset_evaluation_states re-empties both buffers.

        With patterns in lockstep -- the healthy case -- both are empty on the
        same iteration and this is identical to the old behavior; it differs
        only in the state that used to be terminal. The device check keeps its
        intent of not costing a microscope call per image: it now runs about once
        per five iterations in a two-pattern run rather than once per ten.

        The empty-list guard must not be folded away: ``any([])`` is False where
        ``all([])`` was True, so dropping it would silently invert the verdict for
        an empty pattern-states list. Nothing reaches this method with one today
        -- _process_rtm_data runs only after _initialize_pattern_states succeeded
        -- but the boundary has to stay True if anything ever does, or the
        parameter drain and the FIB active-device check would be gated off
        entirely.

        :return: bool
        """
        return (not self.pattern_states
                or any(len(ps.accumulated_images) == 0
                       for ps in self.pattern_states))

    def _process_rtm_data(self) -> bool:
        """
        Acquire and process RTM data for all patterns.

        Device validation and parameter updates are performed at batch
        boundaries (see _at_batch_boundary) to reduce microscope API calls
        during mid-batch image collection.

        :return: True if processing successful, False otherwise
        """
        if self._at_batch_boundary():
            # Validate FIB is still the active imaging device (tri-state)
            is_valid, message = validate_active_device(self.microscope)

            if is_valid is None:
                # Genuine communication failure -- count it. Do not reset on a
                # later FIB-active poll here; only successful RTM acquisition
                # clears the streak (the caller escalates at RTM_FAILURE_LIMIT).
                self._consecutive_rtm_failures += 1
                logger.warning(
                    f"Active-device check failed ({message}) - consecutive "
                    f"RTM comm failures: {self._consecutive_rtm_failures}"
                )
                return False

            if not is_valid:
                # FIB is not the active device -- benign operator state, not a
                # comm failure. Comm is working, so clear the failure streak.
                self._consecutive_rtm_failures = 0
                if not self._fib_inactive_logged:
                    logger.info(
                        f"FIB is no longer the active device - "
                        f"pausing at batch boundary ({message})"
                    )
                    self.set_monitoring_icon_active.emit(False)
                    # Same state as the loop-level quadrant pause, so the
                    # same words -- and persistent, not timed: a timed
                    # message expired after 5 s and restored stale text
                    # while the pause continued.
                    self.persistent_status_update.emit(
                        StatusText.FIB_INACTIVE_PAUSED
                    )
                    self._fib_inactive_logged = True
                return False

            # Log recovery if FIB was previously inactive
            if self._fib_inactive_logged:
                logger.info("FIB is the active device again - resuming monitoring")
                self.set_monitoring_icon_active.emit(True)
                # Clear the pause text, confirm briefly (operator request
                # 2026-08-27) -- persistent clear first, or it would erase
                # the timed overlay it is meant to underlie.
                self.persistent_status_update.emit("")
                self.status_update.emit(StatusText.FIB_ACTIVE_RESUMED)
                self._fib_inactive_logged = False

            # Apply pending parameter updates (thread-safe)
            if self.param_manager.has_pending_updates():
                status_messages = self.param_manager.apply_pending_updates(
                    self.pattern_states,
                    self.parameters
                )

                # Emit status messages to GUI
                for msg in status_messages:
                    if msg:
                        self.status_update.emit(msg)
                        logger.info(f"Parameter update applied: {msg}")

                # Recalculate window size if analysis interval changed (TIME_BASED mode)
                self._recalculate_window_size()

                # Rebuild cached criteria (thresholds or enabled flags may have changed)
                for ps in self.pattern_states:
                    self._build_criteria(ps)

        # Acquire RTM data. A None return means acquire_rtm_data caught an
        # exception (genuine comm failure) -- distinct from an empty result,
        # which AutoScript returns when RTM simply has no data yet (benign).
        rtm_data, rtm_positions, message = acquire_rtm_data(self.microscope)

        if rtm_data is None or rtm_positions is None:
            self._consecutive_rtm_failures += 1
            logger.warning(
                f"RTM acquisition failed ({message}) - consecutive RTM comm "
                f"failures: {self._consecutive_rtm_failures}"
            )
            return False

        # Communication with the RTM subsystem succeeded -- clear the streak.
        self._consecutive_rtm_failures = 0

        # Precompute metadata if first time
        if self._pattern_metadata is None:
            self._pattern_metadata = precompute_pattern_metadata(rtm_positions)

        # Convert RTM data to images, keyed by pattern_id (see note at the
        # dispatch loop below).
        rtm_images = get_images_from_rtm_data(
            rtm_data,
            rtm_positions,
            self._pattern_metadata
        )

        if not rtm_images or all(img is None for img in rtm_images.values()):
            # Empty/no images is benign (RTM off at the scope or no data yet);
            # comm succeeded above, so this does not count as a failure.
            logger.warning("No RTM images received")
            return False

        # RTM data is now flowing. Run the one-shot CB calibration here (rather
        # than at the stopped->running transition, where the freshly-restarted
        # RTM has no data yet) -- but before any batch is accumulated, so CB is
        # locked before the analysis baseline is captured. Skip this cycle's data
        # so the baseline starts clean under the locked CB; the next cycle
        # resumes normal processing.
        if self._needs_cb_calibration:
            self._needs_cb_calibration = False
            # The opt-in and the cadence are read here, after the parameter
            # drain above, so a mid-run change takes effect on the next
            # patterning session as the operator was told it would. Reading
            # them at the arming point instead made enabling Auto CB mid-run
            # a silent no-op. Cadence: by default only the first patterning
            # session of a run calibrates -- every calibration mills the
            # live pattern for its duration, and sessions after the first
            # start from the previous session's still-held lock (field
            # decision 2026-08-26, after an 11 s non-convergence at high beam
            # current). The Recalibrate Every Session checkbox restores the
            # per-session behavior, and is live-editable so one session can
            # be recalibrated mid-run after detector drift.
            if self.parameters.auto_cb_on_start:
                if (self.parameters.cb_recalibrate_every_session
                        or not self._cb_calibrated_this_run):
                    self._cb_calibrated_this_run = True
                    self._calibrate_detector_cb()
                    return True
                logger.info(
                    "Auto CB: already calibrated this run - holding the "
                    "existing lock (enable Recalibrate Every Session to "
                    "calibrate each patterning session)"
                )

        # Process each pattern (delay is managed per-pattern).
        # Route by pattern_id, not by list position: the microscope can return
        # the RTM patterns in a different order between frames, so a positional
        # zip would assign one pattern's image to the other pattern on those
        # frames (manifests as the two crops' dimensions swapping every frame and
        # corrupting the rolling match score). rtm_images is keyed by pattern_id;
        # a missing/None entry means RTM data was unavailable for that pattern.
        for pattern_state in self.pattern_states:
            rtm_image = rtm_images.get(pattern_state.pattern_id)
            if rtm_image is None:
                logger.debug(
                    f"Pattern {pattern_state.pattern_index + 1}: "
                    f"No RTM image this cycle - skipping"
                )
                continue
            self._process_pattern(
                pattern_state,
                rtm_image
            )

        return True

    # ===========================
    # Auto contrast/brightness calibration (one-shot at patterning start)
    # ===========================

    def _rtm_frame_period(self) -> float:
        """
        Seconds between RTM frames: measured once the image rate is calibrated,
        else a conservative default.

        Auto CB runs before _calibrate_image_rate on a fresh patterning session,
        so the default is the normal case. It sits deliberately above the
        ~0.22 s observed on a Hydra so a settle can never be shorter than one
        frame.
        """
        if self._rate_calibrated and self._images_per_second > 0:
            return 1.0 / self._images_per_second
        return self.CB_MIN_SETTLE_SECONDS

    def _cb_settle_seconds(self) -> float:
        """
        Post-write settle: the operator's cb_settle_seconds, floored at one RTM
        frame period.

        The setting stays authoritative as a lower bound rather than verbatim.
        Its 0.2 s default is shorter than the ~0.22 s frame period, which
        guaranteed the first grab after a CB write could return a frame acquired
        before the write landed -- work the stale-frame guard then had to absorb
        (2026-08-23).
        """
        return max(self.parameters.cb_settle_seconds, self._rtm_frame_period())

    def _frames_frozen(self, balance, prev_mean, prev_balance, cfg) -> bool:
        """
        True when the RTM buffer did not advance across a CB change.

        Three sources of evidence:

        1. The SDK. wait_for_next_data makes get_data() return the next data set
           or nothing at all, so a False _cb_last_frame_fresh is decisive: the
           RTM itself reported no new frame. A True decides nothing the other
           way -- on 2026-08-27 (field Runs 16/17) the SDK kept vouching fresh
           data sets after a large gain-up move while their pixel content stayed
           bit-identical across up to 22 knob moves (including brightness
           driven to 0.0, which no live detector ignores); an RTM restart
           cleared it. "New data set" evidently does not guarantee new pixel
           content, so a fresh flag no longer skips the statistical test.
        2. A rail-dominated previous frame exempts the statistical test
           entirely. A saturated scene reproduces its statistics exactly, so
           identical numbers there are evidence of physics, not of a stale
           buffer -- and railed is precisely the state the search exists to walk
           out of, one deliberately-identical step at a time.
        3. Otherwise: a bit-identical pooled mean across a knob change. A live
           detector's pooled mean over thousands of noisy pixels cannot repeat
           bit-for-bit after a knob change unless the frame is a copy.

        A None balance the SDK has not already called frozen is a measurement
        failure (empty grabs, comms blip), not evidence about frame advance --
        it is reported as not-frozen so the caller's retry path handles it.
        When _cb_last_frame_fresh is False, evidence 1 wins: the RTM said
        outright that no new frame exists, so the empty pool behind that None is
        the frozen-buffer signature and is reported as frozen.

        :param balance: the CBBalance just measured, or None
        :param prev_mean: pooled mean before this measurement, or None
        :param prev_balance: CBBalance behind prev_mean, or None
        :param cfg: CBEdgeConfig (for white_level)
        :return: bool
        """
        if self._cb_last_frame_fresh is False:
            return True
        if balance is None:
            return False
        if prev_mean is None or prev_balance is None:
            return False
        if cb_rail_dominated(prev_balance):
            return False
        if not 0.0 < prev_mean < cfg.white_level:
            # Perfectly rail-pinned (mean 0 or full scale) is legitimately
            # identical. Subsumed by the rail-dominated test above; kept as a
            # cheap backstop against an unexpected white_level.
            return False
        return self._cb_last_pooled_mean == prev_mean

    def _measure_cb_balance(self, white_level=None, frames=None):
        """
        Grab RTM frames, pool their valid pixels, and return a CBBalance (or None
        if no usable pixels were captured).

        Deliberately bypasses _process_pattern's batch accumulation so calibration
        speed is independent of number_of_images / analysis_interval.

        :param white_level: detector full-scale to normalize against; defaults to the
            configured cb_white_level (the calibration loop passes the auto-detected
            full-scale instead).
        :param frames: frames to pool; defaults to cb_frames_per_measurement. The
            calibration loop passes 1 for cheap search measurements and the
            configured count for accept/verify measurements.
        """
        pooled = []
        if frames is None:
            frames = self.parameters.cb_frames_per_measurement
        frames = max(1, int(frames))
        # Freshness evidence for this measurement, tallied across the grabs and
        # resolved into the tri-state _cb_last_frame_fresh below.
        confirmed_grabs = 0     # SDK vouched the frame was one we had not seen
        unconfirmed_grabs = 0   # SDK flag unavailable -> freshness unknown
        no_new_data = False     # SDK said outright that no new frame exists
        # A wholly empty result gets retried after one acquisition period: the
        # RTM buffer is legitimately empty right after a restart and during
        # transient comm blips, and a single un-retried grab used to abort the
        # entire calibration ('no-measure'/'measure-failed') on that timing
        # (review finding, 2026-08-15). Three attempts ~ the old 3-frame
        # pooling's tolerance.
        for attempt in range(3):
            if self._stop_requested:
                # wait_for_next_data blocks in the SDK; never start another one
                # after a Stop, and do not sit out the empty-grab retries either
                # -- teardown has to stay prompt.
                break
            for fi in range(frames):
                if self._stop_requested:
                    break
                try:
                    # Ask the RTM for a frame the calibration has not seen. A
                    # "no new data" answer is decisive; a "fresh" answer is
                    # advisory only -- the 2026-08-27 field freeze delivered
                    # "fresh" data sets with bit-identical content, so the
                    # pooled-mean discriminator always gets the final word
                    # (see _frames_frozen).
                    rtm_data, rtm_positions, msg = acquire_rtm_data(
                        self.microscope,
                        wait_for_next_data=self.CB_WAIT_FOR_NEW_RTM_DATA)
                    if msg == RTM_NO_NEW_DATA:
                        no_new_data = True
                        continue
                    if rtm_data is None or rtm_positions is None:
                        continue
                    if msg == RTM_FRESH_DATA:
                        confirmed_grabs += 1
                    else:
                        unconfirmed_grabs += 1
                    images = get_images_from_rtm_data(rtm_data, rtm_positions)
                except Exception:
                    logger.warning("Auto CB: measurement frame acquisition failed", exc_info=True)
                    continue
                # images is keyed by pattern_id; order is irrelevant here (pooling pixels).
                for img in images.values():
                    if img is None:
                        continue
                    arr = np.asarray(img)
                    valid = arr[arr >= 0]
                    if valid.size:
                        pooled.append(valid.ravel())
                if fi < frames - 1:
                    time.sleep(self.parameters.acquisition_delay_seconds)
            if pooled:
                break
            if attempt < 2:
                logger.debug("Auto CB: empty RTM grab - waiting one period and retrying")
                time.sleep(self.parameters.acquisition_delay_seconds)

        # Resolve the freshness tri-state before the empty-pool exit, so a
        # measurement that failed purely because the RTM had nothing new is
        # distinguishable from one that failed on comms.
        if confirmed_grabs and not unconfirmed_grabs:
            self._cb_last_frame_fresh = True
        elif no_new_data and not confirmed_grabs:
            self._cb_last_frame_fresh = False
        else:
            self._cb_last_frame_fresh = None

        if not pooled:
            return None
        pixels = np.concatenate(pooled)
        # Stale-frame discriminator: across a CB change, a bit-identical pooled
        # mean means the RTM buffer did not refresh -- regardless of what the
        # SDK freshness flag claimed (2026-08-27: "fresh" data sets carried
        # frozen content). It is still only a suggestion on its own -- a
        # rail-dominated scene produces genuinely identical frames -- so the
        # calibration closure gates it on the previous balance, not on the
        # mean alone.
        self._cb_last_pooled_mean = float(pixels.mean())
        logger.debug(
            f"Auto CB measure: pooled_px={pixels.size}, "
            f"mean={self._cb_last_pooled_mean:.3f}, fresh={self._cb_last_frame_fresh}"
        )
        wl = self.parameters.cb_white_level if white_level is None else white_level
        balance = compute_cb_balance(
            pixels,
            wl,
            contrast_percentile=self.CB_CONTRAST_PERCENTILE,
        )
        self._cb_last_balance = balance
        return balance

    def _resolve_white_level(self) -> float:
        """
        Determine the detector full-scale (raw-count ceiling) for clip detection.

        RTM low-resolution rides the imaging pipeline, so full-scale is
        ``2**scanning.bit_depth - 1`` -- not fixed at 255 by the RTM mode. Prefer the
        actual bit depth; if it can't be read, infer from an observed sample (bounded
        below by the configured cb_white_level fallback so a dim scene never under-
        scales). This prevents the silent failure where 16-bit data against a 255
        ceiling reads as entirely white-clipped.
        """
        bits = read_scanning_bit_depth(self.microscope)
        if bits and int(bits) > 0:
            fs = float(2 ** int(bits) - 1)
            logger.info(f"Auto CB: white_level from scanning bit_depth={int(bits)} -> {fs:.0f}")
            return fs

        fallback = float(self.parameters.cb_white_level)
        sample = self._measure_cb_balance(white_level=max(fallback, 65535.0))
        if sample is not None and sample.vmax > 0:
            inferred = float(2 ** int(math.ceil(math.log2(sample.vmax + 1.0))) - 1)
            fs = max(fallback, inferred)
            logger.info(
                f"Auto CB: scanning bit_depth unavailable; inferred white_level {fs:.0f} "
                f"from observed max {sample.vmax:.0f} (fallback {fallback:.0f})"
            )
            return fs

        logger.info(f"Auto CB: white_level fallback {fallback:.0f} (no bit depth, no sample)")
        return fallback

    def _calibrate_detector_cb(self) -> bool:
        """
        One-shot detector contrast/brightness calibration (gated by
        auto_cb_on_start), called exactly once per patterning session.

        Always fail-open: on any skip or *Python* error it emits a visible
        warning and returns True so monitoring continues with whatever CB is set.

        NOTE: a native crash inside the AutoScript C++ binding cannot be caught
        here (it bypasses Python entirely). The bracketed INFO logs in this path
        and in read_detector_cb / set_detector_cb / restart_rtm exist so the last
        console line before a silent crash pinpoints the offending call.
        """
        if not self.parameters.auto_cb_on_start:
            return True

        if self.microscope is None:
            self.warning_status_update.emit("Auto CB: skipped - microscope unavailable")
            logger.warning("Auto CB skipped: microscope is None")
            self._cb_summary = {"status": "skipped-no-microscope", "converged": False}
            return True

        try:
            self._run_cb_calibration_loop()
        except Exception:
            logger.error("Auto CB failed with a Python exception - continuing", exc_info=True)
            self.warning_status_update.emit("Auto CB: error - continuing without calibration")
            self._cb_summary = {"status": "error", "converged": False}
        return True

    def _run_cb_calibration_loop(self):
        """
        One-shot CB calibration body (see _calibrate_detector_cb).

        Auto-sizes the detector full-scale, seeds the edge-margin correction
        with anything learned about this tool in previous runs (cb_plant.json),
        then runs rtm_data_processing.run_cb_correction: search measurements at
        1 pooled frame, acceptance verified at cb_frames_per_measurement frames.
        The chosen (accepted / best-so-far) CB is committed and held static for
        the session, and what the run learned about the tool's response is
        persisted for the next run.
        """
        logger.info("Auto CB: enabled - reading current detector contrast/brightness")
        contrast, brightness = read_detector_cb(self.microscope)
        if contrast is None or brightness is None:
            self.warning_status_update.emit("Auto CB: skipped - could not read detector contrast/brightness")
            logger.warning("Auto CB skipped: detector CB read failed")
            self._cb_summary = {"status": "skipped-cb-read", "converged": False}
            return

        # A new calibration retires the previous run's standing advisory
        # (field decision, 2026-08-27): whatever it warned about is being
        # re-measured right now, and this attempt will post its own outcome.
        self.warning_status_update.emit("")
        self.status_update.emit("Auto CB: calibrating contrast/brightness...")
        start_contrast, start_brightness = contrast, brightness

        white_level = self._resolve_white_level()
        cfg = CBEdgeConfig(
            white_level=white_level,
            margin_lo=float(self.parameters.cb_lower_margin),
            margin_hi=float(self.parameters.cb_upper_margin),
            max_white_clip=float(self.parameters.cb_max_white_clip_fraction),
            max_black_clip=float(self.parameters.cb_max_black_clip_fraction),
            min_bound=float(self.parameters.cb_min_bound),
            max_bound=float(self.parameters.cb_max_bound),
        )

        # Per-tool plant seeding: the knob slopes are plant properties
        # (independent of the scene), so a previous run on the same
        # microscope is a legitimate head start. A missing or invalid seed
        # is safe: the correction falls back to its sign-correct defaults
        # and re-learns from its own measurements (a non-positive persisted
        # slope is rejected at seeding -- the sign is physics). Persisted
        # seeds are deliberately never clamped toward the defaults: a clamp
        # cannot distinguish a stale harsh seed from a legitimate
        # gentle-tool slope. A wrong seed self-heals instead -- a
        # gentle-seeded solve that lands on a rail is slope evidence
        # (run_cb_correction's _rail_evidence raises the slope on the
        # spot and marks it learned, so this very seeding path overwrites
        # the stale value on the next run).
        system_name = get_system_name(self.microscope) or "unknown"
        learned = self._load_cb_plant(system_name)
        init_k_b = init_k_c = None
        if learned:
            init_k_b = self._as_float_or_none(learned.get("k_brightness"))
            init_k_c = self._as_float_or_none(learned.get("k_contrast"))
            logger.info(
                f"Auto CB: seeded from learned plant for '{system_name}': "
                f"k_brightness={init_k_b}, k_contrast={init_k_c}"
            )

        # One RTM restart per calibration (flushes the pre-calibration buffer);
        # per-measurement staleness is handled by the re-grab guard below.
        restart_rtm(self.microscope)
        cb_limits = read_detector_cb_limits(self.microscope)

        first = {"done": False}
        held = {"c": None, "b": None}
        self._cb_trace = []
        self._cb_last_pooled_mean = None
        self._cb_last_balance = None
        self._cb_last_frame_fresh = None

        def _trace_row(c_app, b_app, frames, tag=None):
            """Append a telemetry row so cb_trace.csv rows stay 1:1 with the
            controller's measurement count (failed writes and stop-requests
            included -- the review found those desynchronized the CSV)."""
            row = {"n": len(self._cb_trace) + 1, "tag": tag, "frames": frames,
                   "contrast": round(c_app, 4) if c_app is not None else None,
                   "brightness": round(b_app, 4) if b_app is not None else None}
            self._cb_trace.append(row)
            return row

        def _apply_and_measure(c, b, frames):
            """Apply CB (skipping unchanged channels), settle, re-image."""
            if self._stop_requested:
                _trace_row(c, b, frames, tag="stopped")
                return None, c, b
            c_req = c if (held["c"] is None or abs(c - held["c"]) > 1e-9) else None
            b_req = b if (held["b"] is None or abs(b - held["b"]) > 1e-9) else None
            if c_req is not None or b_req is not None:
                ok, c_after, b_after, msg = set_detector_cb(
                    self.microscope, contrast=c_req, brightness=b_req,
                    bounds=(cfg.min_bound, cfg.max_bound), limits=cb_limits,
                )
                c_app = c_after if c_after is not None else c
                b_app = b_after if b_after is not None else b
                if not ok:
                    self.persistent_status_update.emit(f"Auto CB: warning - {msg}; continuing")
                    logger.warning(f"Auto CB write failed: {msg}")
                    _trace_row(c_app, b_app, frames, tag="write-failed")
                    return None, c_app, b_app
            else:
                c_app, b_app = held["c"], held["b"]
            held["c"], held["b"] = c_app, b_app
            if self.CB_RESTART_RTM_EACH_MEASUREMENT:
                # Rollback mode: the restart flushes the buffer, so restore the
                # pre-3.3.1 robustness that made a post-restart measurement
                # survivable (longer settle + multi-frame pooling with the
                # empty-grab retries in _measure_cb_balance).
                restart_rtm(self.microscope)
                time.sleep(max(self._cb_settle_seconds(), 0.5))
                frames = max(frames, 3)
            else:
                # Settle at least one RTM frame period. The operator's setting stays
                # authoritative as a floored value, not verbatim: cb_settle_seconds
                # defaults to 0.2 s while the measured frame period is ~0.22 s, so
                # the verbatim setting guaranteed the first grab returned a frame
                # acquired before the write landed (2026-08-23).
                time.sleep(self._cb_settle_seconds())
            prev_mean = self._cb_last_pooled_mean
            prev_balance = self._cb_last_balance
            knob_changed = c_req is not None or b_req is not None
            bal = self._measure_cb_balance(white_level=cfg.white_level, frames=frames)
            if knob_changed and self._frames_frozen(bal, prev_mean, prev_balance, cfg):
                # The RTM buffer looks frozen. Retry a few times, one acquisition
                # period apart; if it never advances, fail the measurement rather
                # than ingest frozen data -- the 2026-08-18 episode fed five
                # identical stats into the controller while brightness walked
                # 0.404 -> 0.000.
                retries = 0
                while (retries < 3
                       and self._frames_frozen(bal, prev_mean, prev_balance, cfg)):
                    logger.debug("Auto CB: stale RTM frame suspected - re-grabbing")
                    time.sleep(max(self.parameters.acquisition_delay_seconds,
                                   self._rtm_frame_period()))
                    bal = self._measure_cb_balance(white_level=cfg.white_level,
                                                   frames=frames)
                    retries += 1
                if self._frames_frozen(bal, prev_mean, prev_balance, cfg):
                    logger.warning(
                        "Auto CB: RTM frames not advancing across a CB change "
                        "- measurement rejected (stale buffer)"
                    )
                    _trace_row(c_app, b_app, frames, tag="stale-frames")
                    return None, c_app, b_app
            if bal is not None and not first["done"]:
                first["done"] = True
                # The start point's own reading, kept for the commit-check
                # revert decision: reverting to start must actually improve on
                # the drifted lock, and this is the start's evidence. The first
                # successful measurement is always at the start knobs -- the
                # controller's baseline take (retried at the same point on
                # failure) precedes every move, and a run whose baseline never
                # measured ends no-measure before any commit.
                first["start_cost"] = cb_edge_cost(bal, cfg)
                if bal.vmax > cfg.white_level * 1.01:
                    logger.warning(
                        f"Auto CB: observed max {bal.vmax:.0f} exceeds white_level "
                        f"{cfg.white_level:.0f}; clip detection may be wrong"
                    )
                logger.info(
                    f"Auto CB start: contrast={c_app:.3f}, brightness={b_app:.3f}, "
                    f"valid_px={bal.n_valid}, min={bal.vmin:.1f}, max={bal.vmax:.1f}, "
                    f"white_clip={bal.white_clip:.3f}, black_clip={bal.black_clip:.3f}, "
                    f"median_frac={bal.median_fraction:.3f}, span_frac={bal.span_fraction:.3f} "
                    f"(white_level={cfg.white_level:.0f})"
                )
            row = _trace_row(c_app, b_app, frames,
                             tag=None if bal is not None else "failed")
            if bal is not None:
                row.update(median=round(bal.median_fraction, 4),
                           span=round(bal.span_fraction, 4),
                           # The edges the controller actually steers; the
                           # 2026-08-27 analysis had to reconstruct them from
                           # the log's note lines.
                           p2=(round(bal.p2_fraction, 4)
                               if bal.p2_fraction is not None else None),
                           p98=(round(bal.p98_fraction, 4)
                                if bal.p98_fraction is not None else None),
                           wclip=round(bal.white_clip, 4),
                           bclip=round(bal.black_clip, 4),
                           cost=round(cb_edge_cost(bal, cfg), 4))
            return bal, c_app, b_app

        def measure_at(c, b):
            return _apply_and_measure(c, b, 1)

        def verify_at(c, b):
            return _apply_and_measure(c, b, self.parameters.cb_frames_per_measurement)

        def note_cb(msg):
            # The controller's note lines are "Auto CB {tag}: ..." -- reuse the
            # tag for the telemetry trace row the measurement just appended.
            logger.info(msg)
            if self._cb_trace and self._cb_trace[-1]["tag"] is None:
                head = msg.split(":", 1)[0]
                if head.startswith("Auto CB "):
                    self._cb_trace[-1]["tag"] = head[len("Auto CB "):].strip()

        def recover_rtm():
            """Repair hook for a failed measurement: restart the RTM (the SDK
            clears all its data on restart, so this is the strongest available
            un-wedge) and settle before the controller re-reads the point."""
            if self._stop_requested:
                return
            logger.info("Auto CB: measurement failed - restarting the RTM and "
                        "re-measuring")
            ok_r, msg_r = restart_rtm(self.microscope)
            if not ok_r:
                logger.warning(f"Auto CB: recovery RTM restart failed: {msg_r}")
            # A restart empties the buffer; wait a frame before re-grabbing so
            # the re-measure is not just the empty-buffer answer again.
            time.sleep(self._cb_settle_seconds())
            # The knobs are unchanged by the restart, but the buffer is not the
            # one the discriminator last saw -- drop the stale comparison basis.
            self._cb_last_pooled_mean = None
            self._cb_last_balance = None

        budget = max(1, int(self.parameters.cb_max_iterations))
        result = run_cb_correction(
            measure_at, contrast, brightness, cfg, budget,
            note=note_cb, verify_at=verify_at, recover=recover_rtm,
            init_k_contrast=init_k_c, init_k_brightness=init_k_b,
        )

        if result.n_measurements == 0 or result.status == "no-measure":
            self.warning_status_update.emit(
                "Auto CB: skipped - no usable RTM frame to measure"
            )
            logger.warning("Auto CB: no valid pixels to measure - aborting calibration")
            self._cb_summary = {"status": "no-measure", "converged": False}
            return

        # Commit the chosen (accepted / best-so-far) CB and re-image clean.
        ok, c_after, b_after, _msg = set_detector_cb(
            self.microscope, contrast=result.contrast, brightness=result.brightness,
            bounds=(cfg.min_bound, cfg.max_bound), limits=cb_limits,
        )
        if not ok:
            # One retry -- then report honestly: the hardware is wherever the
            # read-back says it is, not at the chosen point. (The review found
            # a failed commit was reported as a successful lock everywhere.)
            ok, c_after, b_after, _msg = set_detector_cb(
                self.microscope, contrast=result.contrast,
                brightness=result.brightness,
                bounds=(cfg.min_bound, cfg.max_bound), limits=cb_limits,
            )
        commit_ok = bool(ok)
        if commit_ok:
            contrast = c_after if c_after is not None else result.contrast
            brightness = b_after if b_after is not None else result.brightness
        else:
            # Read-back is returned even on a failed write; prefer the truth.
            contrast = c_after if c_after is not None else result.contrast
            brightness = b_after if b_after is not None else result.brightness
            self.warning_status_update.emit(
                "Auto CB: final CB write failed - detector left at "
                "its last-applied settings"
            )
            logger.warning(
                f"Auto CB: final commit write failed; hardware at "
                f"contrast={contrast:.3f}, brightness={brightness:.3f} instead "
                f"of chosen ({result.contrast:.3f}, {result.brightness:.3f})"
            )
        restart_rtm(self.microscope)
        time.sleep(self._cb_settle_seconds())

        # Commit-time verification: every lock (converged or best-so-far) gets
        # one pooled re-measure at its knobs -- search/verify measurements can
        # be noisy or stale-attributed, and a converged lock that ships
        # unchecked is how the 07:58 2026-08-20 dark lock went out with no
        # backstop. If the re-measure is actionably clipped, fall back to the
        # best clip-clean point the search measured (reverting to the start
        # discards known-good ground; the clean best is never worse) -- start
        # remains the fallback when no distinct clean point exists, and only
        # when its own baseline reading beats the failed check (the Run-12
        # rule in the revert decision below).
        commit_check = None
        commit_check_fallback = None
        commit_failed = False
        reverted = False
        reverted_to = None

        def _balance_dict(bal):
            return {
                "median": round(bal.median_fraction, 4),
                "span": round(bal.span_fraction, 4),
                "wclip": round(bal.white_clip, 4),
                "bclip": round(bal.black_clip, 4),
                "cost": round(cb_edge_cost(bal, cfg), 4),
            }

        def _pooled_check():
            """One pooled measurement for commit verification. Returns None when
            unusable -- a genuinely frozen RTM buffer -- because acting on
            phantom evidence would revert a good lock.

            Every caller re-measures immediately after an RTM restart, and the
            SDK guarantees restart() clears all RTM data, so the buffer is fresh
            by contract. That makes the pooled-mean comparison unsound here in a
            way it is not in the search: the check re-measures at the same knobs
            as the last measurement, so on a static scene it reproduces that
            mean by construction. Run 1 of 2026-08-23 hit exactly this and every
            such lock shipped unverified. Only the SDK's own verdict is trusted
            at this point."""
            check = self._measure_cb_balance(
                white_level=cfg.white_level,
                frames=self.parameters.cb_frames_per_measurement)
            if self._cb_last_frame_fresh is False:
                logger.warning(
                    "Auto CB: commit check unusable - RTM reports no new data; "
                    "keeping the current setting unverified"
                )
                return None
            return check

        def _write_cb_retry(c_target, b_target):
            """CB write with the same one-shot retry as the commit write."""
            ok_w, c_w, b_w, _m = set_detector_cb(
                self.microscope, contrast=c_target, brightness=b_target,
                bounds=(cfg.min_bound, cfg.max_bound), limits=cb_limits,
            )
            if not ok_w:
                ok_w, c_w, b_w, _m = set_detector_cb(
                    self.microscope, contrast=c_target, brightness=b_target,
                    bounds=(cfg.min_bound, cfg.max_bound), limits=cb_limits,
                )
            return (ok_w,
                    c_w if c_w is not None else c_target,
                    b_w if b_w is not None else b_target)

        def _restart_and_settle():
            ok_r, msg_r = restart_rtm(self.microscope)
            if not ok_r:
                logger.warning(f"Auto CB: RTM restart failed: {msg_r}")
            time.sleep(self._cb_settle_seconds())

        if commit_ok and result.n_measurements > 0:
            check = _pooled_check()
            if check is not None:
                commit_check = _balance_dict(check)
                if cb_clip_actionable(check, cfg):
                    commit_failed = True
                    logger.warning(
                        f"Auto CB: locked point measured actionably clipped at "
                        f"commit (wclip={check.white_clip:.3f}, "
                        f"bclip={check.black_clip:.3f})"
                    )
                    # A revert must actually improve on the drifted lock. The
                    # start is a rescue only when its own baseline reading
                    # beat this check: field Run-12 (2026-08-27) verified a
                    # clean lock, the scene drifted to bclip 0.110 by the
                    # commit check, and the unconditional revert put the
                    # detector back onto the wclip-0.967 blown-out start --
                    # strictly worse than the drifted lock it replaced. The
                    # baseline is a 1-frame reading from calibration start, so
                    # it carries drift risk of its own -- but the decision gap
                    # it exists to close (0.31 vs 2.62 edge cost in Run-12)
                    # dwarfs that noise.
                    check_cost = cb_edge_cost(check, cfg)
                    start_cost = first.get("start_cost")
                    start_viable = (start_cost is not None
                                    and start_cost < check_cost)
                    lock_is_start = (
                        abs(result.contrast - start_contrast) <= 1e-6
                        and abs(result.brightness - start_brightness) <= 1e-6)
                    best_clean = result.best_clean
                    if (best_clean is not None
                            and (abs(best_clean[0] - result.contrast) > 1e-6
                                 or abs(best_clean[1] - result.brightness) > 1e-6)):
                        revert_c, revert_b = best_clean
                        reverted_to = "best-clean"
                    elif start_viable and not lock_is_start:
                        revert_c, revert_b = start_contrast, start_brightness
                        reverted_to = "start"
                    else:
                        reverted_to = None
                    if reverted_to is None:
                        # No fallback improves on the drifted lock: either the
                        # lock is itself the start (nothing to actuate), or the
                        # start's own reading was no better than the failed
                        # check. Say so honestly instead of reporting a no-op
                        # -- or worse-making -- mitigation.
                        if lock_is_start or start_cost is None:
                            logger.warning(
                                "Auto CB: no measured fallback available - "
                                "continuing at the locked (= starting) setting"
                            )
                            self.warning_status_update.emit(
                                "Auto CB: calibrated setting failed verification "
                                "- no measured fallback available; continuing "
                                "with it"
                            )
                        else:
                            logger.warning(
                                f"Auto CB: keeping the drifted lock - the only "
                                f"fallback (start, cost {start_cost:.3f}) "
                                f"measured worse than the commit check "
                                f"({check_cost:.3f})"
                            )
                            self.warning_status_update.emit(
                                "Auto CB: calibrated setting failed "
                                "verification - keeping it; the start "
                                "measured worse"
                            )
                    else:
                        ok2, c2, b2 = _write_cb_retry(revert_c, revert_b)
                        if not ok2:
                            # Honest failure: the detector stays wherever the
                            # read-back says; do not claim the revert happened.
                            reverted_to = None
                            contrast, brightness = c2, b2
                            logger.warning(
                                "Auto CB: revert write failed - detector left "
                                f"at contrast={contrast:.3f}, "
                                f"brightness={brightness:.3f}"
                            )
                            self.warning_status_update.emit(
                                "Auto CB: verification failed and the "
                                "fallback write failed - detector left at "
                                "its last-applied settings"
                            )
                        else:
                            reverted = True
                            contrast, brightness = c2, b2
                            _restart_and_settle()
                            fallback_kept_unverified = False
                            if reverted_to == "best-clean":
                                # The fallback's cleanliness rests on a single
                                # search reading -- the evidence class the
                                # commit check exists to distrust. Verify it;
                                # chain to the starting CB if it fails too --
                                # but only when the start's own reading beats
                                # what the fallback just measured (the Run-12
                                # rule again: never chain onto known-worse
                                # ground).
                                check2 = _pooled_check()
                                if check2 is not None:
                                    commit_check_fallback = _balance_dict(check2)
                                if (check2 is None
                                        or cb_clip_actionable(check2, cfg)):
                                    chain_cost = (cb_edge_cost(check2, cfg)
                                                  if check2 is not None
                                                  else check_cost)
                                    if (start_cost is not None
                                            and start_cost < chain_cost):
                                        logger.warning(
                                            "Auto CB: best clip-clean fallback "
                                            "failed its own check - chaining to "
                                            "the starting CB"
                                        )
                                        reverted_to = "start"
                                        ok3, c3, b3 = _write_cb_retry(
                                            start_contrast, start_brightness)
                                        if ok3:
                                            contrast, brightness = c3, b3
                                            _restart_and_settle()
                                        else:
                                            contrast, brightness = c3, b3
                                            logger.warning(
                                                "Auto CB: chain-to-start write "
                                                "failed - detector left at "
                                                f"contrast={contrast:.3f}, "
                                                f"brightness={brightness:.3f}"
                                            )
                                    else:
                                        fallback_kept_unverified = True
                                        logger.warning(
                                            "Auto CB: best clip-clean fallback "
                                            "failed its own check, but the "
                                            "start measured no better - "
                                            "keeping the fallback"
                                        )
                            self.warning_status_update.emit(
                                "Auto CB: calibrated setting failed "
                                "verification - "
                                + ("keeping the best measured setting"
                                   if fallback_kept_unverified
                                   else "keeping the verified clip-clean setting"
                                   if reverted_to == "best-clean"
                                   else "keeping the starting "
                                        "contrast/brightness")
                            )

        def _locked_clips():
            """Clip fractions measured at the locked knobs: the commit check
            when it ran, else this run's own trace row for that point. None when
            the locked point was never successfully read."""
            if commit_check is not None:
                return commit_check["wclip"], commit_check["bclip"]
            for row in reversed(self._cb_trace):
                if (row.get("wclip") is not None
                        and row.get("contrast") is not None
                        and row.get("brightness") is not None
                        and abs(row["contrast"] - contrast) <= 1e-3
                        and abs(row["brightness"] - brightness) <= 1e-3):
                    return row["wclip"], row["bclip"]
            return None

        if result.converged and commit_ok and not commit_failed:
            self.status_update.emit("Auto CB: contrast/brightness calibrated")
        elif commit_ok and not commit_failed:
            # Name the actual fault when the setting we locked is clipped. The
            # generic non-convergence line hid a 99.1%-white-clipped detector
            # for a whole milling run on 2026-08-23; a saturated image destroys
            # the top-hat texture the match score is built on, so the operator
            # has to be told what is wrong, not merely that it is.
            clips = _locked_clips()
            if clips is not None and cb_clips_actionable(clips[0], clips[1], cfg):
                w_clip, b_clip = clips
                rail = "white" if w_clip >= b_clip else "black"
                worst = max(w_clip, b_clip)
                # Only call it saturated when a rail actually dominates the
                # frame; a few percent over the limit is clipping, not a lost
                # image, and overstating it would train the operator to ignore
                # the warning.
                detail = ("the image is saturated and match scores are "
                          "unreliable" if cb_rail_dominated_clips(w_clip, b_clip)
                          else "match scores may suffer")
                self.warning_status_update.emit(
                    f"Auto CB: did not converge - holding a setting with "
                    f"{round(worst * 100)}% {rail} clipping; {detail}"
                )
            else:
                self.warning_status_update.emit(
                    "Auto CB: did not converge - holding the best setting found"
                )
        logger.info(
            f"Auto CB locked: contrast {start_contrast:.3f}->{contrast:.3f}, "
            f"brightness {start_brightness:.3f}->{brightness:.3f}, "
            f"converged={result.converged}, measurements={result.n_measurements}, "
            f"cost={result.cost:.3f}, status={result.status}, "
            f"commit_ok={commit_ok}, reverted={reverted}"
        )
        self._cb_summary = {
            "status": result.status,
            "converged": result.converged,
            "n_measurements": result.n_measurements,
            "cost": round(float(result.cost), 4),
            "start_contrast": round(start_contrast, 4),
            "start_brightness": round(start_brightness, 4),
            "final_contrast": round(contrast, 4),
            "final_brightness": round(brightness, 4),
            "white_level": white_level,
            # The settings this calibration actually ran against, snapshotted
            # from cfg and the locals it consumed. run_metadata's
            # processing_parameters block reads self.parameters at finalize
            # time, so once the CB settings became editable mid-run it could
            # report a value the calibration never saw -- with nothing in the
            # record to contradict it. These are the redundant copy that makes
            # the run self-describing (the block exists because the campaign
            # that motivated it was reconstructable only by log forensics).
            "cb_lower_margin": round(cfg.margin_lo, 4),
            "cb_upper_margin": round(cfg.margin_hi, 4),
            "max_white_clip": round(cfg.max_white_clip, 4),
            "max_black_clip": round(cfg.max_black_clip, 4),
            "cb_min_bound": round(cfg.min_bound, 4),
            "cb_max_bound": round(cfg.max_bound, 4),
            "cb_max_iterations": budget,
            "cb_settle_seconds": round(
                float(self.parameters.cb_settle_seconds), 4),
            "system_name": system_name,
            "commit_ok": commit_ok,
            "commit_check": commit_check,
            "commit_check_fallback": commit_check_fallback,
            "commit_failed": commit_failed,
            "reverted_to_start": reverted and reverted_to == "start",
            "reverted_to": reverted_to if reverted else None,
        }
        # A run that barely read the tool has not learned anything about it.
        # 2026-08-23 run 4 aborted after 2 measurements and still rewrote the
        # plant (walk_seed 0.020 -> 0.0466, plus a c_window bracketed by a
        # single saturated jump), seeding every later run from a failure.
        # startswith is defensive: today the measure-failed override replaces
        # the status outright (no "-infeasible" suffix survives it), so this
        # is equivalent to equality.
        if (result.status.startswith("measure-failed")
                and result.n_measurements < 3):
            logger.info(
                f"Auto CB: learned plant not persisted for '{system_name}' - "
                f"the run ended {result.status} after "
                f"{result.n_measurements} measurement(s)"
            )
        else:
            self._persist_cb_plant(system_name, result.plant)

    @staticmethod
    def _as_float_or_none(value):
        """Coerce to a finite float or None. json happily round-trips NaN and
        Infinity, and a non-finite slope seed makes every solve step degenerate
        (a persisted Infinity would silently disable calibration on that tool
        forever -- review finding, 2026-08-15)."""
        try:
            f = None if value is None else float(value)
        except (TypeError, ValueError):
            return None
        return f if (f is not None and math.isfinite(f)) else None

    def _cb_plant_path(self) -> Path:
        """
        Learned-plant file. Lives beside the script since v3.3.2 (travels with
        the deployment and across Windows accounts -- %LOCALAPPDATA% is
        per-account, so learned state used to vanish when a different operator
        logged in). Entries are keyed by system name, so multiple tools
        sharing one script directory share the file safely; the concurrent
        write window (read-merge-write + atomic replace in _persist_cb_plant)
        is seconds against a rare event -- accepted.
        """
        if self.parameters.script_root:
            # In logs/ since v3.3.8: app-managed state does not belong loose
            # in the deployment root the operator browses.
            return Path(self.parameters.script_root) / "logs" / "cb_plant.json"
        return self._legacy_cb_plant_path()

    def _script_root_legacy_cb_plant_path(self):
        """v3.3.2-3.3.7 location (<script root>/cb_plant.json). Still read as
        a fallback, and migrated (entries carried over, file removed) on the
        first successful v3.3.8 persist; never written anymore."""
        if self.parameters.script_root:
            return Path(self.parameters.script_root) / "cb_plant.json"
        return None

    @staticmethod
    def _legacy_cb_plant_path() -> Path:
        """Pre-3.3.2 location (%LOCALAPPDATA%/ATC_Monitor/cb_plant.json).
        Still read as a seed fallback so plants learned under 3.3.1 (e.g. the
        2026-08-14 field campaign) survive the move; never written anymore."""
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "ATC_Monitor" / "cb_plant.json"

    def _load_cb_plant(self, system_name: str):
        """Learned per-tool plant model, or None. Never raises.

        Reads the logs-dir file first; if it has no entry for this system,
        falls back to the pre-3.3.8 script-root file, then the legacy
        local-app-data file (one-way migration: the next _persist_cb_plant
        writes to the logs-dir file and retires the root file)."""
        candidates = (self._cb_plant_path(),
                      self._script_root_legacy_cb_plant_path(),
                      self._legacy_cb_plant_path())
        for path in (p for p in candidates if p is not None):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = data.get(system_name)
                if isinstance(entry, dict):
                    return entry
            except Exception:
                continue
        return None

    def _persist_cb_plant(self, system_name: str, plant):
        """
        Persist what this run learned about the tool's response. Never raises.

        The correction reports only slopes it measured this run (a seed is
        never echoed back), so a None here means "nothing new" -- the previous
        entry's value is kept rather than erased. Old-format entries
        (walk_seed / c_window, pre-v3.3.16) are read tolerantly and rewritten
        in the new shape.
        """
        try:
            plant = plant or {}
            previous = self._load_cb_plant(system_name) or {}
            k_b = self._as_float_or_none(plant.get("k_brightness"))
            k_c = self._as_float_or_none(plant.get("k_contrast"))
            # A contradicted (measured-wrong-sign) seed is poison: drop the
            # stored value instead of preserving it through the merge.
            k_b_previous = (None if plant.get("k_brightness_contradicted")
                            else self._as_float_or_none(
                                previous.get("k_brightness")))
            k_c_previous = (None if plant.get("k_contrast_contradicted")
                            else self._as_float_or_none(
                                previous.get("k_contrast")))
            entry = {
                "k_brightness": (k_b if k_b is not None else k_b_previous),
                "k_contrast": (k_c if k_c is not None else k_c_previous),
                "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            path = self._cb_plant_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if not isinstance(data, dict):
                data = {}
            # Merge the pre-3.3.8 script-root file unconditionally (not just
            # when the logs file is missing): several tools can share one
            # deployment, and a root file resurrected by a version rollback or
            # a previously-failed unlink may hold entries the logs file never
            # saw. Logs-file entries win (matching _load_cb_plant's read
            # order); the unlink below only fires when the root was actually
            # read this call, so it can never delete content it did not merge.
            root_merged = False
            old_root = self._script_root_legacy_cb_plant_path()
            if old_root is not None:
                try:
                    migrated = json.loads(old_root.read_text(encoding="utf-8"))
                    if isinstance(migrated, dict):
                        for key, value in migrated.items():
                            data.setdefault(key, value)
                        root_merged = True
                except Exception:
                    pass
            data[system_name] = entry
            # Atomic replace (write-then-rename) so a crash or a second app
            # instance mid-write can never leave a truncated file, and
            # allow_nan=False so a non-finite value can never be serialized
            # (it would round-trip as literal Infinity and poison every
            # subsequent run's seed).
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(data, indent=2, allow_nan=False),
                                encoding="utf-8")
            os.replace(tmp_path, path)
            # Retire the superseded root-level file so the deployment root
            # stays clean -- but only when this call actually read and merged
            # it (a persist must never delete root content it did not see).
            try:
                if (root_merged and old_root is not None and old_root != path
                        and old_root.exists()):
                    old_root.unlink()
            except Exception:
                pass
            logger.info(f"Auto CB: learned plant persisted for '{system_name}': {entry}")
        except Exception:
            logger.warning("Auto CB: could not persist learned plant", exc_info=True)

    def _compute_clamped_window(self, desired_interval: float) -> tuple[int, int, bool]:
        """
        Compute clamped window size from desired interval and calibrated image rate.

        :param desired_interval: Desired analysis interval in seconds
        :return: (clamped_window, actual_interval_seconds, was_clamped)
        """
        raw_window = round(self._images_per_second * desired_interval)
        clamped = max(
            self.parameters.min_window_size,
            min(self.parameters.max_window_size, raw_window)
        )
        actual = max(1, round(clamped / self._images_per_second))
        return clamped, actual, clamped != raw_window

    def _log_clamped_window(
        self, desired_interval: float, clamped_window: int, was_clamped: bool
    ):
        """Log a clamping warning if the window was clamped to bounds."""
        if was_clamped:
            raw_window = round(self._images_per_second * desired_interval)
            logger.info(
                f"Requested {desired_interval}s interval clamped: "
                f"raw window={raw_window} -> clamped to {clamped_window} "
                f"(bounds: {self.parameters.min_window_size}-{self.parameters.max_window_size})"
            )

    def _calibrate_image_rate(self):
        """
        Calibrate image arrival rate and calculate window size for TIME_BASED mode.

        Called after each successful image acquisition until calibration is complete.
        Once calibrated, updates number_of_images and emits calibration_complete signal.
        """
        # Already calibrated - nothing to do
        if self._rate_calibrated:
            return

        self._image_timestamps.append(time.time())

        # Need enough samples to calculate a reliable rate
        if len(self._image_timestamps) < self.parameters.calibration_sample_count:
            return

        # Calculate average interval between consecutive images
        intervals = np.diff(self._image_timestamps)
        avg_interval = np.mean(intervals)
        self._images_per_second = 1.0 / avg_interval

        # Calculate and apply window size
        desired_interval = self.parameters.analysis_interval_seconds
        clamped_window, actual_interval, was_clamped = (
            self._compute_clamped_window(desired_interval)
        )

        self.parameters.number_of_images = clamped_window
        self._rate_calibrated = True

        # Free memory - timestamps no longer needed
        self._image_timestamps.clear()

        logger.info(
            f"Image rate calibrated: {self._images_per_second:.2f} img/s "
            f"(avg interval: {avg_interval:.3f}s) -> "
            f"window={clamped_window} images ~ {actual_interval}s"
        )
        self._log_clamped_window(desired_interval, clamped_window, was_clamped)

        # Notify GUI. Outcome only (field decision, 2026-08-27): the operator
        # acts on the batch size, not the raw rate -- which the calibration
        # log line above already records.
        self.calibration_complete.emit(actual_interval, was_clamped)
        self.status_update.emit(
            f"Analyzing {clamped_window} images per batch (~{actual_interval}s)"
        )

    def _recalculate_window_size(self):
        """
        Recalculate window size after analysis interval changes during monitoring.

        Uses the previously calibrated image rate to compute a new number_of_images
        from the updated analysis_interval_seconds. Only runs when in TIME_BASED
        mode and the rate has already been calibrated.

        This enables live updates to the analysis interval without requiring
        a full recalibration.
        """
        if self.parameters.window_mode != WindowMode.TIME_BASED:
            return

        if not self._rate_calibrated or self._images_per_second <= 0:
            return

        desired_interval = self.parameters.analysis_interval_seconds
        clamped_window, actual_interval, was_clamped = (
            self._compute_clamped_window(desired_interval)
        )

        old_window = self.parameters.number_of_images
        if clamped_window == old_window:
            return  # No change needed

        # Apply updated window size
        self.parameters.number_of_images = clamped_window

        logger.info(
            f"Window size recalculated: {old_window} -> {clamped_window} images "
            f"(interval={desired_interval}s, rate={self._images_per_second:.2f} img/s, "
            f"actual~{actual_interval}s)"
        )
        self._log_clamped_window(desired_interval, clamped_window, was_clamped)

        # Notify GUI of updated interval. Named after the control the
        # operator just changed, outcome only (the log line above keeps the
        # rate and window arithmetic).
        self.calibration_complete.emit(actual_interval, was_clamped)
        self.status_update.emit(
            f"Analysis Interval updated - now analyzing {clamped_window} "
            f"images per batch (~{actual_interval}s)"
        )

    def _initialize_pattern_states(self, pattern_infos):
        """Initialize state tracking for each pattern."""
        # Clear existing states
        self.pattern_states = []
        self._pattern_metadata = None

        for i, pattern_info in enumerate(pattern_infos):
            # Determine which crop rect and thresholds to use
            if i == 0:
                crop_rect = self.parameters.pattern_1_crop_rect
                mean_slope_thresh = self.parameters.pattern_1_mean_slope_threshold
                match_score_thresh = self.parameters.pattern_1_match_score_threshold
                max_pixels_thresh = self.parameters.pattern_1_max_pixels_threshold
            else:
                crop_rect = self.parameters.pattern_2_crop_rect
                mean_slope_thresh = self.parameters.pattern_2_mean_slope_threshold
                match_score_thresh = self.parameters.pattern_2_match_score_threshold
                max_pixels_thresh = self.parameters.pattern_2_max_pixels_threshold

            pattern_state = PatternState(
                pattern_index=i,
                pattern_id=pattern_info.pattern_id,
                crop_rect=crop_rect,
                crop_rect_history=[
                    {"batch": 0, "rect": list(crop_rect) if crop_rect else None}
                ],
                mean_slope_threshold=mean_slope_thresh,
                match_score_threshold=match_score_thresh,
                max_pixels_threshold=max_pixels_thresh,
                mean_pixel_values=[],
                specimen_currents=[],
                specimen_current_batch_numbers=[],
                match_scores=[],
                match_score_batch_numbers=[],
                white_pixel_percentages=[],
                batch_numbers=[],
                accumulated_images=[],  # For batch processing
                reference_template=None,
                evaluation_state=EvaluationState(),
                scan_direction=pattern_info.scan_direction or "",
                rotation=pattern_info.rotation
            )

            self.pattern_states.append(pattern_state)

            # Publish the pattern's scan direction + rotation so the RTM plot can
            # highlight the corresponding crop-box edge (static for the session).
            self.pattern_scan_direction_ready.emit(
                i, pattern_state.scan_direction, pattern_state.rotation
            )

            # Build initial evaluation criteria (cached, rebuilt on parameter change)
            self._build_criteria(pattern_state)

            logger.info(
                f"Initialized Pattern {i+1} state:\n"
                f"  ID: {pattern_info.pattern_id}\n"
                f"  Crop: {crop_rect}\n"
                f"  Scan direction: {pattern_state.scan_direction or '(none)'}\n"
                f"  Rotation: {pattern_state.rotation:.3f} rad\n"
                f"  Thresholds: slope={mean_slope_thresh}, "
                f"pixels={max_pixels_thresh}%"
            )

    def _build_criteria(self, pattern_state: PatternState):
        """
        Build and cache EvaluationCriteria on a pattern state.

        Called at initialization and after parameter updates to avoid
        reconstructing the criteria dataclass on every batch cycle.

        :param pattern_state: Pattern state to update
        """
        pattern_state.criteria = EvaluationCriteria(
            mean_pixel_slope_threshold=pattern_state.mean_slope_threshold,
            match_score_threshold=pattern_state.match_score_threshold,
            maximum_pixels_threshold=pattern_state.max_pixels_threshold,
            confirmation_rounds=self.parameters.confirmation_rounds,
            mean_slope_enabled=self.parameters.mean_slope_enabled,
            match_score_enabled=self.parameters.match_score_enabled,
            percent_pixels_enabled=self.parameters.percent_pixels_enabled,
            foreground_completion_mode=self.parameters.foreground_completion_mode,
        )

    def _reset_evaluation_states(self):
        """Reset evaluation states after patterns complete."""
        for ps in self.pattern_states:
            ps.evaluation_state = EvaluationState()
            ps.reference_template = None
            ps.start_monitoring_time = None
            ps.completion_time = None
            ps.completion_logged = False
            ps.criteria_met_batch = None
            ps.accumulated_images = []  # Clear batch for next session
            ps.mean_pixel_at_delay = None  # Reset threshold reference
            # Reset per-pattern delay state
            ps.delay_active = True
            ps.delay_match_below_count = 0
            ps.initial_mean_pixel_value = None
            ps.stall_history_start = 0  # Stall latch reads the new session's history from the top
            ps.stall_dropped_carry = False
            ps.last_gain_corrected = False
            # Clear accumulated plot data for fresh session
            ps.mean_pixel_values = []
            ps.specimen_currents = []
            ps.specimen_current_batch_numbers = []
            ps.match_scores = []
            ps.match_score_batch_numbers = []
            ps.white_pixel_percentages = []
            ps.batch_numbers = []
            ps.batch_count = 0
            # Reseed crop history so a second session doesn't inherit stale entries
            ps.crop_rect_history = [
                {"batch": 0, "rect": list(ps.crop_rect) if ps.crop_rect else None}
            ]

        logger.info("Evaluation states reset for next patterning session")

    def _process_pattern(
        self,
        pattern_state: PatternState,
        rtm_image: np.ndarray
    ):
        """
        Process a single pattern's RTM image with independent batch processing.

        Images are accumulated until a full batch (number_of_images) is collected,
        then the batch is processed and cleared. Each batch is independent -- no
        overlap with previous batches.

        Workflow:
        1. Accumulate images until batch is full
        2. Filter the complete batch
        3. Emit full filtered image for display
        4. Crop filtered image to analysis region
        5. Calculate mean pixel value
        6. Clear batch and start collecting next batch
        7. If delay active: check delay disengage conditions
        8. If delay off: calculate metrics, evaluate completion criteria
        """
        pattern_idx = pattern_state.pattern_index

        # Accumulate images for batch processing
        # If image dimensions changed (new pattern size), clear stale batch
        if (pattern_state.accumulated_images and
                pattern_state.accumulated_images[0].shape != rtm_image.shape):
            logger.info(
                f"Pattern {pattern_idx+1}: Image dimensions changed "
                f"{pattern_state.accumulated_images[0].shape} -> {rtm_image.shape} "
                f"- clearing batch"
            )
            pattern_state.accumulated_images.clear()

        # No .copy() needed -- rtm_image is a fresh array from get_images_from_rtm_data
        pattern_state.accumulated_images.append(rtm_image)

        # Need full batch before processing
        if len(pattern_state.accumulated_images) < self.parameters.number_of_images:
            return

        # ---- Full batch collected - process and clear ----

        # Capture last raw image before batch processing (for save data)
        last_raw_image = pattern_state.accumulated_images[-1]

        # Batch-averaged raw frame (raw-count space) for the top-hat foreground
        # map. Averaging the whole batch lowers per-frame noise versus a single
        # frame, so the energy floor is cleaner and the foreground map steadier.
        # Computed before the batch is cleared, and only when a top-hat path
        # actually needs it (the energy metric or foreground-based matching).
        need_foreground = (
            self.parameters.binarization_method == BinarizationMethod.TOPHAT_ENERGY
            or self.parameters.match_on_foreground
        )
        batch_mean_raw = (
            blend_images(pattern_state.accumulated_images) if need_foreground else None
        )

        # Apply batch filtering on full-size images. On the grayscale match
        # path, also take the fixed-scale match image (|current - background|
        # in counts/255, never per-frame stretched): the display/metrics output
        # is per-frame rescaled and must not feed the matcher -- the stretch
        # amplifies residual noise to full scale and pins the score at 1.0
        # (the 2026-08-02 grayscale incident; same amplifier the foreground
        # path lost in v3.3.3).
        if self.parameters.match_on_foreground:
            processed_image = filter_images(
                pattern_state.accumulated_images,
                gaussian_sigma=self.parameters.gaussian_sigma,
                apply_dilation=self.parameters.apply_dilation
            )
            gray_match_full = None
        else:
            processed_image, gray_match_full = filter_images(
                pattern_state.accumulated_images,
                gaussian_sigma=self.parameters.gaussian_sigma,
                apply_dilation=self.parameters.apply_dilation,
                with_match_image=True
            )

        # Clear batch for next collection cycle
        pattern_state.accumulated_images.clear()

        # Emit full-size processed image for display
        self.processed_image_ready.emit(processed_image, pattern_idx)

        # Crop the filtered image for metrics analysis
        if pattern_state.crop_rect:
            analysis_image = crop_to_rect(processed_image, pattern_state.crop_rect)
        else:
            analysis_image = processed_image

        # Image used for template matching. When foreground matching is enabled,
        # match on the normalized top-hat foreground map (tracks structural change
        # and ignores background/brightness drift) instead of the grayscale image.
        # Kept separate from analysis_image so the mean-pixel and percent-pixels
        # metrics are unaffected. Used for the reference template in both phases.
        if self.parameters.match_on_foreground and batch_mean_raw is not None:
            fg_full = white_tophat_map(
                batch_mean_raw, radius=self.parameters.tophat_radius
            )
            fg_crop = (crop_to_rect(fg_full, pattern_state.crop_rect)
                       if pattern_state.crop_rect else fg_full)
            match_image = tophat_normalized(fg_crop)
        else:
            # Grayscale path at fixed absolute scale (filter_images' signed
            # change map, not the per-frame-stretched analysis_image), folded
            # sign-split into the matcher's [0,1] domain after cropping so a
            # contrast inversion still reads as change.
            gray_crop = (crop_to_rect(gray_match_full, pattern_state.crop_rect)
                         if pattern_state.crop_rect else gray_match_full)
            match_image = signed_to_match_image(gray_crop)

        logger.debug(
            f"Pattern {pattern_idx+1} image dimensions - "
            f"full: {processed_image.shape[1]}x{processed_image.shape[0]}, "
            f"cropped: {analysis_image.shape[1]}x{analysis_image.shape[0]}"
        )

        # Calculate mean pixel value from processed image
        mean_pixel_value = calculate_mean_pixel_value(analysis_image)
        pattern_state.mean_pixel_values.append(mean_pixel_value)
        pattern_state.batch_count += 1
        pattern_state.batch_numbers.append(pattern_state.batch_count)
        if pattern_state.batch_count % 25 == 0:
            crash_breadcrumbs.drop(
                f"p{pattern_idx + 1} batch {pattern_state.batch_count}")

        # Store initial mean pixel value for delay percentage comparison
        if pattern_state.initial_mean_pixel_value is None:
            pattern_state.initial_mean_pixel_value = mean_pixel_value

        # Emit mean pixel data for plotting (x-axis = batch number)
        # Scale from [0.0, 1.0] to [0, 255] for display (internal values stay in [0, 1])
        self.mean_pixel_data_ready.emit(
            pattern_state.batch_numbers.copy(),
            [v * 255.0 for v in pattern_state.mean_pixel_values],
            pattern_idx
        )

        # Read and emit specimen current for plotting (skipped on systems
        # that do not support specimen current, e.g. Arctis)
        if self._specimen_current_available:
            current_value = get_specimen_current(self.microscope)
            if current_value is not None:
                # Append the batch number in lockstep with the current value so
                # the two series stay aligned even when a read returns None on a
                # later batch (otherwise x and y desync permanently and the GUI
                # raises "x and y must have the same first dimension" at draw).
                pattern_state.specimen_currents.append(current_value)
                pattern_state.specimen_current_batch_numbers.append(
                    pattern_state.batch_count
                )
                self.specimen_current_data_ready.emit(
                    pattern_state.specimen_current_batch_numbers.copy(),
                    pattern_state.specimen_currents.copy(),
                    pattern_idx
                )

        # ---- Delay phase (per-pattern) ----
        if pattern_state.delay_active:
            self._process_delay_phase(
                pattern_state, analysis_image, mean_pixel_value, pattern_idx,
                last_raw_image, processed_image, match_image
            )
            return

        # ---- Active monitoring phase (post-delay) ----
        self._process_monitoring_phase(
            pattern_state, processed_image, analysis_image,
            mean_pixel_value, pattern_idx,
            last_raw_image, batch_mean_raw, match_image
        )

    def _process_delay_phase(
        self,
        pattern_state: PatternState,
        analysis_image: np.ndarray,
        mean_pixel_value: float,
        pattern_idx: int,
        last_raw_image: np.ndarray,
        processed_image: np.ndarray,
        match_image: np.ndarray
    ):
        """
        Handle delay phase processing for a single pattern.

        During delay, we capture a reference template and compute match scores
        to detect when milling has visibly begun. Delay disengages when either:
        1. Match score transitions from below to above threshold (milling started)
        2. Mean pixel value drops by percent_difference_threshold from initial value

        :param pattern_state: State for this pattern
        :param analysis_image: Cropped filtered image for analysis
        :param mean_pixel_value: Current mean pixel value
        :param pattern_idx: Pattern index (0 or 1)
        :param last_raw_image: Last raw RTM image from batch (for save data)
        :param processed_image: Batch-filtered image (for save data)
        :param match_image: Image used for template matching (grayscale analysis
            image, or the normalized top-hat foreground map when match-on-foreground
            is enabled); reference template is captured from and compared on this
        """
        # Save batch images (no binary during delay). Wrapped like the metrics
        # save below: an image-write failure (transient share error, disk full)
        # must not abort the run -- a dead monitor would fail to stop milling
        # at criteria.
        if self._save_manager and last_raw_image is not None:
            try:
                self._save_manager.save_batch_images(
                    pattern_idx,
                    batch_number=pattern_state.batch_count,
                    raw_image=last_raw_image,
                    grayscale_image=processed_image,
                    match_image=match_image
                )
            except Exception as e:
                logger.error(
                    f"Failed to save batch images for pattern "
                    f"{pattern_idx + 1}: {e}",
                    exc_info=True,
                )

        # First batch: capture reference template, no comparison possible yet
        if pattern_state.reference_template is None:
            pattern_state.reference_template = match_image.copy()
            logger.info(
                f"Pattern {pattern_idx+1}: Reference template captured during delay"
            )
            self._emit_delay_results(pattern_idx)
            return

        # Subsequent batches: calculate match score against previous batch
        match_score = self._calculate_match(
            pattern_state.reference_template, match_image, pattern_idx,
            pattern_state
        )

        # Rolling reference: update to current batch for next comparison (also
        # self-heals a shape mismatch on the next batch)
        pattern_state.reference_template = match_image.copy()

        # Delay may only disengage on a live frame: a blank (sparse-zeroed) filtered
        # frame is "no signal", not a milling signature, and disengaging on it would
        # freeze a degenerate binarization threshold from an all-zero image.
        frame_live = mean_pixel_value > 0.0

        if match_score is not None:
            # Store match score and emit for plotting during delay
            pattern_state.match_scores.append(match_score)
            pattern_state.match_score_batch_numbers.append(pattern_state.batch_count)
            self.match_score_data_ready.emit(
                pattern_state.match_score_batch_numbers.copy(),
                pattern_state.match_scores.copy(),
                pattern_idx
            )

            # Condition 1: Match score transition (below threshold -> above threshold)
            # Track times match score was below threshold
            if match_score <= pattern_state.match_score_threshold:
                pattern_state.delay_match_below_count += 1

            # If score has been below threshold before and now crosses above,
            # milling has begun
            if (match_score >= pattern_state.match_score_threshold
                    and pattern_state.delay_match_below_count > 0
                    and frame_live):
                logger.info(
                    f"Pattern {pattern_idx+1}: Delay off - match score transition "
                    f"(score={match_score:.4f}, threshold={pattern_state.match_score_threshold})"
                )
                self._disengage_delay(pattern_state, mean_pixel_value, pattern_idx, analysis_image)
                return
        else:
            # Score unavailable (shape mismatch / degenerate input): skip only the
            # match-transition condition -- the mean-pixel drop below is
            # score-independent and must still be able to disengage delay.
            logger.warning(
                f"Pattern {pattern_idx+1}: match score unavailable this batch - "
                f"match transition check skipped"
            )

        # Condition 2: Mean pixel percentage drop from initial value
        initial = pattern_state.initial_mean_pixel_value
        if initial is not None and initial > 0 and frame_live:
            avg = (initial + mean_pixel_value) / 2
            if avg > 0:
                percent_diff = (abs(initial - mean_pixel_value) / avg) * 100

                if (percent_diff >= self.parameters.percent_difference_threshold
                        and mean_pixel_value < initial):
                    logger.info(
                        f"Pattern {pattern_idx+1}: Delay off - mean pixel drop "
                        f"({percent_diff:.1f}% >= {self.parameters.percent_difference_threshold}%)"
                    )
                    self._disengage_delay(pattern_state, mean_pixel_value, pattern_idx, analysis_image)
                    return

        # Still in delay
        self._emit_delay_results(pattern_idx)

    def _emit_delay_results(self, pattern_idx: int):
        """Emit delay-phase placeholder results to the GUI."""
        results = {
            'mean_slope': 'Delay',
            'match_score': 'Delay',
            'white_pixels': 'Delay',
            'confirmation_count': 0,
            'confirmation_total': self.parameters.confirmation_rounds,
            'is_delay': True,
            'is_complete': False,
            'all_criteria_met': False
        }
        self.pattern_results_ready.emit(results, pattern_idx)

    def _calculate_match(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        pattern_idx: int,
        pattern_state: PatternState
    ) -> float | None:
        """
        Calculate match score between reference and current images
        using normalized squared difference template matching.

        Inter-frame gain handling lives here (not in the scorer) because it
        needs cross-batch state: a genuine CB/detector step is a one-off
        event, while a "gain" detected on consecutive batches is a sustained
        uniform fade -- real milling change that must not be normalized away
        (adversarial review 2026-08-18: end-stage uniform thinning of bright
        residuals satisfied every uniformity gate batch after batch and would
        have scored ~0.01 instead of 0.14-0.28 for the whole collapse).

        :param reference: Reference template image
        :param current: Current analysis image
        :param pattern_idx: Pattern index (0 or 1), for log attribution
        :param pattern_state: Pattern state carrying the consecutive-gain flag
        :return: Match score (0.0 = perfect match, 1.0 = maximum difference),
            or None when no score could be computed this batch
        """
        gain = estimate_interframe_gain(reference, current)
        if gain is not None and pattern_state.last_gain_corrected:
            logger.warning(
                f"Pattern {pattern_idx + 1}: inter-frame gain detected on "
                f"consecutive batches (k={gain:.3f}) - correction refused; "
                f"sustained uniform change is milling, not a CB step"
            )
            gain = None
        pattern_state.last_gain_corrected = gain is not None
        return calculate_match_score(
            reference, current,
            min_splits=self.parameters.min_pattern_splits,
            target_tile_size=self.parameters.target_tile_size,
            log_tag=f"Pattern {pattern_idx + 1}: ",
            interframe_gain=gain
        )

    def _calculate_slope(self, values: list[float]) -> float:
        """
        Calculate the absolute slope of a series of values using the configured method.

        Returns 0.0 if insufficient data points for the configured method.

        :param values: List of metric values (mean pixels, match scores, etc.)
        :return: Absolute slope value
        """
        if self.parameters.slope_method == SlopeMethod.LINEAR_REGRESSION:
            if len(values) >= self.parameters.linear_regression_fit_points:
                return abs(calculate_slope_linear_regression(
                    values,
                    fit_number=self.parameters.linear_regression_fit_points
                ))
        else:
            # Gradient method
            if len(values) >= self.parameters.num_points_for_slope:
                slope = calculate_slope_of_mean_pixel_values(
                    values,
                    num_points=self.parameters.num_points_for_slope
                )
                if slope is not None:
                    return abs(slope)
        return 0.0

    def _disengage_delay(
        self,
        pattern_state: PatternState,
        current_mean_pixel: float,
        pattern_idx: int,
        analysis_image: np.ndarray
    ):
        """
        Disengage delay for a pattern and capture binarization threshold.

        Uses multi-Otsu thresholding on the current analysis image to determine
        the optimal threshold for separating material from background. This
        data-driven threshold is then fixed for the remainder of monitoring,
        providing a principled binarization boundary without the drift that
        continuous re-calculation would cause.

        :param pattern_state: State for this pattern
        :param current_mean_pixel: Current mean pixel value
        :param pattern_idx: Pattern index (0 or 1)
        :param analysis_image: Current cropped filtered image for Otsu calculation
        """
        # Capture the frozen binary threshold from the current image histogram. The
        # binarization method selects which multi-Otsu class boundary becomes the
        # white cutoff (TOP = brightest class only, the original behavior; FROZEN_MID
        # / FROZEN_LOW pick a lower boundary so faint mid-gray features count). The
        # mid/low options use 4 classes so there are distinct low/mid/high boundaries.
        method = self.parameters.binarization_method
        if method == BinarizationMethod.TOPHAT_ENERGY:
            # Top-hat needs no frozen threshold (it is computed per-frame and is
            # background-level agnostic); leave the frozen value unset.
            otsu_value = None
        elif method == BinarizationMethod.FROZEN_LOW:
            otsu_value = frozen_threshold_for_boundary(analysis_image, num_classes=4, boundary_index=0)
        elif method == BinarizationMethod.FROZEN_MID:
            otsu_value = frozen_threshold_for_boundary(analysis_image, num_classes=4, boundary_index=1)
        else:  # BinarizationMethod.TOP - original behavior
            otsu_value = frozen_threshold_for_boundary(
                analysis_image, num_classes=self.parameters.threshold_num_classes, boundary_index=-1
            )
        pattern_state.mean_pixel_at_delay = otsu_value

        # Mark delay as off for this pattern
        pattern_state.delay_active = False

        # Start timing from when monitoring begins
        pattern_state.start_monitoring_time = datetime.datetime.now()

        # otsu_value is None for TOPHAT_ENERGY (no frozen threshold is captured -
        # it is computed per frame), so guard the numeric format to avoid a
        # TypeError: unsupported format string passed to NoneType.__format__.
        frozen_desc = (
            "n/a (top-hat, per-frame)" if otsu_value is None else f"{otsu_value:.4f}"
        )
        logger.info(
            f"Pattern {pattern_idx+1}: Delay disengaged - "
            f"frozen_threshold={frozen_desc} "
            f"(method={method.name}, current_mean={current_mean_pixel:.2f})"
        )
        self.status_update.emit(
            f"Pattern {pattern_idx+1}: Delay off - monitoring active"
        )

    def _process_monitoring_phase(
        self,
        pattern_state: PatternState,
        processed_image: np.ndarray,
        analysis_image: np.ndarray,
        mean_pixel_value: float,
        pattern_idx: int,
        last_raw_image: np.ndarray,
        batch_mean_raw: np.ndarray,
        match_image: np.ndarray
    ):
        """
        Handle active monitoring phase for a single pattern (post-delay).

        Calculates slope, binary images, white pixels, match score, match score
        slope, and evaluates completion criteria.

        :param pattern_state: State for this pattern
        :param processed_image: Full-size filtered image
        :param analysis_image: Cropped filtered image
        :param mean_pixel_value: Current mean pixel value
        :param pattern_idx: Pattern index (0 or 1)
        :param last_raw_image: Last raw RTM image from batch (for save data)
        :param batch_mean_raw: Batch-averaged raw frame (top-hat source for the
            energy metric); None when no top-hat path is active
        :param match_image: Image used for template matching (grayscale analysis
            image, or the normalized top-hat foreground map when match-on-foreground
            is enabled)
        """
        # Calculate slope from mean pixel values (each value is from an independent batch)
        mean_slope = self._calculate_slope(pattern_state.mean_pixel_values)

        # Compute the foreground metric (white_pixel_percentage) and a display map.
        # TOPHAT_ENERGY uses a background-level-agnostic morphological top-hat on the
        # raw frame and reports a continuous energy; the frozen-threshold methods
        # binarize the filtered image and count white pixels. Either way the scalar
        # flows unchanged into the criteria (maximum_pixels_threshold).
        if self.parameters.binarization_method == BinarizationMethod.TOPHAT_ENERGY:
            # Top-hat on the full batch-averaged raw frame (real surrounding context
            # -> no crop-edge ring in the local-background estimate; averaging the
            # batch lowers per-frame noise for a cleaner energy floor)...
            fg_map_full = white_tophat_map(batch_mean_raw, radius=self.parameters.tophat_radius)
            # ...but measure energy over the crop so the metric stays scoped to this
            # pattern (bright static structure outside the crop must not inflate it).
            fg_map_metric = (crop_to_rect(fg_map_full, pattern_state.crop_rect)
                             if pattern_state.crop_rect else fg_map_full)
            white_pixel_percentage = tophat_energy(fg_map_metric)
            binary_image_full = tophat_display(fg_map_full)
        else:
            # Create binary image from cropped filtered image (for metrics)
            binary_image = threshold_adaptive(
                analysis_image,
                delay_active=False,
                mean_pixel_at_delay=pattern_state.mean_pixel_at_delay
            )

            # Create binary image from full-size filtered image (for display)
            binary_image_full = threshold_adaptive(
                processed_image,
                delay_active=False,
                mean_pixel_at_delay=pattern_state.mean_pixel_at_delay
            )

            # Calculate white pixel percentage from binary image
            white_pixel_percentage = calculate_num_white_pixels(
                binary_image,
                return_percentage=True
            )

        self.binary_image_ready.emit(binary_image_full, pattern_idx)

        # Save batch images (raw + grayscale + binary/foreground map). Skipped once
        # the pattern is complete: evaluation runs after this save, so is_complete
        # here reflects the previous batch -- the criteria-met batch itself is still
        # saved (and renamed below), only the frozen post-completion frames are not.
        if (self._save_manager and last_raw_image is not None
                and not (pattern_state.evaluation_state
                         and pattern_state.evaluation_state.is_complete)):
            try:
                self._save_manager.save_batch_images(
                    pattern_idx,
                    batch_number=pattern_state.batch_count,
                    raw_image=last_raw_image,
                    grayscale_image=processed_image,
                    binary_image=binary_image_full,
                    match_image=match_image
                )
            except Exception as e:
                logger.error(
                    f"Failed to save batch images for pattern "
                    f"{pattern_idx + 1}: {e}",
                    exc_info=True,
                )

        pattern_state.white_pixel_percentages.append(white_pixel_percentage)

        # Grid-bar stall latch: evaluate over the foreground history from the
        # last crop change onward (a new crop is a new energy scale, so the
        # running peak must not span it). Cheap (pure Python over <=~200
        # values) and unused by plain ABSOLUTE mode -- fails closed there.
        stall_config = StallConfig(
            stall_window=self.parameters.stall_window,
            stall_drop_fraction=self.parameters.stall_drop_fraction,
            stall_rel_tolerance=self.parameters.stall_rel_tolerance,
            stall_abs_tolerance=self.parameters.stall_abs_tolerance,
        )
        stall_slice = pattern_state.white_pixel_percentages[
            pattern_state.stall_history_start:]
        stall = evaluate_foreground_stall(stall_slice, stall_config)
        # A crop change while the trace was already floored must not
        # permanently disarm the latch (its flagship scenario): fold the
        # carried dropped-state in, discarding it if the post-crop trace shows
        # renewed activity or once the new history re-arms on its own.
        stall, carry = apply_stall_carry(
            stall, stall_slice, stall_config, pattern_state.stall_dropped_carry)
        if pattern_state.stall_dropped_carry and not carry:
            logger.info(
                f"Pattern {pattern_idx+1}: stall-latch carry cleared "
                f"(post-crop trace re-armed or shows renewed activity)"
            )
        pattern_state.stall_dropped_carry = carry

        # Template matching - recapture reference if needed (e.g. after crop rect
        # change). Scoring is skipped on the recapture batch: the reference is the
        # current image, so scoring would inject a fabricated ~0.0 "perfect match"
        # into the confirmation counter (the delay phase returns after capture for
        # the same reason). None fails the match criterion closed downstream.
        if pattern_state.reference_template is None:
            pattern_state.reference_template = match_image.copy()
            logger.info(
                f"Pattern {pattern_idx+1}: Reference template recaptured "
                f"(crop rect change or new session); match scoring skipped this batch"
            )
            match_score = None
        else:
            # Calculate match score using configured method
            match_score = self._calculate_match(
                pattern_state.reference_template, match_image, pattern_idx,
                pattern_state
            )

        # Rolling reference: update to current batch for next comparison (also
        # self-heals a shape mismatch on the next batch)
        pattern_state.reference_template = match_image.copy()

        # Store match score and emit for plotting. A None score (shape mismatch /
        # degenerate input) is not plotted or stored -- and downstream it fails the
        # match criterion closed rather than passing as a fake "perfect match".
        if match_score is not None:
            pattern_state.match_scores.append(match_score)
            pattern_state.match_score_batch_numbers.append(pattern_state.batch_count)
            self.match_score_data_ready.emit(
                pattern_state.match_score_batch_numbers.copy(),
                pattern_state.match_scores.copy(),
                pattern_idx
            )
        else:
            logger.warning(
                f"Pattern {pattern_idx+1}: match score unavailable this batch"
            )

        # Save batch metrics. Wrapped in try/except so a persistence error logs
        # and the monitoring loop continues rather than aborting the run (a dead
        # monitor would fail to stop milling at criteria).
        if self._save_manager:
            try:
                self._save_manager.save_batch_metrics(
                    pattern_idx,
                    batch_number=pattern_state.batch_count,
                    image_number=pattern_state.batch_count,
                    raw_image_size=(processed_image.shape[1], processed_image.shape[0]),
                    crop_image_size=(analysis_image.shape[1], analysis_image.shape[0]),
                    mean_pixel_value=mean_pixel_value,
                    mean_pixel_slope=mean_slope,
                    match_score=match_score,
                    white_pixel_percentage=white_pixel_percentage,
                    smoothed_foreground=stall.smoothed,
                    running_peak=stall.baseline,
                    stall_latched=stall.latched
                )
            except Exception as e:
                logger.error(
                    f"Failed to save batch metrics for pattern "
                    f"{pattern_idx + 1}: {e}",
                    exc_info=True,
                )
        
        # Evaluate completion criteria (criteria cached, rebuilt on parameter change)
        pattern_state.evaluation_state = evaluate_pattern_completion(
            state=pattern_state.evaluation_state,
            criteria=pattern_state.criteria,
            mean_pixel_slope=mean_slope,
            match_score=match_score,
            white_pixels_percentage=white_pixel_percentage,
            delay_active=False,  # We're past delay now
            stall=stall,
            log_tag=f"Pattern {pattern_idx + 1}: "
        )
        
        # Log completion with duration if pattern just completed
        if (pattern_state.evaluation_state.is_complete and 
            not pattern_state.completion_logged and
            pattern_state.start_monitoring_time is not None):
            
            pattern_state.completion_time = datetime.datetime.now()
            duration = pattern_state.completion_time - pattern_state.start_monitoring_time
            duration_str = self._format_duration(duration)
            
            # Record the batch on which criteria were met (drives the GUI freeze,
            # plot marker, badge, JSON field, and the PNG rename below).
            pattern_state.criteria_met_batch = pattern_state.batch_count

            # Any-round-in-streak, not just the final round: a streak can mix
            # ABS and STALL rounds (raw value straddling the threshold while
            # the smoothed latch holds), and a completion where the latch was
            # load-bearing on any round must carry the review flag.
            via_stall = pattern_state.evaluation_state.stall_used_in_streak
            logger.info(
                f"Pattern {pattern_idx + 1} complete at batch "
                f"{pattern_state.criteria_met_batch} "
                f"Duration: {duration_str} "
                f"(started {pattern_state.start_monitoring_time.strftime('%H:%M:%S')}, "
                f"completed {pattern_state.completion_time.strftime('%H:%M:%S')})"
                + (" [stall latch was load-bearing for completion]"
                   if via_stall else "")
            )

            pattern_state.completion_logged = True

            # Tag the criteria-met batch's saved images. Wrapped in try/except so a
            # rename error logs and the monitoring loop continues rather than aborting
            # the run (a dead monitor would fail to stop milling at criteria).
            if self._save_manager:
                try:
                    self._save_manager.mark_batch_criteria_met(
                        pattern_idx, pattern_state.batch_count
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to mark criteria-met batch for pattern "
                        f"{pattern_idx + 1}: {e}",
                        exc_info=True,
                    )
        
        # Emit results for GUI update
        results = {
            'mean_slope': mean_slope,
            'match_score': match_score,
            'white_pixels': white_pixel_percentage,
            'confirmation_count': pattern_state.evaluation_state.confirmation_count,
            'confirmation_total': self.parameters.confirmation_rounds,
            'is_complete': pattern_state.evaluation_state.is_complete,
            'all_criteria_met': pattern_state.evaluation_state.all_criteria_met,
            # Authoritative per-criterion pass/fail (drives the GUI indicators so
            # they match the confirmation counter exactly).
            'slope_ok': pattern_state.evaluation_state.slope_ok,
            'match_ok': pattern_state.evaluation_state.match_ok,
            'pixels_ok': pattern_state.evaluation_state.pixels_ok,
            # Which path satisfied the foreground criterion ("ABS"/"STALL"/None)
            # and the raw latch condition -- drives the "(stall)" annotation.
            'pixels_via': pattern_state.evaluation_state.pixels_via,
            'stall_latched': stall.latched,
            # Sticky across the confirmation streak: True when any round of
            # the streak passed via the latch -- the completion-time flag.
            'stall_in_streak': pattern_state.evaluation_state.stall_used_in_streak,
            'is_delay': False,
            'mean_slope_threshold': pattern_state.mean_slope_threshold,
            'match_score_threshold': pattern_state.match_score_threshold,
            'max_pixels_threshold': pattern_state.max_pixels_threshold,
            'duration': None,
            'criteria_met_batch': pattern_state.criteria_met_batch
        }

        # Include duration if pattern is complete (uses frozen completion_time)
        if (pattern_state.evaluation_state.is_complete and
            pattern_state.completion_time is not None):
            duration = pattern_state.completion_time - pattern_state.start_monitoring_time
            results['duration'] = self._format_duration(duration)
        
        self.pattern_results_ready.emit(results, pattern_idx)
    
    def _cleanup(self):
        """Cleanup resources and emit completion signal."""
        self.status_update.emit("Cleaning up...")
        
        # Finalize any in-progress saved data (user stopped mid-session). The
        # criteria-met path finalizes itself (and clears _save_manager), so
        # reaching here means the session ended for another reason -- label it.
        if self._save_manager and self._patterning_was_running:
            if self._rtm_failure_stop:
                run_status = RUN_STATUS_RTM_FAILURE
            elif self._error_stop:
                run_status = RUN_STATUS_ERROR
            else:
                run_status = RUN_STATUS_USER_STOPPED
            self._finalize_saved_data(run_status, self._stop_reason)
        
        if self.microscope is not None:
            disconnect_from_microscope(self.microscope)
            self.microscope = None
        # The right-side indicator must not keep claiming a connection the
        # disconnect above just ended (independent of the left side's stop
        # reason, so ordering against monitoring_stopped does not matter).
        self.connection_status_update.emit(StatusText.NOT_CONNECTED)

        # Update UI states
        self.set_monitoring_icon_active.emit(False)
        self.monitoring_stopped.emit(self._stop_reason)  # Signal for button states
        
        logger.info("Workflow cleanup complete")
    
    def _finalize_saved_data(self, completion_status: str, stop_reason: str = ""):
        """
        Finalize saved data for the current patterning session.

        Builds metadata from current state, writes metrics CSVs and JSON,
        and emits save_plots_requested so the GUI can save plot snapshots.

        :param completion_status: Run-level reason the session ended, written to
            run_info.completion_status (e.g. RUN_STATUS_CRITERIA_MET /
            RUN_STATUS_USER_STOPPED). The same data is saved either way; this only
            labels why the run ended.
        :param stop_reason: Human-readable reason, written to run_info.stop_reason.
        """
        if not self._save_manager:
            return
        
        try:
            # Build per-pattern detail dicts
            pattern_details = []
            for ps in self.pattern_states:
                info = None
                if self._pattern_infos and ps.pattern_index < len(self._pattern_infos):
                    info = self._pattern_infos[ps.pattern_index]
                
                completion_time = None
                duration_str = ""
                if (ps.evaluation_state and ps.evaluation_state.is_complete
                        and ps.completion_time is not None):
                    completion_time = ps.completion_time
                    duration = completion_time - ps.start_monitoring_time
                    duration_str = self._format_duration(duration)

                # Per-pattern status is independent of the run-level status: a
                # user-stopped run can still contain a pattern that completed
                # earlier in the session.
                pattern_status = (
                    PATTERN_STATUS_CRITERIA_MET if completion_time is not None
                    else PATTERN_STATUS_INCOMPLETE
                )

                detail = self._save_manager.build_pattern_detail(
                    pattern_idx=ps.pattern_index,
                    pattern_id=ps.pattern_id,
                    pattern_type=info.pattern_type if info else "",
                    width_um=info.width * 1e6 if info else 0.0,
                    height_um=info.height * 1e6 if info else 0.0,
                    aspect_ratio=info.aspect_ratio if info else 0.0,
                    crop_rect=ps.crop_rect,
                    crop_rect_history=ps.crop_rect_history,
                    monitoring_start=ps.start_monitoring_time,
                    completion_time=completion_time,
                    duration=duration_str,
                    criteria_met_batch=ps.criteria_met_batch,
                    completion_status=pattern_status,
                    completed_via_stall=(
                        completion_time is not None
                        and ps.evaluation_state is not None
                        and ps.evaluation_state.stall_used_in_streak
                    ),
                )
                pattern_details.append(detail)
            
            # Build UI parameters dict
            ui_params = {
                "window_mode": self.parameters.window_mode.name,
                "number_of_images": self.parameters.number_of_images,
                "analysis_interval_seconds": self.parameters.analysis_interval_seconds,
                "mean_pixel_slope_threshold": self.pattern_states[0].mean_slope_threshold if self.pattern_states else 0,
                "match_score_threshold": self.pattern_states[0].match_score_threshold if self.pattern_states else 0,
                "maximum_pixels_threshold": self.pattern_states[0].max_pixels_threshold if self.pattern_states else 0,
                "confirmation_rounds": self.parameters.confirmation_rounds,
            }
            
            # Build processing parameters dict
            processing_params = {
                "gaussian_sigma": self.parameters.gaussian_sigma,
                "apply_dilation": self.parameters.apply_dilation,
                "binarization_method": self.parameters.binarization_method.name,
                "tophat_radius": self.parameters.tophat_radius,
                "match_on_foreground": self.parameters.match_on_foreground,
                "foreground_completion_mode": self.parameters.foreground_completion_mode.name,
                "stall_window": self.parameters.stall_window,
                "stall_drop_fraction": self.parameters.stall_drop_fraction,
                "stall_rel_tolerance": self.parameters.stall_rel_tolerance,
                "stall_abs_tolerance": self.parameters.stall_abs_tolerance,
                "slope_method": self.parameters.slope_method.name,
                "num_points_for_slope": self.parameters.num_points_for_slope,
                "linear_regression_fit_points": self.parameters.linear_regression_fit_points,
                "min_pattern_splits": self.parameters.min_pattern_splits,
                "target_tile_size": self.parameters.target_tile_size,
                # Auto-CB configuration in force for this run (the campaign that
                # motivated this block was only reconstructable by log forensics).
                "auto_cb_on_start": self.parameters.auto_cb_on_start,
                # The cadence flag is load-bearing for reading a run's data:
                # it says whether per-session cb_calibration blocks were
                # expected (the 2026-08-27 evening campaign could not be
                # interpreted without asking the operator).
                "cb_recalibrate_every_session":
                    self.parameters.cb_recalibrate_every_session,
                "cb_white_level": self.parameters.cb_white_level,
                "cb_lower_margin": self.parameters.cb_lower_margin,
                "cb_upper_margin": self.parameters.cb_upper_margin,
                "cb_max_white_clip_fraction": self.parameters.cb_max_white_clip_fraction,
                "cb_max_black_clip_fraction": self.parameters.cb_max_black_clip_fraction,
                "cb_min_bound": self.parameters.cb_min_bound,
                "cb_max_bound": self.parameters.cb_max_bound,
                "cb_max_iterations": self.parameters.cb_max_iterations,
                "cb_settle_seconds": self.parameters.cb_settle_seconds,
                "cb_frames_per_measurement": self.parameters.cb_frames_per_measurement,
                # Load-bearing internal CB constant (makes traces interpretable).
                "cb_edge_tol": CB_EDGE_TOL_DEFAULT,
            }
            
            # Build criteria enabled dict
            criteria_enabled = {
                "mean_slope": self.parameters.mean_slope_enabled,
                "match_score": self.parameters.match_score_enabled,
                "percent_pixels": self.parameters.percent_pixels_enabled,
            }
            
            # Assemble and save metadata
            metadata = self._save_manager.build_run_metadata(
                microscope_data=self._microscope_data,
                ui_parameters=ui_params,
                processing_parameters=processing_params,
                criteria_enabled=criteria_enabled,
                pattern_details=pattern_details,
                completion_status=completion_status,
                stop_reason=stop_reason,
                cb_calibration=self._cb_summary,
            )

            # Metrics CSVs + run_metadata.json are written synchronously here.
            self._save_manager.finalize(metadata)
            self._save_manager.save_cb_trace_csv(self._cb_trace)

            logger.info(
                f"Metrics and metadata saved: "
                f"{self._save_manager.run_directory.name}"
            )

            # Request the GUI thread to save plot-snapshot PNGs (the figures
            # live on the GUI thread). This is a queued, non-blocking emit; the
            # GUI handler logs once the PNGs land. On app-close the GUI flushes
            # this queued request before teardown so the plots are not dropped.
            self.save_plots_requested.emit(
                str(self._save_manager.plots_directory)
            )
            
        except Exception as e:
            logger.error(f"Failed to finalize saved data: {e}", exc_info=True)
        finally:
            # Clear save manager for next session
            self._save_manager = None
            self._pattern_infos = None
    
    def _read_microscope_data(self) -> dict:
        """
        Read microscope settings for metadata.
        
        Reads ion beam and detector properties. Each property is wrapped in
        try/except since not all microscopes support all properties.
        
        :return: Dictionary of microscope settings
        """
        data = {}
        
        if self.microscope is None:
            return data
        
        # System identification
        system_name = get_system_name(self.microscope)
        if system_name:
            data["system_name"] = system_name
        
        # Ion beam properties
        try:
            data["ion_species"] = str(
                self.microscope.beams.ion_beam.source.plasma_gas.value
            )
        except Exception:
            data["ion_species"] = "unknown"
        
        try:
            data["ion_voltage_kv"] = float(
                self.microscope.beams.ion_beam.high_voltage.value
            )
        except Exception:
            data["ion_voltage_kv"] = None
        
        try:
            data["beam_current_na"] = float(
                self.microscope.beams.ion_beam.beam_current.value
            )
        except Exception:
            data["beam_current_na"] = None
        
        # Detector properties
        try:
            data["detector_type"] = str(
                self.microscope.detector.type.value
            )
        except Exception:
            data["detector_type"] = "unknown"
        
        try:
            data["detector_brightness"] = float(
                self.microscope.detector.brightness.value
            )
        except Exception:
            data["detector_brightness"] = None
        
        try:
            data["detector_contrast"] = float(
                self.microscope.detector.contrast.value
            )
        except Exception:
            data["detector_contrast"] = None
        
        logger.info(f"Microscope data read: {data}")
        return data
    
    @Slot()
    def request_stop(self):
        """Request the worker to stop gracefully."""
        logger.info("Stop requested")
        crash_breadcrumbs.drop("stop-requested")
        self._stop_requested = True
        self.status_update.emit("Stopping monitoring...")
    
    @Slot()
    def request_pause(self):
        """Pause the monitoring workflow.

        No bar emission: monitoring_paused drives the catbug label, which
        owns the paused state (surface separation, 2026-08-27).
        """
        if not self._paused:
            self._paused = True
            self.monitoring_paused.emit()
            logger.info("Monitoring paused")

    @Slot()
    def request_resume(self):
        """Resume the monitoring workflow (catbug label carries the state)."""
        if self._paused:
            self._paused = False
            self.monitoring_resumed.emit()
            logger.info("Monitoring resumed")
    
    @Slot(str, object, int)
    def update_parameter(self, param_name: str, value: object, pattern_idx: int = -1):
        """
        Queue a parameter update (thread-safe via signal/slot).
        
        This slot receives parameter changes from the GUI and queues them
        for application at the next safe point (start of batch processing).
        
        :param param_name: Name of parameter to update
        :param value: New value
        :param pattern_idx: Which pattern to update (-1 for all, 0/1 for specific)
        """
        # Validate before queuing
        is_valid, error_msg = validate_parameter_update(param_name, value)
        
        if not is_valid:
            logger.warning(f"Invalid parameter update rejected: {error_msg}")
            self.status_update.emit(f"Invalid update: {error_msg}")
            return
        
        # Queue the update. A refusal here means the name carries no declared
        # impact, so there is no safe state handling for it -- say nothing
        # reassuring to the operator in that case.
        if not self.param_manager.queue_update(param_name, value, pattern_idx):
            self.status_update.emit(
                f"Update refused: {display_name(param_name)}")
            return

        # Give immediate feedback
        self.status_update.emit(f"Update queued: {display_name(param_name)}")