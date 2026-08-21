"""
Global parameter management for the ATC Monitor application.
"""
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from PySide6.QtCore import QRect, QSettings


logger = logging.getLogger(__name__)


# QSettings identity
ORGANIZATION_NAME = "TFS_AutoScript"
APPLICATION_NAME = "TFS_AutoScript_ATCMonitor_3"


class WindowMode(Enum):
    """Mode for determining the analysis window size."""
    MANUAL = auto()         # User sets the number of images directly
    TIME_BASED = auto()     # User sets the analysis interval in seconds, window auto-calculated

class SlopeMethod(Enum):
    """Method for calculating slope of mean pixel values."""
    GRADIENT = auto()           # Gradient method
    LINEAR_REGRESSION = auto()  # Linear regression


class BinarizationMethod(Enum):
    """
    How the foreground is isolated for the "percent pixels" completion metric.

    TOP / FROZEN_MID / FROZEN_LOW freeze ONE multi-Otsu class boundary captured once
    when monitoring begins (so the threshold doesn't drift). They differ only in WHICH
    boundary becomes the white cutoff, i.e. how much faint mid-grey material counts as
    white. TOPHAT_ENERGY is a different approach: a morphological white top-hat that is
    agnostic to the absolute background level (see image_processing.white_tophat_map).
    """
    TOP = auto()         # Brightest Class only (thresholds[-1]) - original behavior
    FROZEN_MID = auto()  # Middle Boundary (4-class thresholds[1]) - balanced (default)
    FROZEN_LOW = auto()  # Lowest boundary (4-class thresholds[0]) - most inclusive
    TOPHAT_ENERGY = auto()  # White top-hat continuous foreground energy (background-level agnostic)


class ForegroundCompletionMode(Enum):
    """
    How the foreground (percent-pixels / Top-Hat energy) completion criterion decides "done".

    ABSOLUTE: the original behavior - foreground metric <= Maximum Pixels threshold.

    ABSOLUTE_PLUS_STALL (default): ABSOLUTE, plus a grid-bar backstop. The
    criterion also passes when the foreground trace has (a) dropped to <=
    stall_drop_fraction of its running peak AND (b) stayed flat for
    stall_window consecutive rounds (windowed span on a median-smoothed series
    within max(stall_rel_tolerance * value, stall_abs_tolerance)). Exposed
    static material (e.g. a grid bar) floors the trace at a sample-dependent
    non-zero level that no absolute threshold can be tuned to sit above in
    advance (the 2026-08-17 hard failures, Runs 11/12); the latch completes
    such runs once the trace has provably floored, while the ANDed match
    criterion attests no structural change is still occurring. Validated
    offline against the full 2026-08-17 campaign: identical completion batch
    to plain ABSOLUTE on all 26 completed channels, and fires on the two hung
    runs within 3 cycles of the operator's manual stop.

    (RELATIVE_PLATEAU, 3.3.3's unused draft of this idea, was removed in
    3.3.4: field replay showed it latching an innocent mid-run stall -- Run 17
    P1 at batch 30 of 76 -- and never firing on floors above its
    start-relative target. Persisted settings naming it fall back to the
    default via _enum_from_name.)
    """
    ABSOLUTE = auto()
    ABSOLUTE_PLUS_STALL = auto()


class SupportedPatternType(Enum):
    """Supported pattern types for pattern monitoring."""
    RECTANGLE = auto()                  # Rectangular patterns (RCS and CCS are not supported)


class RTMMode(Enum):
    """RTM mode for data acquisition."""
    LOW_RESOLUTION = auto()                       
    HIGH_RESOLUTION = auto()


@dataclass
class UIParameters:
    """UI-related parameters controlled via the UI."""
    # Window mode
    window_mode: WindowMode = WindowMode.MANUAL
    # Number of images
    number_of_images: int = 10
    # Analysis interval in seconds (used if window_mode is TIME_BASED)
    analysis_interval_seconds: int = 8
    # Threshold parameters
    mean_pixel_slope_threshold: float = 0.5
    match_score_threshold: float = 0.075
    maximum_pixels_threshold: float = 4.0
    confirmation_rounds: int = 3

    # Criteria enabled states (checkboxes in pattern results group box)
    mean_slope_enabled: bool = True
    match_score_enabled: bool = True
    percent_pixels_enabled: bool = True

    # UI options
    show_grayscale_images: bool = True
    save_data: bool = False

    def to_dict(self):
        """Convert to dictionary for each signal emission."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "UIParameters":
        """Create from dictionary (e.g. from controls group box)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ProcessingParameters:
    """Internal parameters not exposed to the UI."""
    # Crop areas (independent per pattern - RTM data comes separately for each)
    crop_rect_pattern_1: QRect | None = None
    crop_rect_pattern_2: QRect | None = None

    acquisition_delay_seconds: float = 0.2                          # This is an arbitrary delay time to accommodate data processing
    percent_difference_threshold: float = 20.0                      # Percentage drop in mean pixel value to disengage delay
    # Image processing parameters
    gaussian_sigma: float = 1.0                                     # Gaussian blur strength for background subtraction
    apply_dilation: bool = True                                     # Apply morphological dilation during background subtraction
    threshold_num_classes: int = 2                                  # Number of classes for multi-Otsu thresholding
    # Image analysis parameters
    num_points_for_slope: int = 3                                   # Number of points for gradient-based slope calculation
    linear_regression_fit_points: int = 3                           # Number of recent points to fit for linear regression slope method
    slope_method: SlopeMethod = SlopeMethod.GRADIENT                # Method for calculating mean pixel value slope
    binarization_method: BinarizationMethod = BinarizationMethod.FROZEN_MID  # How the foreground is isolated for the percent-pixels metric (incl. TOPHAT_ENERGY)
    tophat_radius: int = 5                                          # White top-hat disk radius (px) = max foreground feature scale; only used when binarization_method=TOPHAT_ENERGY
    match_on_foreground: bool = True                               # Compute the match score on the normalized top-hat foreground map instead of the grayscale change map (independent of binarization_method). Default True: the only field-validated matching path (2026-08-17 campaign)
    foreground_completion_mode: ForegroundCompletionMode = ForegroundCompletionMode.ABSOLUTE_PLUS_STALL  # How the foreground criterion decides "done" (plain ABSOLUTE threshold, or ABSOLUTE plus the grid-bar stall latch)
    stall_window: int = 16                                         # Stall latch: consecutive monitoring rounds the smoothed foreground trace must stay flat (~95 s at the 5 s default cadence)
    stall_drop_fraction: float = 0.35                              # Stall latch: smoothed trace must be <= this fraction of its running peak before the latch can arm
    stall_rel_tolerance: float = 0.10                              # Stall latch: relative flatness tolerance (fraction of the current smoothed value)
    stall_abs_tolerance: float = 0.15                              # Stall latch: absolute flatness tolerance (foreground units); the larger of the two tolerances governs
    aspect_ratio_threshold: float = 0.3                             # Minimum aspect ratio for valid patterns (filters stress relief cuts)
    # Pattern matching parameters
    min_pattern_splits: int = 3                                     # Minimum number of splits per dimension for pattern matching
    target_tile_size: int = 100                                     # Target tile dimension in pixels (creates ~100x100 pixel tiles)
    # Auto contrast/brightness calibration (one-shot at patterning start, then held static)
    auto_cb_on_start: bool = True                                   # On by default; runs the CB controller once when patterning starts
    cb_white_level: float = 255.0                                   # FALLBACK full-scale; calibration auto-detects from imaging bit depth (2^bits-1) when readable
    cb_target_median_fraction: float = 0.45                         # Desired median brightness as a fraction of full-scale
    cb_target_contrast_span: float = 0.55                           # Desired robust occupied span (p2-p98) as a fraction of full-scale; the contrast setpoint
    cb_max_white_clip_fraction: float = 0.02                        # Max acceptable fraction of pixels clipped at the ceiling (white saturation destroys the top-hat texture signal)
    cb_max_black_clip_fraction: float = 0.05                        # Max acceptable fraction of pixels clipped at the floor (black pixels are background the top-hat removes anyway)
    cb_min_bound: float = 0.0                                       # Lower clamp for normalized detector CB writes
    cb_max_bound: float = 1.0                                       # Upper clamp for normalized detector CB writes
    cb_max_iterations: int = 12                                     # Measurement budget (verify included) before locking best-so-far; loop exits early on acceptance
    cb_settle_seconds: float = 0.2                                  # Settle after each CB write before re-grabbing RTM (the RTM is no longer restarted per measurement)
    cb_frames_per_measurement: int = 2                             # Frames pooled for ACCEPT/VERIFY measurements; search measurements always use 1 frame


@dataclass
class SpinBoxConstraints:
    """Constraints for UI spin boxes."""
    # Number of images
    number_of_images_min: int = 3
    number_of_images_max: int = 1000
    number_of_images_decimals: int = 0
    number_of_images_single_step: int = 1
    # Number of seconds
    analysis_interval_seconds_min: int = 1
    analysis_interval_seconds_max: int = 1000
    analysis_interval_seconds_decimals: int = 0
    analysis_interval_seconds_single_step: int = 1
    # Mean pixel slope threshold
    mean_pixel_slope_threshold_min: float = 0.001
    mean_pixel_slope_threshold_max: float = 9.999
    mean_pixel_slope_threshold_decimals: int = 3
    mean_pixel_slope_threshold_single_step: float = 0.001
    # Match score threshold
    match_score_threshold_min: float = 0.000
    match_score_threshold_max: float = 1.000
    match_score_threshold_decimals: int = 3
    match_score_threshold_single_step: float = 0.001
    # Maximum pixels threshold
    maximum_pixels_threshold_min: float = 0.1
    maximum_pixels_threshold_max: float = 10.0
    maximum_pixels_threshold_decimals: int = 1
    maximum_pixels_threshold_single_step: float = 0.1
    # Confirmation rounds
    confirmation_rounds_min: int = 1
    confirmation_rounds_max: int = 10
    confirmation_rounds_decimals: int = 0
    confirmation_rounds_single_step: int = 1
    # Acquisition delay seconds
    acquisition_delay_seconds_min: float = 0.0
    acquisition_delay_seconds_max: float = 10.0
    acquisition_delay_seconds_decimals: int = 1
    acquisition_delay_seconds_single_step: float = 0.1
    # Percent difference threshold
    percent_difference_threshold_min: float = 1.0
    percent_difference_threshold_max: float = 100.0
    percent_difference_threshold_decimals: int = 1
    percent_difference_threshold_single_step: float = 1.0
    # Gaussian sigma
    gaussian_sigma_min: float = 0.1
    gaussian_sigma_max: float = 10.0
    gaussian_sigma_decimals: int = 1
    gaussian_sigma_single_step: float = 0.1
    # Threshold number of classes
    threshold_num_classes_min: int = 2
    threshold_num_classes_max: int = 10
    threshold_num_classes_decimals: int = 0
    threshold_num_classes_single_step: int = 1
    # Top-hat disk radius (max foreground feature scale, px)
    tophat_radius_min: int = 1
    tophat_radius_max: int = 25
    tophat_radius_decimals: int = 0
    tophat_radius_single_step: int = 1
    # Foreground stall latch: flat-window length in monitoring rounds
    stall_window_min: int = 4
    stall_window_max: int = 100
    stall_window_single_step: int = 1
    # Foreground stall latch: drop-from-running-peak fraction required to arm
    stall_drop_fraction_min: float = 0.05
    stall_drop_fraction_max: float = 1.0
    stall_drop_fraction_decimals: int = 2
    stall_drop_fraction_single_step: float = 0.05
    # Foreground stall latch: relative flatness tolerance (fraction of value)
    stall_rel_tolerance_min: float = 0.0
    stall_rel_tolerance_max: float = 1.0
    stall_rel_tolerance_decimals: int = 2
    stall_rel_tolerance_single_step: float = 0.01
    # Foreground stall latch: absolute flatness tolerance (foreground units)
    stall_abs_tolerance_min: float = 0.0
    stall_abs_tolerance_max: float = 10.0
    stall_abs_tolerance_decimals: int = 2
    stall_abs_tolerance_single_step: float = 0.05
    # Number of points for slope calculation (gradient method)
    num_points_for_slope_min: int = 2
    num_points_for_slope_max: int = 10
    num_points_for_slope_decimals: int = 0
    num_points_for_slope_single_step: int = 1
    # Number of points for linear regression fit
    linear_regression_fit_points_min: int = 2
    linear_regression_fit_points_max: int = 10
    linear_regression_fit_points_decimals: int = 0
    linear_regression_fit_points_single_step: int = 1
    # Aspect ratio threshold    
    aspect_ratio_threshold_min: float = 0.0
    aspect_ratio_threshold_max: float = 1.0
    aspect_ratio_threshold_decimals: int = 2
    aspect_ratio_threshold_single_step: float = 0.05
    # Minimum pattern splits
    min_pattern_splits_min: int = 1
    min_pattern_splits_max: int = 10
    min_pattern_splits_decimals: int = 0
    min_pattern_splits_single_step: int = 1
    # Target tile size
    target_tile_size_min: int = 10
    target_tile_size_max: int = 500
    target_tile_size_decimals: int = 0
    target_tile_size_single_step: int = 10
    # Auto CB: detector full-scale / saturation ceiling
    cb_white_level_min: float = 1.0
    cb_white_level_max: float = 65535.0
    cb_white_level_decimals: int = 0
    cb_white_level_single_step: float = 1.0
    # Auto CB: target median brightness (fraction of full-scale)
    cb_target_median_fraction_min: float = 0.05
    cb_target_median_fraction_max: float = 0.95
    cb_target_median_fraction_decimals: int = 2
    cb_target_median_fraction_single_step: float = 0.05
    # Auto CB: target contrast span (robust occupied fraction of full-scale)
    cb_target_contrast_span_min: float = 0.20
    cb_target_contrast_span_max: float = 0.95
    cb_target_contrast_span_decimals: int = 2
    cb_target_contrast_span_single_step: float = 0.05
    # Auto CB: max white clip fraction
    cb_max_white_clip_fraction_min: float = 0.0
    cb_max_white_clip_fraction_max: float = 0.8
    cb_max_white_clip_fraction_decimals: int = 3
    cb_max_white_clip_fraction_single_step: float = 0.005
    # Auto CB: max black clip fraction
    cb_max_black_clip_fraction_min: float = 0.0
    cb_max_black_clip_fraction_max: float = 0.8
    cb_max_black_clip_fraction_decimals: int = 3
    cb_max_black_clip_fraction_single_step: float = 0.005
    # Auto CB: lower CB clamp bound
    cb_min_bound_min: float = 0.0
    cb_min_bound_max: float = 1.0
    cb_min_bound_decimals: int = 2
    cb_min_bound_single_step: float = 0.05
    # Auto CB: upper CB clamp bound
    cb_max_bound_min: float = 0.0
    cb_max_bound_max: float = 1.0
    cb_max_bound_decimals: int = 2
    cb_max_bound_single_step: float = 0.05
    # Auto CB: max controller iterations
    cb_max_iterations_min: int = 1
    cb_max_iterations_max: int = 50
    cb_max_iterations_decimals: int = 0
    cb_max_iterations_single_step: int = 1
    # Auto CB: settle seconds after each CB write
    cb_settle_seconds_min: float = 0.0
    cb_settle_seconds_max: float = 10.0
    cb_settle_seconds_decimals: int = 1
    cb_settle_seconds_single_step: float = 0.1
    # Auto CB: frames averaged per measurement
    cb_frames_per_measurement_min: int = 1
    cb_frames_per_measurement_max: int = 20
    cb_frames_per_measurement_decimals: int = 0
    cb_frames_per_measurement_single_step: int = 1



@dataclass
class RTMParameters:
    """Container with all parameter types."""
    ui: UIParameters = field(default_factory=UIParameters)
    processing: ProcessingParameters = field(default_factory=ProcessingParameters)
    spin_box_constraints: SpinBoxConstraints = field(default_factory=SpinBoxConstraints)

    def get_ui_dict(self) -> dict:
        """Get UI parameters as dictionary."""
        return self.ui.to_dict()
    
    def update_ui_from_dict(self, data: dict):
        """Update UI parameters from dictionary."""
        for key, value in data.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
    
    def update_crop_rect_pattern_1(self, x: int, y: int, width: int, height: int):
        """Update Pattern 1 crop rectangle."""
        self.processing.crop_rect_pattern_1 = QRect(x, y, width, height)
    
    def update_crop_rect_pattern_2(self, x: int, y: int, width: int, height: int):
        """Update Pattern 2 crop rectangle."""
        self.processing.crop_rect_pattern_2 = QRect(x, y, width, height)

    # ===========================
    # QSettings Persistence
    # ===========================

    def save_settings(self):
        """
        Save user-facing parameters to QSettings (Windows registry).
        
        Persists UI parameters (spinbox values, criteria checkboxes, options)
        and crop rectangles. Processing parameters are NOT saved — they are
        code-level constants set directly in the dataclass defaults.
        """
        settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        
        # --- UI Parameters ---
        settings.beginGroup("ui")
        settings.setValue("window_mode", self.ui.window_mode.name)
        settings.setValue("number_of_images", self.ui.number_of_images)
        settings.setValue("analysis_interval_seconds", self.ui.analysis_interval_seconds)
        settings.setValue("mean_pixel_slope_threshold", self.ui.mean_pixel_slope_threshold)
        settings.setValue("match_score_threshold", self.ui.match_score_threshold)
        settings.setValue("maximum_pixels_threshold", self.ui.maximum_pixels_threshold)
        settings.setValue("confirmation_rounds", self.ui.confirmation_rounds)
        settings.setValue("show_grayscale_images", self.ui.show_grayscale_images)
        settings.setValue("save_data", self.ui.save_data)
        settings.endGroup()
        
        # --- Criteria Checkboxes ---
        settings.beginGroup("criteria")
        settings.setValue("mean_slope_enabled", self.ui.mean_slope_enabled)
        settings.setValue("match_score_enabled", self.ui.match_score_enabled)
        settings.setValue("percent_pixels_enabled", self.ui.percent_pixels_enabled)
        settings.endGroup()
        
        # --- Crop Rectangles ---
        settings.beginGroup("crop")
        for i, rect in enumerate([
            self.processing.crop_rect_pattern_1,
            self.processing.crop_rect_pattern_2
        ], 1):
            if rect is not None:
                settings.setValue(f"pattern_{i}_x", rect.x())
                settings.setValue(f"pattern_{i}_y", rect.y())
                settings.setValue(f"pattern_{i}_width", rect.width())
                settings.setValue(f"pattern_{i}_height", rect.height())
            else:
                settings.remove(f"pattern_{i}_x")
                settings.remove(f"pattern_{i}_y")
                settings.remove(f"pattern_{i}_width")
                settings.remove(f"pattern_{i}_height")
        settings.endGroup()

        # --- Processing Parameters ---
        settings.beginGroup("processing")
        settings.setValue("acquisition_delay_seconds", self.processing.acquisition_delay_seconds)
        settings.setValue("percent_difference_threshold", self.processing.percent_difference_threshold)
        settings.setValue("gaussian_sigma", self.processing.gaussian_sigma)
        settings.setValue("apply_dilation", self.processing.apply_dilation)
        settings.setValue("threshold_num_classes", self.processing.threshold_num_classes)
        settings.setValue("tophat_radius", self.processing.tophat_radius)
        settings.setValue("match_on_foreground", self.processing.match_on_foreground)
        settings.setValue("num_points_for_slope", self.processing.num_points_for_slope)
        settings.setValue("linear_regression_fit_points", self.processing.linear_regression_fit_points)
        settings.setValue("slope_method", self.processing.slope_method.name)
        settings.setValue("binarization_method", self.processing.binarization_method.name)
        settings.setValue("foreground_completion_mode", self.processing.foreground_completion_mode.name)
        settings.setValue("stall_window", self.processing.stall_window)
        settings.setValue("stall_drop_fraction", self.processing.stall_drop_fraction)
        settings.setValue("stall_rel_tolerance", self.processing.stall_rel_tolerance)
        settings.setValue("stall_abs_tolerance", self.processing.stall_abs_tolerance)
        # Marks the 3.3.4 foreground-completion migration as applied (see
        # load_settings). The old energy_drop_fraction / energy_slope_threshold
        # keys are intentionally never reused: a persisted 0.25 would silently
        # shadow the stall defaults.
        settings.setValue("fg_schema_version", 2)
        settings.setValue("aspect_ratio_threshold", self.processing.aspect_ratio_threshold)
        settings.setValue("min_pattern_splits", self.processing.min_pattern_splits)
        settings.setValue("target_tile_size", self.processing.target_tile_size)
        # Auto contrast/brightness calibration
        settings.setValue("auto_cb_on_start", self.processing.auto_cb_on_start)
        settings.setValue("cb_white_level", self.processing.cb_white_level)
        settings.setValue("cb_target_median_fraction", self.processing.cb_target_median_fraction)
        settings.setValue("cb_target_contrast_span", self.processing.cb_target_contrast_span)
        settings.setValue("cb_max_white_clip_fraction", self.processing.cb_max_white_clip_fraction)
        settings.setValue("cb_max_black_clip_fraction", self.processing.cb_max_black_clip_fraction)
        settings.setValue("cb_min_bound", self.processing.cb_min_bound)
        settings.setValue("cb_max_bound", self.processing.cb_max_bound)
        settings.setValue("cb_max_iterations", self.processing.cb_max_iterations)
        settings.setValue("cb_settle_seconds", self.processing.cb_settle_seconds)
        settings.setValue("cb_frames_per_measurement", self.processing.cb_frames_per_measurement)
        # Marks the 3.3.1 CB-retune migration as applied (see load_settings).
        settings.setValue("cb_schema_version", 2)
        settings.endGroup()

        logger.info("Settings saved")
    
    def load_settings(self):
        """
        Load user-facing parameters from QSettings.
        
        Restores UI parameters (spinbox values, criteria checkboxes, options)
        and crop rectangles. Processing parameters are left as their dataclass
        defaults — change those directly in the source code.
        
        Falls back to defaults for any missing or invalid values.
        """
        settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        defaults_ui = UIParameters()
        
        # --- UI Parameters ---
        settings.beginGroup("ui")
        
        mode_name = settings.value("window_mode", defaults_ui.window_mode.name)
        self.ui.window_mode = _enum_from_name(WindowMode, mode_name, defaults_ui.window_mode)
        
        self.ui.number_of_images = int(
            settings.value("number_of_images", defaults_ui.number_of_images))
        self.ui.analysis_interval_seconds = int(
            settings.value("analysis_interval_seconds", defaults_ui.analysis_interval_seconds))
        self.ui.mean_pixel_slope_threshold = float(
            settings.value("mean_pixel_slope_threshold", defaults_ui.mean_pixel_slope_threshold))
        self.ui.match_score_threshold = float(
            settings.value("match_score_threshold", defaults_ui.match_score_threshold))
        self.ui.maximum_pixels_threshold = float(
            settings.value("maximum_pixels_threshold", defaults_ui.maximum_pixels_threshold))
        self.ui.confirmation_rounds = int(
            settings.value("confirmation_rounds", defaults_ui.confirmation_rounds))
        self.ui.show_grayscale_images = _bool_from_settings(
            settings.value("show_grayscale_images", defaults_ui.show_grayscale_images))
        self.ui.save_data = _bool_from_settings(
            settings.value("save_data", defaults_ui.save_data))
        
        settings.endGroup()
        
        # --- Criteria Checkboxes ---
        settings.beginGroup("criteria")
        
        self.ui.mean_slope_enabled = _bool_from_settings(
            settings.value("mean_slope_enabled", defaults_ui.mean_slope_enabled))
        self.ui.match_score_enabled = _bool_from_settings(
            settings.value("match_score_enabled", defaults_ui.match_score_enabled))
        self.ui.percent_pixels_enabled = _bool_from_settings(
            settings.value("percent_pixels_enabled", defaults_ui.percent_pixels_enabled))
        
        settings.endGroup()
        
        # --- Crop Rectangles ---
        settings.beginGroup("crop")
        
        for i, attr in enumerate([
            "crop_rect_pattern_1", "crop_rect_pattern_2"
        ], 1):
            if settings.contains(f"pattern_{i}_x"):
                x = int(settings.value(f"pattern_{i}_x"))
                y = int(settings.value(f"pattern_{i}_y"))
                w = int(settings.value(f"pattern_{i}_width"))
                h = int(settings.value(f"pattern_{i}_height"))
                setattr(self.processing, attr, QRect(x, y, w, h))
        
        settings.endGroup()

        # --- Processing Parameters ---
        defaults_proc = ProcessingParameters()
        settings.beginGroup("processing")
        
        self.processing.acquisition_delay_seconds = float(
            settings.value("acquisition_delay_seconds", defaults_proc.acquisition_delay_seconds))
        self.processing.percent_difference_threshold = float(
            settings.value("percent_difference_threshold", defaults_proc.percent_difference_threshold))
        self.processing.gaussian_sigma = float(
            settings.value("gaussian_sigma", defaults_proc.gaussian_sigma))
        self.processing.apply_dilation = _bool_from_settings(
            settings.value("apply_dilation", defaults_proc.apply_dilation))
        self.processing.threshold_num_classes = int(
            settings.value("threshold_num_classes", defaults_proc.threshold_num_classes))
        self.processing.tophat_radius = int(
            settings.value("tophat_radius", defaults_proc.tophat_radius))
        self.processing.match_on_foreground = _bool_from_settings(
            settings.value("match_on_foreground", defaults_proc.match_on_foreground))
        self.processing.num_points_for_slope = int(
            settings.value("num_points_for_slope", defaults_proc.num_points_for_slope))
        self.processing.linear_regression_fit_points = int(
            settings.value("linear_regression_fit_points", defaults_proc.linear_regression_fit_points))
        
        slope_name = settings.value("slope_method", defaults_proc.slope_method.name)
        self.processing.slope_method = _enum_from_name(
            SlopeMethod, slope_name, defaults_proc.slope_method)

        binarization_name = settings.value(
            "binarization_method", defaults_proc.binarization_method.name)
        self.processing.binarization_method = _enum_from_name(
            BinarizationMethod, binarization_name, defaults_proc.binarization_method)

        # One-time migration (fg schema v2, ATC Monitor 3.3.4): the stall latch
        # replaced RELATIVE_PLATEAU and became part of the default mode. Every
        # install that ran 3.3.x has "ABSOLUTE" persisted (the old default),
        # and stored values win over dataclass defaults -- so without this the
        # grid-bar backstop would silently never deploy on existing tools. A
        # mode the user changes after the migration is respected like any
        # other setting. (A persisted "RELATIVE_PLATEAU" no longer names an
        # enum member and falls back to the default via _enum_from_name.)
        fg_migrated = int(settings.value("fg_schema_version", 1)) < 2
        if fg_migrated:
            logger.info(
                "Foreground completion schema upgrade: enabling the "
                "ABSOLUTE_PLUS_STALL default (grid-bar stall latch) and the "
                "field-validated foreground matching path"
            )
            self.processing.foreground_completion_mode = (
                defaults_proc.foreground_completion_mode)
            # match_on_foreground has been persisted since 3.3.0 with the old
            # False default, and stored values win over dataclass defaults --
            # without this one-time force, the v3.3.4 default flip to the only
            # field-validated matching path would silently never deploy on any
            # tool upgraded in place (the same trap this migration exists for).
            # This assignment intentionally OVERRIDES the unconditional read
            # earlier in load_settings; a value the user changes after the
            # migration is respected like any other setting.
            self.processing.match_on_foreground = (
                defaults_proc.match_on_foreground)
        else:
            foreground_mode_name = settings.value(
                "foreground_completion_mode",
                defaults_proc.foreground_completion_mode.name)
            self.processing.foreground_completion_mode = _enum_from_name(
                ForegroundCompletionMode, foreground_mode_name,
                defaults_proc.foreground_completion_mode)
        self.processing.stall_window = int(
            settings.value("stall_window", defaults_proc.stall_window))
        self.processing.stall_drop_fraction = float(
            settings.value("stall_drop_fraction", defaults_proc.stall_drop_fraction))
        self.processing.stall_rel_tolerance = float(
            settings.value("stall_rel_tolerance", defaults_proc.stall_rel_tolerance))
        self.processing.stall_abs_tolerance = float(
            settings.value("stall_abs_tolerance", defaults_proc.stall_abs_tolerance))

        self.processing.aspect_ratio_threshold = float(
            settings.value("aspect_ratio_threshold", defaults_proc.aspect_ratio_threshold))
        self.processing.min_pattern_splits = int(
            settings.value("min_pattern_splits", defaults_proc.min_pattern_splits))
        self.processing.target_tile_size = int(
            settings.value("target_tile_size", defaults_proc.target_tile_size))

        # Auto contrast/brightness calibration
        self.processing.auto_cb_on_start = _bool_from_settings(
            settings.value("auto_cb_on_start", defaults_proc.auto_cb_on_start))
        self.processing.cb_white_level = float(
            settings.value("cb_white_level", defaults_proc.cb_white_level))
        self.processing.cb_target_median_fraction = float(
            settings.value("cb_target_median_fraction", defaults_proc.cb_target_median_fraction))
        self.processing.cb_target_contrast_span = float(
            settings.value("cb_target_contrast_span", defaults_proc.cb_target_contrast_span))
        # One-time migration (schema v2, ATC Monitor 3.3.1): the acceptance/
        # latency retune changed these four defaults (clip 0.01/0.01 ->
        # 0.02/0.05, settle 0.5 -> 0.2, verify frames 3 -> 2). Every install
        # that ran 3.3.0 has the OLD values persisted, and stored values win
        # over dataclass defaults -- so without this the retune would silently
        # never deploy on the very tools it was built for. The version key is
        # bumped once (in save_settings); values the user changes after the
        # migration are respected like any other setting.
        cb_migrated = int(settings.value("cb_schema_version", 1)) < 2
        if cb_migrated:
            logger.info(
                "Auto CB settings schema upgrade: applying the 3.3.1 retuned "
                "defaults for clip limits, settle time, and verify frames"
            )
            self.processing.cb_max_white_clip_fraction = defaults_proc.cb_max_white_clip_fraction
            self.processing.cb_max_black_clip_fraction = defaults_proc.cb_max_black_clip_fraction
            self.processing.cb_settle_seconds = defaults_proc.cb_settle_seconds
            self.processing.cb_frames_per_measurement = defaults_proc.cb_frames_per_measurement
        else:
            self.processing.cb_max_white_clip_fraction = float(
                settings.value("cb_max_white_clip_fraction", defaults_proc.cb_max_white_clip_fraction))
            self.processing.cb_max_black_clip_fraction = float(
                settings.value("cb_max_black_clip_fraction", defaults_proc.cb_max_black_clip_fraction))
            self.processing.cb_settle_seconds = float(
                settings.value("cb_settle_seconds", defaults_proc.cb_settle_seconds))
            self.processing.cb_frames_per_measurement = int(
                settings.value("cb_frames_per_measurement", defaults_proc.cb_frames_per_measurement))
        self.processing.cb_min_bound = float(
            settings.value("cb_min_bound", defaults_proc.cb_min_bound))
        self.processing.cb_max_bound = float(
            settings.value("cb_max_bound", defaults_proc.cb_max_bound))
        self.processing.cb_max_iterations = int(
            settings.value("cb_max_iterations", defaults_proc.cb_max_iterations))

        settings.endGroup()
        
        logger.info("Settings loaded")


def _enum_from_name(enum_class, name, default):
    """
    Safely convert a string name to an enum member.
    
    :param enum_class: The Enum class to look up
    :param name: String name of the enum member
    :param default: Default value if name is invalid
    :return: Enum member or default
    """
    try:
        return enum_class[name]
    except (KeyError, TypeError):
        logger.warning(
            f"Invalid {enum_class.__name__} value '{name}', using default: {default.name}"
        )
        return default


def _bool_from_settings(value):
    """
    Convert QSettings value to bool.
    
    QSettings may return 'true'/'false' strings on some platforms.
    
    :param value: Value from QSettings
    :return: Boolean
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == 'true'
    return bool(value)


def create_default_parameters() -> RTMParameters:
    """Factory function to create default parameters."""
    return RTMParameters()


def load_parameters() -> RTMParameters:
    """
    Factory function to create parameters and load saved settings.
    
    Falls back to defaults for any missing or invalid values.
    """
    params = RTMParameters()
    params.load_settings()
    return params