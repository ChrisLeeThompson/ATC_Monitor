"""
Module for spin box components.
"""
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtCore import Qt
from script_modules.app_styles import AppStyles
from script_modules.atc_monitor_parameters import SpinBoxConstraints


class StyledDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(AppStyles.SpinBox.default())
        self.setMinimumWidth(AppStyles.Dimensions.SPINBOX_WIDTH)
        # Store parent reference for focus transfer
        self._parent = parent
    
    def keyPressEvent(self, event):
        """Handle key press events, transfer focus on Enter or Return."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._parent:
                self._parent.setFocus()
        else:
            super().keyPressEvent(event)

# ==========================
# Controls spin boxes
# ==========================

class NumberOfImagesSpinBox(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 10):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.number_of_images_min)
            self.setMaximum(constraints.number_of_images_max)
            self.setDecimals(constraints.number_of_images_decimals)
            self.setSingleStep(constraints.number_of_images_single_step)
        else:
            self.setMinimum(3)
            self.setMaximum(100)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class NumberOfSecondsSpinBox(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 4):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.analysis_interval_seconds_min)
            self.setMaximum(constraints.analysis_interval_seconds_max)
            self.setDecimals(constraints.analysis_interval_seconds_decimals)
            self.setSingleStep(constraints.analysis_interval_seconds_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(1000)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)
    

class MeanPixelSlopeThreshold(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 0.75):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.mean_pixel_slope_threshold_min)
            self.setMaximum(constraints.mean_pixel_slope_threshold_max)
            self.setDecimals(constraints.mean_pixel_slope_threshold_decimals)
            self.setSingleStep(constraints.mean_pixel_slope_threshold_single_step)
        else:
            self.setMinimum(0.001)
            self.setMaximum(9.999)
            self.setDecimals(3)
            self.setSingleStep(0.001)
        self.setValue(initial_value)


class MatchScoreThreshold(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 0.075):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.match_score_threshold_min)
            self.setMaximum(constraints.match_score_threshold_max)
            self.setDecimals(constraints.match_score_threshold_decimals)
            self.setSingleStep(constraints.match_score_threshold_single_step)
        else:
            self.setMinimum(0.0000)
            self.setMaximum(1.0000)
            self.setDecimals(3)
            self.setSingleStep(0.0001)
        self.setValue(initial_value)


class MaximumPixelsThreshold(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 4.0):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.maximum_pixels_threshold_min)
            self.setMaximum(constraints.maximum_pixels_threshold_max)
            self.setDecimals(constraints.maximum_pixels_threshold_decimals)
            self.setSingleStep(constraints.maximum_pixels_threshold_single_step)
        else:
            self.setMinimum(0.1)
            self.setMaximum(10.0)
            self.setDecimals(1)  
            self.setSingleStep(0.1)
        self.setValue(initial_value)


class ConfirmationRounds(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 3):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.confirmation_rounds_min)
            self.setMaximum(constraints.confirmation_rounds_max)
            self.setDecimals(constraints.confirmation_rounds_decimals)
            self.setSingleStep(constraints.confirmation_rounds_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(10)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)

# ==========================
# Settings dialog spin boxes
# ==========================

class AcquisitionDelay(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 0.2):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.acquisition_delay_seconds_min)
            self.setMaximum(constraints.acquisition_delay_seconds_max)
            self.setDecimals(constraints.acquisition_delay_seconds_decimals)
            self.setSingleStep(constraints.acquisition_delay_seconds_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(10.0)
            self.setDecimals(1)
            self.setSingleStep(0.1)
        self.setValue(initial_value)


class PercentDifferenceThreshold(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 20.0):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.percent_difference_threshold_min)
            self.setMaximum(constraints.percent_difference_threshold_max)
            self.setDecimals(constraints.percent_difference_threshold_decimals)
            self.setSingleStep(constraints.percent_difference_threshold_single_step)
        else:
            self.setMinimum(1.0)
            self.setMaximum(100.0)
            self.setDecimals(1)
            self.setSingleStep(1.0)
        self.setValue(initial_value)


class GaussianSigma(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 1.0):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.gaussian_sigma_min)
            self.setMaximum(constraints.gaussian_sigma_max)
            self.setDecimals(constraints.gaussian_sigma_decimals)
            self.setSingleStep(constraints.gaussian_sigma_single_step)
        else:
            self.setMinimum(0.1)
            self.setMaximum(10.0)
            self.setDecimals(1)
            self.setSingleStep(0.1)
        self.setValue(initial_value)


class ThresholdNumberOfClasses(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 2):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.threshold_num_classes_min)
            self.setMaximum(constraints.threshold_num_classes_max)
            self.setDecimals(constraints.threshold_num_classes_decimals)
            self.setSingleStep(constraints.threshold_num_classes_single_step)
        else:
            self.setMinimum(2)
            self.setMaximum(10)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class TophatRadius(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 5):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.tophat_radius_min)
            self.setMaximum(constraints.tophat_radius_max)
            self.setDecimals(constraints.tophat_radius_decimals)
            self.setSingleStep(constraints.tophat_radius_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(25)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class StallWindow(StyledDoubleSpinBox):
    """Stall latch: flat-window length in monitoring rounds (integer)."""

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 16):
        super().__init__(parent)
        # Integer parameter: constraints define no decimals field, so 0 is fixed here.
        self.setDecimals(0)
        if constraints:
            self.setMinimum(constraints.stall_window_min)
            self.setMaximum(constraints.stall_window_max)
            self.setSingleStep(constraints.stall_window_single_step)
        else:
            self.setMinimum(4)
            self.setMaximum(100)
            self.setSingleStep(1)
        self.setValue(initial_value)


class StallDropFraction(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.35):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.stall_drop_fraction_min)
            self.setMaximum(constraints.stall_drop_fraction_max)
            self.setDecimals(constraints.stall_drop_fraction_decimals)
            self.setSingleStep(constraints.stall_drop_fraction_single_step)
        else:
            self.setMinimum(0.05)
            self.setMaximum(1.0)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class StallRelTolerance(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.10):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.stall_rel_tolerance_min)
            self.setMaximum(constraints.stall_rel_tolerance_max)
            self.setDecimals(constraints.stall_rel_tolerance_decimals)
            self.setSingleStep(constraints.stall_rel_tolerance_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(1.0)
            self.setDecimals(2)
            self.setSingleStep(0.01)
        self.setValue(initial_value)


class StallAbsTolerance(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.15):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.stall_abs_tolerance_min)
            self.setMaximum(constraints.stall_abs_tolerance_max)
            self.setDecimals(constraints.stall_abs_tolerance_decimals)
            self.setSingleStep(constraints.stall_abs_tolerance_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(10.0)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class NumberOfPointsForGradientSlope(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 3):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.num_points_for_slope_min)
            self.setMaximum(constraints.num_points_for_slope_max)
            self.setDecimals(constraints.num_points_for_slope_decimals)
            self.setSingleStep(constraints.num_points_for_slope_single_step)
        else:
            self.setMinimum(2)
            self.setMaximum(20)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class LinearRegressionFitPoints(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 3):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.linear_regression_fit_points_min)
            self.setMaximum(constraints.linear_regression_fit_points_max)
            self.setDecimals(constraints.linear_regression_fit_points_decimals)
            self.setSingleStep(constraints.linear_regression_fit_points_single_step)
        else:
            self.setMinimum(2)
            self.setMaximum(10)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class AspectRatioThreshold(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: float = 0.3):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.aspect_ratio_threshold_min)
            self.setMaximum(constraints.aspect_ratio_threshold_max)
            self.setDecimals(constraints.aspect_ratio_threshold_decimals)
            self.setSingleStep(constraints.aspect_ratio_threshold_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(1.0)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class MinimumNumberOfImageRegionSplits(StyledDoubleSpinBox):
    """Minimum number of splits per dimension for pattern matching.
    For example, if min_pattern_splits = 2, then the RTM image is split into a 2x2 grid.
    """

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_value: int = 2):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.min_pattern_splits_min)
            self.setMaximum(constraints.min_pattern_splits_max)
            self.setDecimals(constraints.min_pattern_splits_decimals)
            self.setSingleStep(constraints.min_pattern_splits_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(10)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class TargetImageRegionTileSize(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 100):
        super().__init__(parent)
        # Use constraints if provided, otherwise use defaults
        if constraints:
            self.setMinimum(constraints.target_tile_size_min)
            self.setMaximum(constraints.target_tile_size_max)
            self.setDecimals(constraints.target_tile_size_decimals)
            self.setSingleStep(constraints.target_tile_size_single_step)
        else:
            self.setMinimum(10)
            self.setMaximum(500)
            self.setDecimals(0)
            self.setSingleStep(10)
        self.setValue(initial_value)


# ==========================
# Auto contrast/brightness calibration spin boxes
# ==========================

class CBWhiteLevel(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 255.0):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_white_level_min)
            self.setMaximum(constraints.cb_white_level_max)
            self.setDecimals(constraints.cb_white_level_decimals)
            self.setSingleStep(constraints.cb_white_level_single_step)
        else:
            self.setMinimum(1.0)
            self.setMaximum(65535.0)
            self.setDecimals(0)
            self.setSingleStep(1.0)
        self.setValue(initial_value)


class CBTargetMedianFraction(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.45):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_target_median_fraction_min)
            self.setMaximum(constraints.cb_target_median_fraction_max)
            self.setDecimals(constraints.cb_target_median_fraction_decimals)
            self.setSingleStep(constraints.cb_target_median_fraction_single_step)
        else:
            self.setMinimum(0.05)
            self.setMaximum(0.95)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class CBTargetContrastSpan(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.55):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_target_contrast_span_min)
            self.setMaximum(constraints.cb_target_contrast_span_max)
            self.setDecimals(constraints.cb_target_contrast_span_decimals)
            self.setSingleStep(constraints.cb_target_contrast_span_single_step)
        else:
            self.setMinimum(0.20)
            self.setMaximum(0.95)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class CBMaxWhiteClipFraction(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.01):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_max_white_clip_fraction_min)
            self.setMaximum(constraints.cb_max_white_clip_fraction_max)
            self.setDecimals(constraints.cb_max_white_clip_fraction_decimals)
            self.setSingleStep(constraints.cb_max_white_clip_fraction_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(0.5)
            self.setDecimals(3)
            self.setSingleStep(0.005)
        self.setValue(initial_value)


class CBMaxBlackClipFraction(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.01):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_max_black_clip_fraction_min)
            self.setMaximum(constraints.cb_max_black_clip_fraction_max)
            self.setDecimals(constraints.cb_max_black_clip_fraction_decimals)
            self.setSingleStep(constraints.cb_max_black_clip_fraction_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(0.5)
            self.setDecimals(3)
            self.setSingleStep(0.005)
        self.setValue(initial_value)


class CBMinBound(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.0):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_min_bound_min)
            self.setMaximum(constraints.cb_min_bound_max)
            self.setDecimals(constraints.cb_min_bound_decimals)
            self.setSingleStep(constraints.cb_min_bound_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(1.0)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class CBMaxBound(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 1.0):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_max_bound_min)
            self.setMaximum(constraints.cb_max_bound_max)
            self.setDecimals(constraints.cb_max_bound_decimals)
            self.setSingleStep(constraints.cb_max_bound_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(1.0)
            self.setDecimals(2)
            self.setSingleStep(0.05)
        self.setValue(initial_value)


class CBMaxIterations(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 8):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_max_iterations_min)
            self.setMaximum(constraints.cb_max_iterations_max)
            self.setDecimals(constraints.cb_max_iterations_decimals)
            self.setSingleStep(constraints.cb_max_iterations_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(50)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)


class CBSettleSeconds(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: float = 0.5):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_settle_seconds_min)
            self.setMaximum(constraints.cb_settle_seconds_max)
            self.setDecimals(constraints.cb_settle_seconds_decimals)
            self.setSingleStep(constraints.cb_settle_seconds_single_step)
        else:
            self.setMinimum(0.0)
            self.setMaximum(10.0)
            self.setDecimals(1)
            self.setSingleStep(0.1)
        self.setValue(initial_value)


class CBFramesPerMeasurement(StyledDoubleSpinBox):

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 initial_value: int = 3):
        super().__init__(parent)
        if constraints:
            self.setMinimum(constraints.cb_frames_per_measurement_min)
            self.setMaximum(constraints.cb_frames_per_measurement_max)
            self.setDecimals(constraints.cb_frames_per_measurement_decimals)
            self.setSingleStep(constraints.cb_frames_per_measurement_single_step)
        else:
            self.setMinimum(1)
            self.setMaximum(20)
            self.setDecimals(0)
            self.setSingleStep(1)
        self.setValue(initial_value)