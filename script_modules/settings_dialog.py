"""
Settings Dialog

This module contains the implementation of the settings modal dialog for the ATC Monitor application.
"""
import logging
from dataclasses import fields
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox,
    QGroupBox, QSizePolicy,
    QCheckBox, QScrollArea, QWidget, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from script_modules.app_styles import AppStyles
from script_modules.atc_monitor_parameters import (
    ProcessingParameters, SlopeMethod, BinarizationMethod,
    ForegroundCompletionMode, SpinBoxConstraints
)
from script_modules.button_widgets import (
    SaveButton, CancelButton, RestoreDefaultsButton
)
from script_modules.spinbox_widgets import (
    AcquisitionDelay, PercentDifferenceThreshold, GaussianSigma,
    ThresholdNumberOfClasses, TophatRadius, NumberOfPointsForGradientSlope,
    LinearRegressionFitPoints, AspectRatioThreshold, MinimumNumberOfImageRegionSplits,
    TargetImageRegionTileSize, StallWindow, StallDropFraction,
    StallRelTolerance, StallAbsTolerance,
    CBWhiteLevel, CBLowerMargin, CBUpperMargin,
    CBMaxWhiteClipFraction, CBMaxBlackClipFraction, CBMinBound, CBMaxBound,
    CBMaxIterations, CBSettleSeconds, CBFramesPerMeasurement
)


logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):

    #: ProcessingParameters fields that may be edited while monitoring is
    #: running. The one-shot edge-margin correction is their only reader, once
    #: per patterning session: the margins, clip limits and bounds when it
    #: builds its CBEdgeConfig; the measurement budget, settle time and frame
    #: count as its loop runs; and auto_cb_on_start at the correction trigger,
    #: which reads it after the parameter queue drains (see _arm_cb_calibration
    #: -- read at the arming point instead, a mid-run enable was a silent
    #: no-op). So a change cannot disturb anything in flight, and takes effect
    #: at the next correction.
    #:
    #: Most of the other settings in this dialog are read per batch, so changing
    #: one mid-run would splice two incompatible scales into a single metric
    #: series that the completion criteria then read as continuous. A few are
    #: not (acquisition_delay_seconds is a per-iteration sleep;
    #: aspect_ratio_threshold is read once per session at pattern validation) and
    #: stay locked anyway -- the audit that produced this list only cleared the
    #: CB group, and widening it is a decision to make deliberately, per
    #: parameter, not a gap to close by pattern-matching on read cadence.
    LIVE_EDITABLE_FIELDS = (
        "auto_cb_on_start",
        "cb_recalibrate_every_session",
        "cb_white_level",
        "cb_lower_margin",
        "cb_upper_margin",
        "cb_max_white_clip_fraction",
        "cb_max_black_clip_fraction",
        "cb_min_bound",
        "cb_max_bound",
        "cb_max_iterations",
        "cb_settle_seconds",
        "cb_frames_per_measurement",
    )

    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None,
                 processing_params: ProcessingParameters | None = None,
                 monitoring_active: bool = False):
        super().__init__(parent)
        # While monitoring runs the dialog opens read-only except for the
        # Contrast/Brightness Calibration group (see LIVE_EDITABLE_FIELDS). The
        # operator can still see every current value, which they could not when
        # the whole dialog was locked behind a disabled button.
        self._monitoring_active = bool(monitoring_active)
        # Dialog title and width
        self.setWindowTitle(AppStyles.AppText.SETTINGS_DIALOG_TITLE)
        self.setMinimumWidth(AppStyles.Dimensions.SETTINGS_DIALOG_WIDTH)
        # Set modal style and size policy
        self.setStyleSheet(AppStyles.Dialog.modal())
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred
        )
        # Set window icon
        self.assets_base_path = Path(__file__).parent.parent / "script_assets"
        _icon_path = self.assets_base_path / "catbug_waiting_color.png"
        self.setWindowIcon(QIcon(str(_icon_path)))
        # Store defaults for restore defaults button
        self._defaults = ProcessingParameters()
        # Current values to populate fields
        self._initial = processing_params if processing_params else ProcessingParameters()
        # Constraints for spin boxes
        self._constraints = constraints if constraints else SpinBoxConstraints()
        # Create components and layout
        self._create_components()
        self._setup_layout()
        self._setup_connections()
        self._apply_monitoring_lock()
    
    def _create_components(self):
        ## Monitoring group
        # Acquisition delay
        self.acquisition_delay_spinbox = AcquisitionDelay(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.acquisition_delay_seconds
        )
        # Percent difference threshold
        self.percent_difference_threshold_spinbox = PercentDifferenceThreshold(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.percent_difference_threshold
        )
        ## Image processing group
        # Gaussian blur sigma
        self.gaussian_sigma_spinbox = GaussianSigma(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.gaussian_sigma
        )
        # Apply dilation checkbox
        self.apply_dilation_checkbox = QCheckBox()
        self.apply_dilation_checkbox.setStyleSheet(AppStyles.CheckBox.settings_dialog())
        self.apply_dilation_checkbox.setChecked(self._initial.apply_dilation)
        # Threshold number of classes
        self.threshold_number_of_classes_spinbox = ThresholdNumberOfClasses(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.threshold_num_classes
        )
        # Foreground isolation method (feeds the percent-pixels completion metric)
        self.binarization_method_combobox = self._create_enum_combobox(
            BinarizationMethod, {
                BinarizationMethod.TOP: "Brightest Class",
                BinarizationMethod.FROZEN_MID: "Middle Boundary",
                BinarizationMethod.FROZEN_LOW: "Low Boundary",
                BinarizationMethod.TOPHAT_ENERGY: "Top-Hat Foreground Energy",
            }
        )
        self.binarization_method_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.binarization_method_combobox.setCurrentIndex(
            self.binarization_method_combobox.findData(self._initial.binarization_method)
        )
        # Top-hat disk radius (only used when method = Top-Hat Foreground Energy)
        self.tophat_radius_spinbox = TophatRadius(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.tophat_radius
        )
        # Match on the top-hat foreground map (independent of binarization method)
        self.match_on_foreground_checkbox = QCheckBox()
        self.match_on_foreground_checkbox.setStyleSheet(AppStyles.CheckBox.settings_dialog())
        self.match_on_foreground_checkbox.setChecked(self._initial.match_on_foreground)
        # Foreground completion mode (plain absolute threshold vs absolute + grid-bar stall latch)
        self.foreground_completion_mode_combobox = self._create_enum_combobox(
            ForegroundCompletionMode, {
                ForegroundCompletionMode.ABSOLUTE: "Absolute (threshold)",
                ForegroundCompletionMode.ABSOLUTE_PLUS_STALL: "Absolute + stall latch",
            }
        )
        self.foreground_completion_mode_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.foreground_completion_mode_combobox.setCurrentIndex(
            self.foreground_completion_mode_combobox.findData(self._initial.foreground_completion_mode)
        )
        # Stall latch settings (only used when mode = Absolute + stall latch)
        self.stall_window_spinbox = StallWindow(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.stall_window
        )
        self.stall_drop_fraction_spinbox = StallDropFraction(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.stall_drop_fraction
        )
        self.stall_rel_tolerance_spinbox = StallRelTolerance(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.stall_rel_tolerance
        )
        self.stall_abs_tolerance_spinbox = StallAbsTolerance(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.stall_abs_tolerance
        )
        ## Image analysis group
        # Method combo box
        self.slope_method_combobox = self._create_enum_combobox(
            SlopeMethod, {
                SlopeMethod.GRADIENT: "Gradient",
                SlopeMethod.LINEAR_REGRESSION: "Linear Regression"
            }
        )
        self.slope_method_combobox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.slope_method_combobox.setCurrentIndex(
            self.slope_method_combobox.findData(self._initial.slope_method)
        )
        # Number of points for gradient slope
        self.number_of_points_for_gradient_slope_spinbox = NumberOfPointsForGradientSlope(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.num_points_for_slope
        )
        # Linear regression fit points
        self.linear_regression_fit_points_spinbox = LinearRegressionFitPoints(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.linear_regression_fit_points
        )
        # Aspect ratio threshold
        self.aspect_ratio_threshold_spinbox = AspectRatioThreshold(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.aspect_ratio_threshold
        )
        ## Pattern matching group
        # Minimum number of image region splits
        self.minimum_number_of_image_region_splits_spinbox = MinimumNumberOfImageRegionSplits(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.min_pattern_splits
        )
        # Target image region tile size
        self.target_image_region_tile_size_spinbox = TargetImageRegionTileSize(
            parent=self,
            constraints=self._constraints,
            initial_value=self._initial.target_tile_size
        )
        ## Contrast/Brightness Calibration group
        # Master enable (opt-in)
        self.auto_cb_on_start_checkbox = QCheckBox()
        self.auto_cb_on_start_checkbox.setStyleSheet(AppStyles.CheckBox.settings_dialog())
        self.auto_cb_on_start_checkbox.setChecked(self._initial.auto_cb_on_start)
        # Calibration cadence: first patterning session only (default) vs
        # every session
        self.cb_recalibrate_every_session_checkbox = QCheckBox()
        self.cb_recalibrate_every_session_checkbox.setStyleSheet(AppStyles.CheckBox.settings_dialog())
        self.cb_recalibrate_every_session_checkbox.setChecked(
            self._initial.cb_recalibrate_every_session)
        # Detector full-scale / saturation ceiling
        self.cb_white_level_spinbox = CBWhiteLevel(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_white_level
        )
        # Lower edge margin: gap between black and the histogram's p2 edge
        self.cb_lower_margin_spinbox = CBLowerMargin(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_lower_margin
        )
        # Upper edge margin: gap between the histogram's p98 edge and white
        self.cb_upper_margin_spinbox = CBUpperMargin(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_upper_margin
        )
        # Max white clip fraction
        self.cb_max_white_clip_fraction_spinbox = CBMaxWhiteClipFraction(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_max_white_clip_fraction
        )
        # Max black clip fraction
        self.cb_max_black_clip_fraction_spinbox = CBMaxBlackClipFraction(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_max_black_clip_fraction
        )
        # Lower CB clamp bound
        self.cb_min_bound_spinbox = CBMinBound(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_min_bound
        )
        # Upper CB clamp bound
        self.cb_max_bound_spinbox = CBMaxBound(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_max_bound
        )
        # Max controller iterations
        self.cb_max_iterations_spinbox = CBMaxIterations(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_max_iterations
        )
        # Settle seconds after each CB write
        self.cb_settle_seconds_spinbox = CBSettleSeconds(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_settle_seconds
        )
        # Frames averaged per measurement
        self.cb_frames_per_measurement_spinbox = CBFramesPerMeasurement(
            parent=self, constraints=self._constraints,
            initial_value=self._initial.cb_frames_per_measurement
        )
        ## Buttons
        self.save_button = SaveButton()
        self.save_button.setFixedWidth(AppStyles.Dimensions.SPINBOX_WIDTH)
        self.cancel_button = CancelButton()
        self.cancel_button.setFixedWidth(AppStyles.Dimensions.SPINBOX_WIDTH)
        self.restore_defaults_button = RestoreDefaultsButton()
        self.restore_defaults_button.setFixedWidth(AppStyles.Dimensions.SPINBOX_WIDTH + 60)
    
    def _setup_layout(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(AppStyles.Dimensions.GRID_LAYOUT_VSPACING)
        main_layout.setContentsMargins(
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN
        )
        # Monitoring group layout
        monitoring_group_box = QGroupBox("Monitoring")
        monitoring_group_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        monitoring_group_box.setStyleSheet(AppStyles.GroupBox.settings())
        monitoring_grid_layout = QGridLayout()
        self._add_row_to_grid_layout(monitoring_grid_layout, 0, "Acquisition Delay (s)", self.acquisition_delay_spinbox, tooltip=AppStyles.AppToolTips.ACQUISITION_DELAY_SECONDS_LABEL)
        self._add_row_to_grid_layout(monitoring_grid_layout, 1, "Percent Difference Threshold (%)", self.percent_difference_threshold_spinbox, tooltip=AppStyles.AppToolTips.PERCENT_DIFFERENCE_THRESHOLD_LABEL)
        monitoring_group_box.setLayout(monitoring_grid_layout)
        # Image processing group layout
        image_processing_group_box = QGroupBox("Image Processing")
        image_processing_group_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        image_processing_group_box.setStyleSheet(AppStyles.GroupBox.settings())
        image_processing_grid_layout = QGridLayout()
        self._add_row_to_grid_layout(image_processing_grid_layout, 0, "Gaussian Blur Sigma", self.gaussian_sigma_spinbox, tooltip=AppStyles.AppToolTips.GAUSSIAN_BLUR_SIGMA_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 1, "Apply Dilation", self.apply_dilation_checkbox, tooltip=AppStyles.AppToolTips.APPLY_DILATION_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 2, "Multi-Otsu Threshold Classes", self.threshold_number_of_classes_spinbox, tooltip=AppStyles.AppToolTips.THRESHOLD_NUM_CLASSES_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 3, "Binarization Method", self.binarization_method_combobox, tooltip=AppStyles.AppToolTips.BINARIZATION_METHOD_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 4, "Top-Hat Radius (px)", self.tophat_radius_spinbox, tooltip=AppStyles.AppToolTips.TOPHAT_RADIUS_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 5, "Match on Foreground Map", self.match_on_foreground_checkbox, tooltip=AppStyles.AppToolTips.MATCH_ON_FOREGROUND_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 6, "Foreground Completion Mode", self.foreground_completion_mode_combobox, tooltip=AppStyles.AppToolTips.FOREGROUND_COMPLETION_MODE_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 7, "Stall Window (rounds)", self.stall_window_spinbox, tooltip=AppStyles.AppToolTips.STALL_WINDOW_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 8, "Stall Drop Fraction", self.stall_drop_fraction_spinbox, tooltip=AppStyles.AppToolTips.STALL_DROP_FRACTION_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 9, "Stall Relative Tolerance", self.stall_rel_tolerance_spinbox, tooltip=AppStyles.AppToolTips.STALL_REL_TOLERANCE_LABEL)
        self._add_row_to_grid_layout(image_processing_grid_layout, 10, "Stall Absolute Tolerance", self.stall_abs_tolerance_spinbox, tooltip=AppStyles.AppToolTips.STALL_ABS_TOLERANCE_LABEL)
        image_processing_group_box.setLayout(image_processing_grid_layout)
        # Image analysis group layout
        image_analysis_group_box = QGroupBox("Image Analysis")
        image_analysis_group_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        image_analysis_group_box.setStyleSheet(AppStyles.GroupBox.settings())
        image_analysis_grid_layout = QGridLayout()
        self._add_row_to_grid_layout(image_analysis_grid_layout, 0, "Slope Method", self.slope_method_combobox, tooltip=AppStyles.AppToolTips.SLOPE_METHOD_LABEL)
        self._add_row_to_grid_layout(image_analysis_grid_layout, 1, "Slope Points (Gradient)", self.number_of_points_for_gradient_slope_spinbox, tooltip=AppStyles.AppToolTips.NUM_POINTS_FOR_SLOPE_LABEL)
        self._add_row_to_grid_layout(image_analysis_grid_layout, 2, "Slope Points (Linear Regression)", self.linear_regression_fit_points_spinbox, tooltip=AppStyles.AppToolTips.LINEAR_REGRESSION_FIT_POINTS_LABEL)
        self._add_row_to_grid_layout(image_analysis_grid_layout, 3, "Aspect Ratio Threshold", self.aspect_ratio_threshold_spinbox, tooltip=AppStyles.AppToolTips.ASPECT_RATIO_THRESHOLD_LABEL)
        image_analysis_group_box.setLayout(image_analysis_grid_layout)
        # Pattern matching group layout
        pattern_matching_group_box = QGroupBox("Pattern Matching")
        pattern_matching_group_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        pattern_matching_group_box.setStyleSheet(AppStyles.GroupBox.settings())
        pattern_matching_grid_layout = QGridLayout()
        self._add_row_to_grid_layout(pattern_matching_grid_layout, 0, "Minimum Number of RTM Sub-Regions", self.minimum_number_of_image_region_splits_spinbox, tooltip=AppStyles.AppToolTips.MIN_PATTERN_SPLITS_LABEL)
        self._add_row_to_grid_layout(pattern_matching_grid_layout, 1, "Target Region Tile Size (px)", self.target_image_region_tile_size_spinbox, tooltip=AppStyles.AppToolTips.TARGET_TILE_SIZE_LABEL)
        pattern_matching_group_box.setLayout(pattern_matching_grid_layout)
        # Contrast/Brightness calibration group layout
        cb_group_box = QGroupBox("Contrast/Brightness Calibration")
        cb_group_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        cb_group_box.setStyleSheet(AppStyles.GroupBox.settings())
        cb_grid_layout = QGridLayout()
        self._add_row_to_grid_layout(cb_grid_layout, 0, "Auto-Calibrate on Start", self.auto_cb_on_start_checkbox, tooltip=AppStyles.AppToolTips.CB_AUTO_ON_START_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 1, "Recalibrate Every Session", self.cb_recalibrate_every_session_checkbox, tooltip=AppStyles.AppToolTips.CB_RECALIBRATE_EVERY_SESSION_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 2, "Detector White Level (fallback)", self.cb_white_level_spinbox, tooltip=AppStyles.AppToolTips.CB_WHITE_LEVEL_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 3, "Lower Margin (fraction)", self.cb_lower_margin_spinbox, tooltip=AppStyles.AppToolTips.CB_LOWER_MARGIN_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 4, "Upper Margin (fraction)", self.cb_upper_margin_spinbox, tooltip=AppStyles.AppToolTips.CB_UPPER_MARGIN_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 5, "Max White Clip (fraction)", self.cb_max_white_clip_fraction_spinbox, tooltip=AppStyles.AppToolTips.CB_MAX_WHITE_CLIP_FRACTION_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 6, "Max Black Clip (fraction)", self.cb_max_black_clip_fraction_spinbox, tooltip=AppStyles.AppToolTips.CB_MAX_BLACK_CLIP_FRACTION_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 7, "C/B Lower Bound", self.cb_min_bound_spinbox, tooltip=AppStyles.AppToolTips.CB_MIN_BOUND_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 8, "C/B Upper Bound", self.cb_max_bound_spinbox, tooltip=AppStyles.AppToolTips.CB_MAX_BOUND_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 9, "Max Iterations", self.cb_max_iterations_spinbox, tooltip=AppStyles.AppToolTips.CB_MAX_ITERATIONS_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 10, "Settle Time (s)", self.cb_settle_seconds_spinbox, tooltip=AppStyles.AppToolTips.CB_SETTLE_SECONDS_LABEL)
        self._add_row_to_grid_layout(cb_grid_layout, 11, "Verify Frames", self.cb_frames_per_measurement_spinbox, tooltip=AppStyles.AppToolTips.CB_FRAMES_PER_MEASUREMENT_LABEL)
        cb_group_box.setLayout(cb_grid_layout)
        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.restore_defaults_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.cancel_button)
        # Put all setting groups in a scrollable content widget so the dialog
        # never outgrows the screen as more settings are added.
        content_layout = QVBoxLayout()
        content_layout.setSpacing(AppStyles.Dimensions.GRID_LAYOUT_VSPACING)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(monitoring_group_box)
        content_layout.addWidget(image_processing_group_box)
        content_layout.addWidget(image_analysis_group_box)
        content_layout.addWidget(pattern_matching_group_box)
        content_layout.addWidget(cb_group_box)
        # Kept for _apply_monitoring_lock: everything except the CB group stays
        # locked while monitoring runs (see LIVE_EDITABLE_FIELDS for why).
        self._locked_while_monitoring = (
            monitoring_group_box, image_processing_group_box,
            image_analysis_group_box, pattern_matching_group_box,
        )
        # Kept because the lock is structural (group boxes) while the live push
        # is by name (LIVE_EDITABLE_FIELDS): the two agree only as long as this
        # group holds exactly the live-editable widgets, and the test suite
        # pins that through this handle.
        self._cb_group_box = cb_group_box
        content_layout.addStretch()

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_widget.setLayout(content_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(AppStyles.ScrollArea.default())
        scroll_area.setWidget(content_widget)

        # Scrollable content (expands) + fixed button row at the bottom.
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)
        self.ensurePolished()  # apply stylesheet fonts so size hints are accurate

        # Width: fit the content's natural width plus the scrollbar gutter so the
        # group boxes are never clipped on the right by the scrollbar.
        scrollbar_width = scroll_area.verticalScrollBar().sizeHint().width()
        content_width = content_widget.sizeHint().width()
        dialog_width = max(
            AppStyles.Dimensions.SETTINGS_DIALOG_WIDTH,
            content_width + scrollbar_width
            + 2 * AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN + 6
        )

        # Cap the dialog height to the available screen so content scrolls rather
        # than overflowing; pick a comfortable initial height within that cap.
        screen = self.screen() or QApplication.primaryScreen()
        max_height = int(screen.availableGeometry().height() * 0.9) if screen else 800
        self.setMinimumWidth(dialog_width)
        self.setMaximumHeight(max_height)
        self.resize(dialog_width, min(max_height, 760))

    def _add_row_to_grid_layout(self, grid: QGridLayout, row: int, label_text: str,
                                widget, tooltip: str = ""):
        """Add a label and widget row to a grid layout with consistent styling."""
        label = QLabel(label_text)
        label.setStyleSheet(AppStyles.Label.settings())
        if tooltip:
            label.setToolTip(tooltip)
            # The input widget carries the same tooltip: the operator hovers
            # the control at least as often as its label, and until 3.4.2
            # that hover showed nothing.
            widget.setToolTip(tooltip)
        grid.setHorizontalSpacing(AppStyles.Dimensions.GRID_LAYOUT_HSPACING)
        grid.setVerticalSpacing(AppStyles.Dimensions.GRID_LAYOUT_VSPACING)
        grid.setContentsMargins(
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN + 10,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN
        )
        grid.addWidget(label, row, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(widget, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
    
    def _create_enum_combobox(self, enum_class, display_map: dict) -> QComboBox:
        """
        Create a QComboBox for an Enum type.
        
        :param enum_class: The Enum class
        :param display_map: Dict mapping enum members to display strings
        :return: A QComboBox populated with the enum members
        """
        combo_box = QComboBox()
        for member, text in display_map.items():
            combo_box.addItem(text, userData=member)
        combo_box.setFixedWidth(AppStyles.Dimensions.COMBOBOX_WIDTH)
        combo_box.setStyleSheet(AppStyles.ComboBox.default())

        return combo_box

    @staticmethod
    def _get_enum_value(combo_box: QComboBox):
        """Get the enum member stored as userData in the current selection."""
        return combo_box.currentData()
    
    def _setup_connections(self):
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.restore_defaults_button.clicked.connect(self._on_restore_defaults)
        # Stall settings apply only to the Absolute + stall latch mode
        self.foreground_completion_mode_combobox.currentIndexChanged.connect(
            self._update_stall_settings_enabled
        )
        self._update_stall_settings_enabled()

    def _update_stall_settings_enabled(self):
        """Enable the stall spinboxes only when the stall latch mode is selected."""
        stall_mode = (
            self._get_enum_value(self.foreground_completion_mode_combobox)
            == ForegroundCompletionMode.ABSOLUTE_PLUS_STALL
        )
        for spinbox in (
            self.stall_window_spinbox,
            self.stall_drop_fraction_spinbox,
            self.stall_rel_tolerance_spinbox,
            self.stall_abs_tolerance_spinbox,
        ):
            spinbox.setEnabled(stall_mode)

    def _on_restore_defaults(self):
        # Reset all fields to defaults
        self.acquisition_delay_spinbox.setValue(self._defaults.acquisition_delay_seconds)
        self.percent_difference_threshold_spinbox.setValue(self._defaults.percent_difference_threshold)
        self.gaussian_sigma_spinbox.setValue(self._defaults.gaussian_sigma)
        self.apply_dilation_checkbox.setChecked(self._defaults.apply_dilation)
        self.threshold_number_of_classes_spinbox.setValue(self._defaults.threshold_num_classes)
        self.tophat_radius_spinbox.setValue(self._defaults.tophat_radius)
        self.match_on_foreground_checkbox.setChecked(self._defaults.match_on_foreground)
        self.foreground_completion_mode_combobox.setCurrentIndex(self.foreground_completion_mode_combobox.findData(self._defaults.foreground_completion_mode))
        self.stall_window_spinbox.setValue(self._defaults.stall_window)
        self.stall_drop_fraction_spinbox.setValue(self._defaults.stall_drop_fraction)
        self.stall_rel_tolerance_spinbox.setValue(self._defaults.stall_rel_tolerance)
        self.stall_abs_tolerance_spinbox.setValue(self._defaults.stall_abs_tolerance)
        # setCurrentIndex only fires currentIndexChanged on an actual change,
        # so refresh the stall-row enable state explicitly.
        self._update_stall_settings_enabled()
        self.binarization_method_combobox.setCurrentIndex(self.binarization_method_combobox.findData(self._defaults.binarization_method))
        self.slope_method_combobox.setCurrentIndex(self.slope_method_combobox.findData(self._defaults.slope_method))
        self.number_of_points_for_gradient_slope_spinbox.setValue(self._defaults.num_points_for_slope)
        self.linear_regression_fit_points_spinbox.setValue(self._defaults.linear_regression_fit_points)
        self.aspect_ratio_threshold_spinbox.setValue(self._defaults.aspect_ratio_threshold)
        self.minimum_number_of_image_region_splits_spinbox.setValue(self._defaults.min_pattern_splits)
        self.target_image_region_tile_size_spinbox.setValue(self._defaults.target_tile_size)
        # Contrast/Brightness calibration
        self.auto_cb_on_start_checkbox.setChecked(self._defaults.auto_cb_on_start)
        self.cb_recalibrate_every_session_checkbox.setChecked(
            self._defaults.cb_recalibrate_every_session)
        self.cb_white_level_spinbox.setValue(self._defaults.cb_white_level)
        self.cb_lower_margin_spinbox.setValue(self._defaults.cb_lower_margin)
        self.cb_upper_margin_spinbox.setValue(self._defaults.cb_upper_margin)
        self.cb_max_white_clip_fraction_spinbox.setValue(self._defaults.cb_max_white_clip_fraction)
        self.cb_max_black_clip_fraction_spinbox.setValue(self._defaults.cb_max_black_clip_fraction)
        self.cb_min_bound_spinbox.setValue(self._defaults.cb_min_bound)
        self.cb_max_bound_spinbox.setValue(self._defaults.cb_max_bound)
        self.cb_max_iterations_spinbox.setValue(self._defaults.cb_max_iterations)
        self.cb_settle_seconds_spinbox.setValue(self._defaults.cb_settle_seconds)
        self.cb_frames_per_measurement_spinbox.setValue(self._defaults.cb_frames_per_measurement)
        logger.info("Settings dialog: restored defaults")
    
    def _apply_monitoring_lock(self):
        """
        Disable everything that cannot safely change mid-run.

        No tooltip: a control grayed out during a run reads as exactly that.

        Restore Defaults is disabled too, and that is not cosmetic. It writes
        every field at once, including the locked ones. Those values would be
        saved and would replace the main window's ProcessingParameters while the
        running worker kept its own copy -- so the worker and the saved settings
        would silently disagree, and the next Start would come up on defaults the
        operator never chose. (The on-tool settings are deliberately not the
        shipped defaults: span 0.70 against a 0.55 default, among others.)
        """
        if not self._monitoring_active:
            return
        for group_box in self._locked_while_monitoring:
            group_box.setEnabled(False)
        self.restore_defaults_button.setEnabled(False)

    def live_editable_changes(self, updated: ProcessingParameters) -> dict:
        """
        The LIVE_EDITABLE_FIELDS whose values differ from the ones this dialog
        opened with.

        Diffed on accept rather than tracked per keystroke on purpose: a spin box
        emits valueChanged on every step, so wiring these to the signal would
        queue an update per scroll tick of a drag.

        :param updated: the ProcessingParameters collected from the dialog
        :return: {field_name: new_value} -- empty when nothing changed
        """
        changed = {}
        for name in self.LIVE_EDITABLE_FIELDS:
            new_value = getattr(updated, name)
            if new_value != getattr(self._initial, name):
                changed[name] = new_value
        return changed

    def get_parameters(self) -> ProcessingParameters:
        """Collect current settings from the dialog and return as a ProcessingParameters object."""
        params = ProcessingParameters(
            crop_rect_pattern_1=self._initial.crop_rect_pattern_1,  # Not editable in dialog, so take from initial
            crop_rect_pattern_2=self._initial.crop_rect_pattern_2,  # Not editable in dialog, so take from initial
            acquisition_delay_seconds=self.acquisition_delay_spinbox.value(),
            percent_difference_threshold=self.percent_difference_threshold_spinbox.value(),
            gaussian_sigma=self.gaussian_sigma_spinbox.value(),
            apply_dilation=self.apply_dilation_checkbox.isChecked(),
            threshold_num_classes=int(self.threshold_number_of_classes_spinbox.value()),
            tophat_radius=int(self.tophat_radius_spinbox.value()),
            match_on_foreground=self.match_on_foreground_checkbox.isChecked(),
            foreground_completion_mode=self._get_enum_value(self.foreground_completion_mode_combobox),
            stall_window=int(self.stall_window_spinbox.value()),
            stall_drop_fraction=self.stall_drop_fraction_spinbox.value(),
            stall_rel_tolerance=self.stall_rel_tolerance_spinbox.value(),
            stall_abs_tolerance=self.stall_abs_tolerance_spinbox.value(),
            binarization_method=self._get_enum_value(self.binarization_method_combobox),
            slope_method=self._get_enum_value(self.slope_method_combobox),
            num_points_for_slope=int(self.number_of_points_for_gradient_slope_spinbox.value()),
            linear_regression_fit_points=int(self.linear_regression_fit_points_spinbox.value()),
            aspect_ratio_threshold=self.aspect_ratio_threshold_spinbox.value(),
            min_pattern_splits=int(self.minimum_number_of_image_region_splits_spinbox.value()),
            target_tile_size=int(self.target_image_region_tile_size_spinbox.value()),
            auto_cb_on_start=self.auto_cb_on_start_checkbox.isChecked(),
            cb_recalibrate_every_session=(
                self.cb_recalibrate_every_session_checkbox.isChecked()),
            cb_white_level=self.cb_white_level_spinbox.value(),
            cb_lower_margin=self.cb_lower_margin_spinbox.value(),
            cb_upper_margin=self.cb_upper_margin_spinbox.value(),
            cb_max_white_clip_fraction=self.cb_max_white_clip_fraction_spinbox.value(),
            cb_max_black_clip_fraction=self.cb_max_black_clip_fraction_spinbox.value(),
            cb_min_bound=self.cb_min_bound_spinbox.value(),
            cb_max_bound=self.cb_max_bound_spinbox.value(),
            cb_max_iterations=int(self.cb_max_iterations_spinbox.value()),
            cb_settle_seconds=self.cb_settle_seconds_spinbox.value(),
            cb_frames_per_measurement=int(self.cb_frames_per_measurement_spinbox.value())
        )
        if self._monitoring_active:
            # A locked widget can still change the value it reports: a spin box
            # rounds to its declared decimals, so a QSettings binary round-trip
            # like aspect_ratio_threshold = 0.44999999999999996 comes back as
            # 0.45. Accepting the dialog mid-run would then silently rewrite a
            # locked field while the running worker keeps the original -- the
            # exact worker/settings divergence the monitoring lock exists to
            # prevent. Locked fields pass through exactly as the dialog opened
            # with them.
            for dc_field in fields(params):
                if dc_field.name not in self.LIVE_EDITABLE_FIELDS:
                    setattr(params, dc_field.name,
                            getattr(self._initial, dc_field.name))
        logger.info("Settings dialog: collected parameters from dialog")
        return params