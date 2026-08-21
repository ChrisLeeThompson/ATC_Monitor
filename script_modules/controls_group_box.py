"""
Module for managing the controls group box.
"""
import logging
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QCheckBox, QRadioButton, QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from script_modules.app_styles import AppStyles
from script_modules.spinbox_widgets import (
    NumberOfImagesSpinBox, NumberOfSecondsSpinBox, MeanPixelSlopeThreshold, 
    MatchScoreThreshold, MaximumPixelsThreshold, ConfirmationRounds
)
from script_modules.atc_monitor_parameters import SpinBoxConstraints, UIParameters, WindowMode
from script_modules.button_widgets import SettingsButton


logger = logging.getLogger(__name__)


class ControlsGroupBox(QGroupBox):

    # Signals
    control_parameters_changed = Signal(dict)   # dictionary of all parameters in the controls group box
    match_score_threshold_value_changed = Signal(float)
    window_mode_changed = Signal(object)  # Emitted when monitoring mode radio button changes (WindowMode enum)
    settings_requested = Signal()  # Emitted when settings button is clicked
    
    def __init__(self, parent=None, constraints: SpinBoxConstraints | None = None, 
                 initial_values: UIParameters | None = None):
        super().__init__(parent)

        # Store contraints and initial values
        self.constraints = constraints if constraints else SpinBoxConstraints()
        self.initial_values = initial_values if initial_values else UIParameters()

        # Set style
        self.setStyleSheet(AppStyles.GroupBox.default())
        # Set size policy
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred
        )
        # Set focus policy
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Create components
        self._create_components()
        # Setup connections
        self._setup_connections()
        # Set layout
        layout = self._setup_layout()
        self.setLayout(layout)
        
    def _create_components(self):
        """Create components in the group box and set their styles."""
        # Store window mode
        self._window_mode = self.initial_values.window_mode
        is_manual_mode = self._window_mode == WindowMode.MANUAL

        # Monitoring mode radio buttons
        self.image_count_radio = QRadioButton("Image Count")
        self.time_interval_radio = QRadioButton("Time Interval")
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.addButton(self.image_count_radio)
        self.mode_button_group.addButton(self.time_interval_radio)
        # Set initial selection
        if is_manual_mode:
            self.image_count_radio.setChecked(True)
        else:
            self.time_interval_radio.setChecked(True)

        # Number of images (manual mode)
        self.number_of_images_label = QLabel("Number of Images")
        self.number_of_images_spinbox = NumberOfImagesSpinBox(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.number_of_images
        )
        # Analysis interval (Time based mode)
        self.analysis_interval_seconds_label = QLabel("Analysis Interval (s)")
        self.analysis_interval_seconds_spinbox = NumberOfSecondsSpinBox(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.analysis_interval_seconds
        )
        # Show the active mode's row
        self.number_of_images_label.setVisible(is_manual_mode)
        self.number_of_images_spinbox.setVisible(is_manual_mode)
        self.analysis_interval_seconds_label.setVisible(not is_manual_mode)
        self.analysis_interval_seconds_spinbox.setVisible(not is_manual_mode)

        # Mean pixel slope threshold
        self.mean_pixel_slope_threshold_label = QLabel("Mean Pixel Slope Threshold")
        self.mean_pixel_slope_threshold_spinbox = MeanPixelSlopeThreshold(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.mean_pixel_slope_threshold
        )
        # Match score threshold
        self.match_score_threshold_label = QLabel("Match Score Threshold")
        self.match_score_threshold_spinbox = MatchScoreThreshold(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.match_score_threshold
        )
        # Maximum pixels threshold (label text is mode-aware: the main window
        # relabels it to "Maximum Foreground Energy" when the Top-Hat method is active,
        # since that metric is a continuous energy, not a white-pixel percentage)
        self.maximum_pixels_threshold_label = QLabel("Maximum Pixels Threshold")
        self.maximum_pixels_threshold_spinbox = MaximumPixelsThreshold(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.maximum_pixels_threshold
        )
        # Confirmation rounds
        self.confirmation_rounds_label = QLabel("Confirmation Rounds")
        self.confirmation_rounds_spinbox = ConfirmationRounds(
            parent=self,
            constraints=self.constraints,
            initial_value=self.initial_values.confirmation_rounds
        )
        # Spacer label
        self.spacer_label = QLabel(" ")
        # Checkboxes
        self.show_grayscale_images_checkbox = QCheckBox("Show Grayscale Images")
        self.show_grayscale_images_checkbox.setChecked(self.initial_values.show_grayscale_images)
        self.save_data_checkbox = QCheckBox("Save Data")
        self.save_data_checkbox.setChecked(self.initial_values.save_data)
        # Settings button
        self.settings_button = SettingsButton()
        self.settings_button.setStyleSheet(AppStyles.Button.default())
        self.settings_button.setFixedWidth(AppStyles.Dimensions.SPINBOX_WIDTH)
        # Set radio button styles
        for radio_button in [self.image_count_radio, self.time_interval_radio]:
             radio_button.setStyleSheet(AppStyles.RadioButton.default())
        # Set label styles
        for label in [
            self.number_of_images_label, self.analysis_interval_seconds_label, 
            self.mean_pixel_slope_threshold_label, self.match_score_threshold_label,
            self.maximum_pixels_threshold_label, self.confirmation_rounds_label
        ]:
             label.setStyleSheet(AppStyles.Label.default())
        # Set checkbox styles
        for checkbox in [self.show_grayscale_images_checkbox, self.save_data_checkbox]:
             checkbox.setStyleSheet(AppStyles.CheckBox.default())
        # Set tooltips
        self.image_count_radio.setToolTip(AppStyles.AppToolTips.IMAGE_COUNT_RADIOBUTTON)
        self.time_interval_radio.setToolTip(AppStyles.AppToolTips.TIME_INTERVAL_RADIOBUTTON)
        self.number_of_images_label.setToolTip(AppStyles.AppToolTips.NUMBER_OF_IMAGES_LABEL)
        self.analysis_interval_seconds_label.setToolTip(AppStyles.AppToolTips.ANALYSIS_INTERVAL_LABEL)
        self.mean_pixel_slope_threshold_label.setToolTip(AppStyles.AppToolTips.MEAN_PIXEL_SLOPE_THRESHOLD_LABEL)
        self.match_score_threshold_label.setToolTip(AppStyles.AppToolTips.MATCH_SCORE_THRESHOLD_LABEL)
        self.maximum_pixels_threshold_label.setToolTip(AppStyles.AppToolTips.MAXIMUM_PIXELS_THRESHOLD_LABEL)
        self.confirmation_rounds_label.setToolTip(AppStyles.AppToolTips.CONFIRMATION_ROUNDS_LABEL)
        self.show_grayscale_images_checkbox.setToolTip(AppStyles.AppToolTips.SHOW_GRAYSCALE_IMAGES_CHECKBOX)
        self.save_data_checkbox.setToolTip(AppStyles.AppToolTips.SAVE_DATA_CHECKBOX)
    
    def _setup_connections(self):
        """Set up connections for the components in the group box."""
        # Monitoring mode radio buttons
        self.image_count_radio.toggled.connect(self._on_mode_changed)
        # Connect spinboxes to centralized handler
        self.number_of_images_spinbox.valueChanged.connect(self._on_parameters_changed)
        self.analysis_interval_seconds_spinbox.valueChanged.connect(self._on_analysis_interval_changed)
        self.mean_pixel_slope_threshold_spinbox.valueChanged.connect(self._on_parameters_changed)
        self.maximum_pixels_threshold_spinbox.valueChanged.connect(self._on_parameters_changed)
        self.confirmation_rounds_spinbox.valueChanged.connect(self._on_parameters_changed)
        # Checkboxes are display-only settings - do not route through parameter pipeline
        self.show_grayscale_images_checkbox.stateChanged.connect(self._on_display_option_changed)
        self.save_data_checkbox.stateChanged.connect(self._on_display_option_changed)
        # Special case: match score threshold needs dedicated signal for real-time plot sync
        self.match_score_threshold_spinbox.valueChanged.connect(self._on_match_score_threshold_changed)
        # Settings button
        self.settings_button.clicked.connect(self.settings_requested.emit)
    
    def _setup_layout(self) -> QVBoxLayout:
        """Set up the group box layout."""
        # Main layout in the group box
        main_layout = QVBoxLayout()

        # Radio button row for monitoring mode
        mode_layout = QHBoxLayout()
        mode_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_layout.setSpacing(40)
        mode_layout.setContentsMargins(
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN + 4
        )
        mode_layout.addWidget(self.image_count_radio)
        mode_layout.addWidget(self.time_interval_radio)
        
        main_layout.addLayout(mode_layout)

        # Grid layout for components in the grid box
        grid_layout = QGridLayout()
        # Set styling of the grid layout
        grid_layout.setContentsMargins(
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN,
            AppStyles.Dimensions.GRID_LAYOUT_CONTENTS_MARGIN
        )
        grid_layout.setHorizontalSpacing(AppStyles.Dimensions.GRID_LAYOUT_HSPACING)
        grid_layout.setVerticalSpacing(AppStyles.Dimensions.GRID_LAYOUT_VSPACING)
        # Add widgets to grid layout
        grid_layout.addWidget(self.number_of_images_label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.number_of_images_spinbox, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.analysis_interval_seconds_label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.analysis_interval_seconds_spinbox, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.mean_pixel_slope_threshold_label, 2, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.mean_pixel_slope_threshold_spinbox, 2, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.match_score_threshold_label, 3, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.match_score_threshold_spinbox, 3, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.maximum_pixels_threshold_label, 4, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.maximum_pixels_threshold_spinbox, 4, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.confirmation_rounds_label, 5, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.confirmation_rounds_spinbox, 5, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.spacer_label, 6, 0)
        grid_layout.addWidget(self.show_grayscale_images_checkbox, 7, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.settings_button, 7, 1, 2, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.save_data_checkbox, 8, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        # Add grid layout to main layout
        main_layout.addLayout(grid_layout)

        return main_layout

    def _on_parameters_changed(self):
        """
        Centralized handler for any parameter change.
        Emits the control_parameters_changed signal with current values.
        """
        params = self.get_control_parameters()
        # DEBUG: same dict is logged one signal-hop later by MainWindow as
        # "Updated global parameters" -- keep only one INFO record per change.
        logger.debug(f"Control parameters changed: {params}")
        self.control_parameters_changed.emit(params)
    
    def get_control_parameters(self) -> dict:
        """Get the current parameters from the controls as a dictionary."""
        return {
            "window_mode": self._window_mode,
            "number_of_images": int(self.number_of_images_spinbox.value()),
            "analysis_interval_seconds": int(self.analysis_interval_seconds_spinbox.value()),
            "mean_pixel_slope_threshold": self.mean_pixel_slope_threshold_spinbox.value(),
            "match_score_threshold": self.match_score_threshold_spinbox.value(),
            "maximum_pixels_threshold": self.maximum_pixels_threshold_spinbox.value(),
            "confirmation_rounds": int(self.confirmation_rounds_spinbox.value()),
            "show_grayscale_images": self.show_grayscale_images_checkbox.isChecked(),
            "save_data": self.save_data_checkbox.isChecked()
        }

    def _on_analysis_interval_changed(self):
        """Handle analysis interval changes - reset auto-adjusted style and emit."""
        self.analysis_interval_seconds_spinbox.setStyleSheet(AppStyles.SpinBox.default())
        self.analysis_interval_seconds_spinbox.setToolTip("")
        self._on_parameters_changed()
    
    @Slot(int, bool)
    def set_auto_adjusted_interval(self, adjusted_seconds: int, was_clamped: bool):
        """
        Update the analysis interval spinbox from worker calibration.
        
        :param adjusted_seconds: The actual analysis interval after calibration
        :param was_clamped: Whether the value was clamped due to rate constraints
        """
        # Block signals to prevent triggering _on_analysis_interval_changed
        self.analysis_interval_seconds_spinbox.blockSignals(True)
        self.analysis_interval_seconds_spinbox.setValue(adjusted_seconds)
        self.analysis_interval_seconds_spinbox.blockSignals(False)
        
        # Apply visual indicator if value was auto-adjusted
        if was_clamped:
            self.analysis_interval_seconds_spinbox.setStyleSheet(AppStyles.SpinBox.auto_adjusted())
            self.analysis_interval_seconds_spinbox.setToolTip(
                f"Adjusted to {adjusted_seconds}s (limited by the RTM image rate)."
            )
        else:
            self.analysis_interval_seconds_spinbox.setStyleSheet(AppStyles.SpinBox.default())
            self.analysis_interval_seconds_spinbox.setToolTip("")
    
    @Slot(float)
    def _on_match_score_threshold_changed(self, value: float):
        """Handle match score threshold changes - emits both signals."""
        self.match_score_threshold_value_changed.emit(value)  # For real-time plot sync
        self._on_parameters_changed()  # Also update global params
    
    def _on_mode_changed(self, checked: bool):
        """
        Handle monitoring mode radio button change.
        
        Connected only to image_count_radio.toggled, so this fires once per
        switch: True when Image Count selected, False when Time Interval selected.
        """
        is_manual = self.image_count_radio.isChecked()
        self._window_mode = WindowMode.MANUAL if is_manual else WindowMode.TIME_BASED
        
        # Toggle visibility of the appropriate label/spinbox row
        self.number_of_images_label.setVisible(is_manual)
        self.number_of_images_spinbox.setVisible(is_manual)
        self.analysis_interval_seconds_label.setVisible(not is_manual)
        self.analysis_interval_seconds_spinbox.setVisible(not is_manual)
        
        logger.info(f"Monitoring mode changed to: {self._window_mode.name}")
        self.window_mode_changed.emit(self._window_mode)
    
    def _on_display_option_changed(self):
        """
        Handle display-only option changes (checkboxes).
        
        These settings do not affect monitoring criteria or the worker parameter
        pipeline. They are stored in the central parameters for UI state only.
        """
        logger.info(
            f"Display option changed: "
            f"show_grayscale={self.show_grayscale_images_checkbox.isChecked()}, "
            f"save_data={self.save_data_checkbox.isChecked()}"
        )
    
    def set_monitoring_mode_enabled(self, enabled: bool):
        """
        Enable or disable the monitoring mode radio buttons.
        
        Called when monitoring starts (disable) or stops (enable) to prevent
        mode changes during active processing.
        
        :param enabled: True to enable, False to disable
        """
        self.image_count_radio.setEnabled(enabled)
        self.time_interval_radio.setEnabled(enabled)
    
    def set_settings_button_enabled(self, enabled: bool):
        """
        Enable or disable the Settings button.
        
        Disabled during monitoring because processing parameters
        are locked once the worker is running.
        
        :param enabled: True to enable, False to disable
        """
        self.settings_button.setEnabled(enabled)

    def set_pixels_threshold_label(self, text: str):
        """
        Set the maximum-pixels threshold label text.

        The main window calls this to keep the label honest per binarization
        method: "Maximum Pixels Threshold" for the white-pixel methods,
        "Maximum Foreground Energy" for the Top-Hat continuous-energy method.

        :param text: Label text to display
        """
        self.maximum_pixels_threshold_label.setText(text)