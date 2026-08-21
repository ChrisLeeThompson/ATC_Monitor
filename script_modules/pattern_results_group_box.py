"""
Module for managing the pattern results group box.

Displays evaluation results for both patterns in a single group box.
Checkboxes control which criteria are enabled for completion evaluation.
"""
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox,
    QGridLayout, QSizePolicy, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from script_modules.app_styles import AppStyles


logger = logging.getLogger(__name__)

# Largest content each grid cell can ever show; reserving its size up front
# keeps the window from resizing mid-run (the "(stall)" annotation on a
# frozen results panel used to widen the whole window permanently, because
# the grid's minimum size tracked its largest-ever content). "(stall)"
# renders on a second line, so the reservation is two lines tall and only
# one value wide.
_LARGEST_RESULT_TEXT = "100.00\n(stall)"      # percent-pixels row, stall-latch completion
_WIDEST_CRITERION_TEXT = "Foreground Energy"  # criterion checkbox flips with binarization method


class PatternResultsGroupBox(QGroupBox):

    # Signal emitted when any checkbox state changes
    # Dict: {'mean_slope': bool, 'match_score': bool, 'percent_pixels': bool}
    criteria_enabled_changed = Signal(dict)

    def __init__(self, parent=None, initial_confirmation_rounds=3):
        super().__init__(parent)
        # Set title
        self.setTitle("Pattern Results")
        # Set style
        self.setStyleSheet(AppStyles.GroupBox.with_title())
        # Set focus policy
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Create components
        self._create_components()
        # Setup connections
        self._setup_connections()
        # Initialize confirmation rounds display
        self.set_confirmation_rounds_total(initial_confirmation_rounds)
        # Set layout
        layout = self._setup_layout()
        self.setLayout(layout)
    
    def _create_components(self):
        """Create components in the group box."""
        # Checkboxes (criteria enable/disable)
        self.mean_slope_checkbox = QCheckBox("Mean Slope")
        self.match_score_checkbox = QCheckBox("Match Score")
        self.percent_pixels_checkbox = QCheckBox("Percent Pixels")
        # All checked by default
        self.mean_slope_checkbox.setChecked(True)
        self.match_score_checkbox.setChecked(True)
        self.percent_pixels_checkbox.setChecked(True)
        # Labels
        self.pattern1_label = QLabel("Pattern 1")
        self.pattern2_label = QLabel("Pattern 2")
        self.confirmation_rounds_label = QLabel("Confirmation Rounds")
        self.pattern_duration_label = QLabel("Duration")
        # Pattern 1 result labels
        self.p1_mean_slope_result_label = QLabel("")
        self.p1_match_score_result_label = QLabel("")
        self.p1_percent_pixels_result_label = QLabel("")
        self.p1_confirmation_rounds_result_label = QLabel("")
        self.p1_pattern_duration_result_label = QLabel("")
        # Pattern 2 result labels
        self.p2_mean_slope_result_label = QLabel("")
        self.p2_match_score_result_label = QLabel("")
        self.p2_percent_pixels_result_label = QLabel("")
        self.p2_confirmation_rounds_result_label = QLabel("")
        self.p2_pattern_duration_result_label = QLabel("")

        # Map pattern index to label sets for routing
        self._result_labels = {
            0: {
                'mean_slope': self.p1_mean_slope_result_label,
                'match_score': self.p1_match_score_result_label,
                'percent_pixels': self.p1_percent_pixels_result_label,
                'confirmation_rounds': self.p1_confirmation_rounds_result_label,
                'duration': self.p1_pattern_duration_result_label,
            },
            1: {
                'mean_slope': self.p2_mean_slope_result_label,
                'match_score': self.p2_match_score_result_label,
                'percent_pixels': self.p2_percent_pixels_result_label,
                'confirmation_rounds': self.p2_confirmation_rounds_result_label,
                'duration': self.p2_pattern_duration_result_label,
            }
        }

        # Set checkbox styles
        for checkbox in [self.mean_slope_checkbox, self.match_score_checkbox, 
                         self.percent_pixels_checkbox]:
            checkbox.setStyleSheet(AppStyles.CheckBox.default())
        # Set label styles
        for label in [self.confirmation_rounds_label, self.pattern_duration_label,
                      self.pattern1_label, self.pattern2_label]:
            label.setStyleSheet(AppStyles.Label.default())
        # Set pattern 1 result label styles
        for label in [self.p1_mean_slope_result_label, self.p1_match_score_result_label,
                      self.p1_percent_pixels_result_label,
                      self.p1_confirmation_rounds_result_label, self.p1_pattern_duration_result_label]:
            label.setStyleSheet(AppStyles.Label.result_default())
        # Set pattern 2 result label styles
        for label in [self.p2_mean_slope_result_label, self.p2_match_score_result_label,
                      self.p2_percent_pixels_result_label,
                      self.p2_confirmation_rounds_result_label, self.p2_pattern_duration_result_label]:
            label.setStyleSheet(AppStyles.Label.result_default())
        # Set tooltips
        self.mean_slope_checkbox.setToolTip(AppStyles.AppToolTips.MEAN_SLOPE_CHECKBOX)
        self.match_score_checkbox.setToolTip(AppStyles.AppToolTips.MATCH_SCORE_CHECKBOX)
        self.percent_pixels_checkbox.setToolTip(AppStyles.AppToolTips.PERCENT_PIXELS_CHECKBOX)
        # The percent-pixels values carry the "(stall)" completion annotation;
        # the tooltip explains it (the other result values have none to explain).
        self.p1_percent_pixels_result_label.setToolTip(
            AppStyles.AppToolTips.PERCENT_PIXELS_RESULT)
        self.p2_percent_pixels_result_label.setToolTip(
            AppStyles.AppToolTips.PERCENT_PIXELS_RESULT)

    def _setup_connections(self):
        """Connect checkbox state changes to emit criteria_enabled_changed."""
        self.mean_slope_checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.match_score_checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.percent_pixels_checkbox.stateChanged.connect(self._on_checkbox_changed)

    def _on_checkbox_changed(self):
        """Handle any checkbox state change - emit current enabled states."""
        enabled_states = self.get_criteria_enabled()
        logger.info(f"Criteria enabled changed: {enabled_states}")
        self.criteria_enabled_changed.emit(enabled_states)

    def _setup_layout(self):
        """Set up the group box layout."""
        # Main layout in the group box
        main_layout = QVBoxLayout()
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
        # Reserve each cell's size for the largest content it can ever show,
        # so the two-line "100.00\n(stall)" result or the "Foreground Energy"
        # criterion label never grows the layout minimum and resizes the
        # window mid-run. Measured on the REAL widgets -- stylesheet fonts,
        # checkbox indicator and spacing included -- by setting the largest
        # strings, taking the size hints, and restoring (font-metrics math
        # under-reserved: stylesheet fonts and indicator spacing are not
        # visible to QFontMetrics on the widget font).
        saved_checkbox_text = self.percent_pixels_checkbox.text()
        saved_label_text = self.p1_percent_pixels_result_label.text()
        self.percent_pixels_checkbox.setText(_WIDEST_CRITERION_TEXT)
        self.p1_percent_pixels_result_label.setText(_LARGEST_RESULT_TEXT)
        grid_layout.setColumnMinimumWidth(
            0, self.percent_pixels_checkbox.sizeHint().width() + 4)
        largest_hint = self.p1_percent_pixels_result_label.sizeHint()
        value_width = largest_hint.width() + 4   # widest LINE of the two
        grid_layout.setColumnMinimumWidth(1, value_width)
        grid_layout.setColumnMinimumWidth(2, value_width)
        # Two-line height reservation on both percent-pixels labels: without
        # it the row would grow downward when "(stall)" lands -- the same
        # mid-run jump, rotated vertically.
        self.p1_percent_pixels_result_label.setMinimumHeight(largest_hint.height())
        self.p2_percent_pixels_result_label.setMinimumHeight(largest_hint.height())
        self.percent_pixels_checkbox.setText(saved_checkbox_text)
        self.p1_percent_pixels_result_label.setText(saved_label_text)
        # Add components to grid layout
        grid_layout.addWidget(self.pattern1_label, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.pattern2_label, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.mean_slope_checkbox, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.p1_mean_slope_result_label, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.p2_mean_slope_result_label, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.match_score_checkbox, 2, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.p1_match_score_result_label, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.p2_match_score_result_label, 2, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.percent_pixels_checkbox, 3, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.p1_percent_pixels_result_label, 3, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.p2_percent_pixels_result_label, 3, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.confirmation_rounds_label, 4, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.p1_confirmation_rounds_result_label, 4, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.p2_confirmation_rounds_result_label, 4, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.pattern_duration_label, 5, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.p1_pattern_duration_result_label, 5, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self.p2_pattern_duration_result_label, 5, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        # Add grid layout to main layout
        main_layout.addLayout(grid_layout)
        
        return main_layout

    # ===========================
    # Public API
    # ===========================

    def get_criteria_enabled(self) -> dict:
        """
        Get the current enabled state of all criteria checkboxes.
        
        :return: Dictionary with enabled flags
        """
        return {
            'mean_slope_enabled': self.mean_slope_checkbox.isChecked(),
            'match_score_enabled': self.match_score_checkbox.isChecked(),
            'percent_pixels_enabled': self.percent_pixels_checkbox.isChecked(),
        }

    def set_criteria_enabled(self, states: dict):
        """
        Set checkbox states from a dictionary (e.g. restored from QSettings).
        
        :param states: Dictionary with keys matching get_criteria_enabled() output
        """
        checkbox_map = {
            'mean_slope_enabled': self.mean_slope_checkbox,
            'match_score_enabled': self.match_score_checkbox,
            'percent_pixels_enabled': self.percent_pixels_checkbox,
        }
        for key, checkbox in checkbox_map.items():
            if key in states:
                checkbox.setChecked(bool(states[key]))

    # ===========================
    # Label Routing Helper
    # ===========================

    def _get_label(self, key: str, pattern_idx: int) -> QLabel:
        """
        Get the result label for a given key and pattern index.
        
        :param key: Label key ('mean_slope', 'match_score',
                    'percent_pixels', 'confirmation_rounds', 'duration')
        :param pattern_idx: Pattern index (0 or 1)
        :return: The QLabel widget
        """
        return self._result_labels[pattern_idx][key]

    # ===========================
    # Mean Slope
    # ===========================

    def set_mean_slope_result(self, value: str, pattern_idx: int = 0):
        """Set the mean slope result label."""
        label = self._get_label('mean_slope', pattern_idx)
        label.setText(f"{value}")
        logger.debug(f"Pattern {pattern_idx + 1} Mean Slope result: {value}")

    def set_mean_slope_result_match(self, is_match: bool, pattern_idx: int = 0):
        """Set the mean slope result label style based on whether it's a match."""
        label = self._get_label('mean_slope', pattern_idx)
        if is_match:
            label.setStyleSheet(AppStyles.Label.result_match())
        else:
            label.setStyleSheet(AppStyles.Label.result_default())

    # ===========================
    # Match Score
    # ===========================

    def set_match_score_result(self, value: str, pattern_idx: int = 0):
        """Set the match score result label."""
        label = self._get_label('match_score', pattern_idx)
        label.setText(f"{value}")
        logger.debug(f"Pattern {pattern_idx + 1} Match Score result: {value}")

    def set_match_score_result_match(self, is_match: bool, pattern_idx: int = 0):
        """Set the match score result label style based on whether it's a match."""
        label = self._get_label('match_score', pattern_idx)
        if is_match:
            label.setStyleSheet(AppStyles.Label.result_match())
        else:
            label.setStyleSheet(AppStyles.Label.result_default())

    # ===========================
    # Percent Pixels
    # ===========================

    def set_pixels_criterion_label(self, text: str):
        """
        Set the percent-pixels criterion checkbox text.

        The main window calls this to keep the criterion honest per binarization
        method: "Percent Pixels" for the white-pixel methods, "Foreground Energy"
        for the Top-Hat continuous-energy method. Only the display text changes;
        the checkbox's checked state still drives the criterion.

        :param text: Checkbox text to display
        """
        self.percent_pixels_checkbox.setText(text)

    def set_percent_pixels_result(self, value: str, pattern_idx: int = 0):
        """Set the percent pixels result label."""
        label = self._get_label('percent_pixels', pattern_idx)
        label.setText(f"{value}")
        logger.debug(f"Pattern {pattern_idx + 1} Percent Pixels result: {value}")

    def set_percent_pixels_result_match(self, is_match: bool, pattern_idx: int = 0):
        """Set the percent pixels result label style based on whether it's a match."""
        label = self._get_label('percent_pixels', pattern_idx)
        if is_match:
            label.setStyleSheet(AppStyles.Label.result_match())
        else:
            label.setStyleSheet(AppStyles.Label.result_default())

    # ===========================
    # Confirmation Rounds
    # ===========================

    def set_confirmation_rounds_total(self, total: int, pattern_idx: int = -1):
        """
        Initialize confirmation rounds display with total (e.g., '/3').
        
        :param total: Total confirmation rounds required
        :param pattern_idx: Pattern index (0 or 1), or -1 for both patterns
        """
        indices = [0, 1] if pattern_idx == -1 else [pattern_idx]
        for idx in indices:
            label = self._get_label('confirmation_rounds', idx)
            label.setText(f"/{total}")
        logger.info(f"Confirmation rounds total set to {total} "
                    f"(pattern_idx={pattern_idx})")

    def update_confirmation_rounds_progress(self, current: int, total: int, pattern_idx: int = 0):
        """Update confirmation rounds with current progress (e.g., '2/3')."""
        label = self._get_label('confirmation_rounds', pattern_idx)
        label.setText(f"{current}/{total}")
        logger.debug(f"Pattern {pattern_idx + 1} Confirmation Rounds progress: {current}/{total}")

    def set_confirmation_rounds_result_match(self, is_match: bool, pattern_idx: int = 0):
        """Set the confirmation rounds result label style based on whether it's a match."""
        label = self._get_label('confirmation_rounds', pattern_idx)
        if is_match:
            label.setStyleSheet(AppStyles.Label.result_match())
        else:
            label.setStyleSheet(AppStyles.Label.result_default())

    # ===========================
    # Duration
    # ===========================

    def set_duration_result(self, value: str, pattern_idx: int = 0):
        """Set the pattern duration result label."""
        label = self._get_label('duration', pattern_idx)
        label.setText(f"{value}")
        logger.info(f"Pattern {pattern_idx + 1} Duration result: {value}")

    def clear_duration_result(self, pattern_idx: int = -1):
        """
        Clear the duration result label and reset style.
        
        :param pattern_idx: Pattern index (0 or 1), or -1 for both patterns
        """
        indices = [0, 1] if pattern_idx == -1 else [pattern_idx]
        for idx in indices:
            label = self._get_label('duration', idx)
            label.setText("")
            label.setStyleSheet(AppStyles.Label.result_default())

    # ===========================
    # Bulk Operations
    # ===========================

    def clear_all_results(self, pattern_idx: int = -1):
        """
        Clear all result labels and reset styles.
        
        :param pattern_idx: Pattern index (0 or 1), or -1 for both patterns
        """
        indices = [0, 1] if pattern_idx == -1 else [pattern_idx]
        for idx in indices:
            for key in self._result_labels[idx]:
                label = self._result_labels[idx][key]
                label.setText("")
                label.setStyleSheet(AppStyles.Label.result_default())