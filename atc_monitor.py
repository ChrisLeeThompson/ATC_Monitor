"""
ATC Monitor
.
This UI is designed to monitor real-time monitor (RTM) data generated from a Thermo Scientific FIB-SEM microscope.
Specifically, it processes RTM images while Thermo Scientific AutoTEM Cryo performs rough milling with one or two patterns (Rectangle or Regular Cross-section).
For now, the script is designed for rectangle patterns (regular cross-section patterns may or may not work well, at the moment).
.
The application runs in the background while FIB patterns are generating RTM data. Based on analysis of the images, the application
will stop FIB patterning if certain criteria are met.
.
Thermo Scientific AutoScript 4.13+ is required, and the application uses only modules included with AutoScript (no extra dependencies).
.
If you have any questions or suggestions for improvements, please contact me (Chris Thompson on GitHub: ChrisLeeThompson).
.
Thank you,
Chris Thompson
.
March 13, 2026
.
.
MIT License
.
Copyright 2026 Christopher Thompson
.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
.
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import logging
import os
import sys
import platform
import faulthandler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QWidget,
    QHBoxLayout, QSizePolicy, QLabel
)
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Slot, Signal, QRect, QThread, Qt, qInstallMessageHandler, QtMsgType
from script_modules.app_styles import AppStyles
from script_modules.monitoring_icon import CatbugMonitoringIcon
from script_modules.pattern_one_group_box import PatternOneGroupBox
from script_modules.pattern_two_group_box import PatternTwoGroupBox
from script_modules.controls_group_box import ControlsGroupBox
from script_modules.pattern_results_group_box import PatternResultsGroupBox
from script_modules.button_widgets import StartButton, StopButton, PauseResumeButton
from script_modules.status_bar_widget import StatusBarWidget
from script_modules.atc_monitor_parameters import load_parameters, RTMMode
from script_modules.workflow_worker import WorkflowWorker, WorkerParameters
from script_modules.settings_dialog import SettingsDialog


logger = logging.getLogger(__name__)

# Held module-global for the process lifetime so the file object backing
# faulthandler is NEVER garbage-collected/closed. faulthandler writes to the
# raw file descriptor at crash time; a closed fd would mean the native-fault
# traceback is silently lost -- the exact failure this is meant to capture.
_FAULT_FP = None


class MainWindow(QMainWindow):

    # Control signals into the worker. These are connected to the worker's slots
    # per-Start with DirectConnection (see _connect_worker_signals), so the slots
    # run on the GUI thread and only set thread-safe flags (GIL-atomic bools) or
    # queue into the lock-protected WorkerParameterManager -- the worker loop polls
    # those each iteration. DirectConnection is required because run() blocks the
    # worker thread without pumping its event loop, so a QueuedConnection would
    # never be delivered. Emitting one with no worker connected is a harmless no-op.
    request_worker_stop = Signal()
    request_worker_pause = Signal()
    request_worker_resume = Signal()
    request_worker_param = Signal(str, object, int)  # (param_name, value, pattern_idx)

    def __init__(self):
        super().__init__()

        # Load parameters (restores last session, falls back to defaults)
        self.params = load_parameters()

        # Per-pattern "results frozen" flags. Once a pattern meets all criteria,
        # its results panel freezes on the met values (further pattern_results_ready
        # updates are ignored) until the next session resets it. Plots/RTM keep
        # updating. Index = pattern index (0 or 1).
        self._results_frozen = [False, False]

        # Window title
        self.setWindowTitle(AppStyles.AppText.WINDOW_TITLE)
        # Set window geometry (position and size)
        self.setGeometry(100, 100, AppStyles.Dimensions.WINDOW_WIDTH, AppStyles.Dimensions.WINDOW_HEIGHT)
        # Set window icon
        self.assets_base_path = Path(__file__).parent / "script_assets"
        _icon_path = self.assets_base_path / "catbug_waiting_color.png"
        self.setWindowIcon(QIcon(str(_icon_path)))
        # Set main window style
        self.setStyleSheet(AppStyles.MainWindow.window())
        # Create central widget
        central_widget = QWidget()
        # Create components
        self._create_components()
        # Make the foreground-metric labels honest for the active binarization
        # method (Top-Hat reports a continuous energy, not a white-pixel %).
        self._refresh_metric_labels()
        # Setup connections
        self._setup_connections()
        # Setup layout
        layout = self._setup_layout()
        # Set layout
        central_widget.setLayout(layout)
        # Set status bar
        self.status_bar = StatusBarWidget(parent=self)
        self.setStatusBar(self.status_bar)
        # Set widget
        self.setCentralWidget(central_widget)
        
        # Hide plots initially (will show when monitoring starts)
        self.set_plot_visibility(False)

        # Worker and its thread (None until Start clicked). The worker is a
        # QObject moved onto worker_thread (see on_start_clicked).
        self.worker = None
        self.worker_thread = None

        # Drives information_label_2 (catbug status message). Combined via
        # _update_monitoring_message() with precedence stopped > paused >
        # detecting > idle.
        self._monitoring_running = False
        self._monitoring_paused = False
        self._monitoring_detecting = False   # mirrors catbug color: True = colored/active

        # Connect UI controls to the worker once. Handlers emit request_worker_*
        # signals (queued to the worker), so there is no per-Start re-connection
        # (which previously leaked connections) and the wiring is worker-agnostic.
        self._connect_live_parameter_updates()

    def _create_components(self):
        """Create components/widgets for the main window."""
        # Create controls group box with parameters
        self.controls_group_box = ControlsGroupBox(
            parent=self,
            constraints=self.params.spin_box_constraints,
            initial_values=self.params.ui
        )
        # Get threshold parameters for pattern one plot 
        initial_threshold = self.params.ui.match_score_threshold
        threshold_min = self.params.spin_box_constraints.match_score_threshold_min
        threshold_max = self.params.spin_box_constraints.match_score_threshold_max
        # Create pattern group boxes with threshold parameters
        self.pattern_one_group_box = PatternOneGroupBox(
            parent=self,
            initial_threshold=initial_threshold,
            threshold_min=threshold_min,
            threshold_max=threshold_max
        )
        self.pattern_two_group_box = PatternTwoGroupBox(
            parent=self,
            initial_threshold=initial_threshold,
            threshold_min=threshold_min,
            threshold_max=threshold_max
        )
        # Create results group box and buttons
        self.pattern_results_group_box = PatternResultsGroupBox(
            parent=self,
            initial_confirmation_rounds=self.params.ui.confirmation_rounds
        )
        # Restore criteria checkbox states from saved settings
        self.pattern_results_group_box.set_criteria_enabled({
            'mean_slope_enabled': self.params.ui.mean_slope_enabled,
            'match_score_enabled': self.params.ui.match_score_enabled,
            'percent_pixels_enabled': self.params.ui.percent_pixels_enabled,
        })
        self.start_button = StartButton(parent=self)
        self.pause_resume_button = PauseResumeButton(parent=self)
        self.stop_button = StopButton(parent=self)

        # Monitoring icon
        _icon_active = self.assets_base_path / "catbug_color_2.png"
        _icon_inactive = self.assets_base_path / "catbug_grayscale_2.png"
        self.monitoring_icon = CatbugMonitoringIcon(
            active_icon_path=str(_icon_active),
            inactive_icon_path=str(_icon_inactive),
            scale_factor=4.0
        )

        # Info label
        self.information_label_1 = QLabel(AppStyles.AppText.WINDOW_INFO_LABEL_1)
        self.information_label_1.setWordWrap(True)
        self.information_label_1.setContentsMargins(10, 10, 10, 10)
        self.information_label_1.setStyleSheet(AppStyles.Label.default())
        self.information_label_2 = QLabel(AppStyles.AppText.WINDOW_INFO_LABEL_2)
        self.information_label_2.setStyleSheet(AppStyles.Label.default())

        # Info label and icon container
        self.info_container = QWidget()
        info_layout = QHBoxLayout(self.info_container)
        info_layout.addWidget(self.information_label_2)
        info_layout.addWidget(self.monitoring_icon, alignment=Qt.AlignmentFlag.AlignRight)
        info_layout.setContentsMargins(10, 10, 10, 10)

        # Initial button states
        self.start_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Indexed access for pattern dispatching
        self.pattern_group_boxes = [
            self.pattern_one_group_box,
            self.pattern_two_group_box
        ]
        
        # Restore saved crop rectangles from previous session
        saved_crops = [
            self.params.processing.crop_rect_pattern_1,
            self.params.processing.crop_rect_pattern_2
        ]
        for group_box, crop_rect in zip(self.pattern_group_boxes, saved_crops):
            if crop_rect is not None:
                group_box.rtm_plot.set_saved_crop_rect(
                    crop_rect.x(), crop_rect.y(),
                    crop_rect.width(), crop_rect.height()
                )
    
    def _setup_connections(self):
        """Setup signal and slot connections."""
        # Pattern threshold line changes (from interactive plot dragging)
        self.pattern_one_group_box.threshold_changed.connect(
            lambda val: self._on_match_score_threshold_changed_from_plot(val, source_idx=0)
        )
        self.pattern_two_group_box.threshold_changed.connect(
            lambda val: self._on_match_score_threshold_changed_from_plot(val, source_idx=1)
        )
        # Controls spinbox to pattern one threshold
        self.controls_group_box.match_score_threshold_value_changed.connect(self._on_controls_match_score_threshold_changed)
        # Controls parameters changed - update central parameters
        self.controls_group_box.control_parameters_changed.connect(self._on_control_parameters_changed)
        # Crop rectangle changed - update central parameters
        self.pattern_one_group_box.crop_changed.connect(self._on_pattern_one_crop_changed)
        self.pattern_two_group_box.crop_changed.connect(self._on_pattern_two_crop_changed)
        
        # Button connections
        self.start_button.clicked.connect(self.on_start_clicked)
        self.pause_resume_button.clicked.connect(self.on_pause_resume_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        # Catbug color/grayscale changes -> update the status message. Explicit
        # QueuedConnection (both objects are on the GUI thread) so the message
        # setText lands on a fresh event-loop turn instead of re-entering
        # synchronously inside set_monitoring_active during the startup signal storm.
        self.monitoring_icon.monitoring_status_changed.connect(
            self._on_monitoring_icon_changed, Qt.ConnectionType.QueuedConnection
        )

        # Settings dialog
        self.controls_group_box.settings_requested.connect(self._on_settings_requested)

    def _create_pattern_column(self, widget: QWidget) -> QVBoxLayout:
        """Create a column layout containing a single pattern group box with consistent width."""
        column_layout = QVBoxLayout()
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        container = QWidget()
        container.setMinimumWidth(AppStyles.Dimensions.COLUMN_LAYOUT_MINIMUM_WIDTH)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        container_layout.addWidget(widget)
        column_layout.addWidget(container)
        return column_layout

    def _right_column_layout(self) -> QVBoxLayout:
        """
        Create the right column layout. This layout contains the controls group box, 
        pattern results group box, and control buttons.
        """
        right_column_layout = QVBoxLayout()
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column_layout.setSpacing(0)
        # Create a container widget to enforce consistent width
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred
        )
        # Right column widgets
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.controls_group_box)
        container_layout.addWidget(self.pattern_results_group_box)
        container_layout.addStretch()
        container_layout.addWidget(self.information_label_1)
        container_layout.addWidget(self.info_container)
        # Add button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_resume_button)
        button_layout.addWidget(self.stop_button)
        container_layout.addLayout(button_layout)
        # Add container to right_column_layout
        right_column_layout.addWidget(container)
        
        return right_column_layout

    def _setup_layout(self) -> QHBoxLayout:
        """Set up the main layout of the application."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self._create_pattern_column(self.pattern_one_group_box))
        main_layout.addLayout(self._create_pattern_column(self.pattern_two_group_box))
        main_layout.addLayout(self._right_column_layout())
        self.setContentsMargins(
            AppStyles.Dimensions.WINDOW_CONTENTS_MARGIN,
            AppStyles.Dimensions.WINDOW_CONTENTS_MARGIN,
            AppStyles.Dimensions.WINDOW_CONTENTS_MARGIN,
            AppStyles.Dimensions.WINDOW_CONTENTS_MARGIN,
        )
        return main_layout

    def _on_match_score_threshold_changed_from_plot(self, threshold: float, source_idx: int):
        """Handle threshold changes from a pattern plot. Update the controls spinbox and the other pattern's line."""
        # Block signals to prevent loop
        spinbox = self.controls_group_box.match_score_threshold_spinbox
        spinbox.blockSignals(True)
        spinbox.setValue(threshold)
        spinbox.blockSignals(False)
        # Update central parameters
        self.params.ui.match_score_threshold = threshold
        # Update the other pattern's threshold line to match
        other_idx = 1 - source_idx
        self.pattern_group_boxes[other_idx].set_match_score_threshold(threshold)
        # Push to worker (spinbox signals blocked, so worker connection won't fire)
        if self.worker:
            self._update_both_patterns('match_score_threshold', threshold)
    
    @Slot(float)
    def _on_controls_match_score_threshold_changed(self, threshold: float):
        """Handle threshold changes from controls spinbox."""
        for group_box in self.pattern_group_boxes:
            group_box.set_match_score_threshold(threshold)

    @Slot(dict)
    def _on_control_parameters_changed(self, params_dict: dict):
        """Handle parameter changes from controls group box."""
        self.params.update_ui_from_dict(params_dict)

        # Update confirmation rounds display in results group box
        if "confirmation_rounds" in params_dict:
            total_rounds = params_dict["confirmation_rounds"]
            self.pattern_results_group_box.set_confirmation_rounds_total(total_rounds)
        logger.info(f"Updated global parameters: {params_dict}")
    
    @Slot(int, int, int, int)
    def _on_pattern_one_crop_changed(self, x: int, y: int, width: int, height: int):
        """Handle crop rectangle changes from Pattern One RTM plot."""
        self.params.update_crop_rect_pattern_1(x, y, width, height)
        logger.info(f"Pattern 1 crop updated: x={x}, y={y}, width={width}, height={height}")
    
    @Slot(int, int, int, int)
    def _on_pattern_two_crop_changed(self, x: int, y: int, width: int, height: int):
        """Handle crop rectangle changes from Pattern Two RTM plot."""
        self.params.update_crop_rect_pattern_2(x, y, width, height)
        logger.info(f"Pattern 2 crop updated: x={x}, y={y}, width={width}, height={height}")
    
    def _refresh_metric_labels(self):
        """
        Keep the foreground-metric labels honest for the active binarization method.

        The Top-Hat method reports a continuous foreground *energy*, not a
        white-pixel percentage, and its display panel is a grayscale foreground
        map rather than a binary image. So when it is active we relabel the
        threshold and the criterion accordingly; every other method keeps the
        original "pixels" wording. Called at construction and whenever settings
        are applied (the method cannot change mid-run). The panel title is
        handled separately in on_binary_image_ready.
        """
        is_energy = self.params.processing.binarization_method.name == "TOPHAT_ENERGY"
        self.controls_group_box.set_pixels_threshold_label(
            "Max Foreground Energy" if is_energy else "Maximum Pixels Threshold"
        )
        self.pattern_results_group_box.set_pixels_criterion_label(
            "Foreground Energy" if is_energy else "Percent Pixels"
        )

    @Slot()
    def _on_settings_requested(self):
        """Open the settings dialog and apply changes on accept."""
        dialog = SettingsDialog(
            parent=self,
            constraints=self.params.spin_box_constraints,
            processing_params=self.params.processing
        )
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.params.processing = dialog.get_parameters()
            self.params.save_settings()
            # The binarization method may have changed -> keep the foreground
            # metric labels honest (white-pixel % vs Top-Hat energy).
            self._refresh_metric_labels()
            logger.info("Advanced settings updated and saved")
    
    # ========================================
    # Button Click Handlers
    # ========================================
    
    @Slot()
    def on_start_clicked(self):
        """Handle Start button click - create and start worker on its thread."""
        # Never start a second run while one is active (that would orphan or
        # destroy a running QThread).
        if self.worker_thread is not None and self.worker_thread.isRunning():
            logger.warning("Start ignored - a monitoring run is already active")
            return

        # Release the previous (finished) worker/thread before creating new ones.
        self._reap_worker()

        # Update button states immediately
        self.start_button.setEnabled(False)
        self.pause_resume_button.setEnabled(True)
        self.pause_resume_button.setText("Pause")
        self.stop_button.setEnabled(True)
        self.controls_group_box.set_settings_button_enabled(False)

        # Disable mode switching and save data during processing
        self.controls_group_box.set_monitoring_mode_enabled(False)
        self.controls_group_box.save_data_checkbox.setEnabled(False)

        # Run begins -> catbug message shows "Monitoring..." (gray until first detection)
        self._monitoring_running = True
        self._monitoring_paused = False
        self._monitoring_detecting = False
        self._update_monitoring_message()

        # Create worker parameters from current UI values
        params = self._create_worker_parameters()

        # Create the worker and move it onto a dedicated thread.
        self.worker = WorkflowWorker(params)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Lifecycle wiring:
        # - started -> run via QueuedConnection so run() executes *inside* the
        #   thread's event loop (lets it return cleanly via quit()).
        # - finished -> quit via DirectConnection so the thread is told to exit
        #   on the worker thread itself, with no dependency on the GUI event
        #   loop (which is blocked during closeEvent's wait()).
        self.worker_thread.started.connect(self.worker.run, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(
            self.worker_thread.quit, Qt.ConnectionType.DirectConnection
        )

        # Connect worker<->GUI signals and the queued control signals.
        self._connect_worker_signals()

        # Start the thread (its event loop will dispatch the queued run()).
        self.worker_thread.start()

        logger.info("Start button clicked - worker thread started")

    def _reap_worker(self):
        """
        Release references to a previously finished worker/thread so Python/Qt
        can delete them. Only called when no run is active (the thread has
        finished), so dropping the references is safe and prevents the prior
        per-Start object leak.
        """
        if self.worker_thread is not None and self.worker_thread.isRunning():
            # Defensive: should not happen (callers guard on isRunning()).
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
        self.worker = None
        self.worker_thread = None
    
    @Slot()
    def on_pause_resume_clicked(self):
        """Handle Pause/Resume button click - toggle pause state."""
        if not self.worker:
            return

        if self.pause_resume_button.text() == "Pause":
            # Request pause (queued to the worker thread)
            self.request_worker_pause.emit()
        else:
            # Request resume (queued to the worker thread)
            self.request_worker_resume.emit()

    @Slot()
    def on_stop_clicked(self):
        """Handle Stop button click - request worker to stop."""
        if self.worker:
            # Disable stop button
            self.stop_button.setEnabled(False)

            # Request worker to stop (queued to the worker thread)
            self.request_worker_stop.emit()

            logger.info("Stop button clicked - stop requested")
    
    def _create_worker_parameters(self) -> WorkerParameters:
        """
        Create WorkerParameters from current UI state.
        
        Uses centralized parameter architecture:
        1. Sync UI params from controls
        2. Read ONLY from self.params when creating worker
        3. Include criteria enabled flags from checkboxes
        """
        # Sync central params with current UI state
        current_ui_params = self.controls_group_box.get_control_parameters()
        self.params.update_ui_from_dict(current_ui_params)
        
        # Get current checkbox enabled states
        criteria_enabled = self.pattern_results_group_box.get_criteria_enabled()
        
        # Create worker params from centralized params
        return WorkerParameters(
            # Microscope
            microscope_host="localhost",
            
            # Processing parameters (from central params)
            number_of_images=self.params.ui.number_of_images,
            gaussian_sigma=self.params.processing.gaussian_sigma,
            apply_dilation=self.params.processing.apply_dilation,
            threshold_num_classes=self.params.processing.threshold_num_classes,
            binarization_method=self.params.processing.binarization_method,
            tophat_radius=self.params.processing.tophat_radius,
            match_on_foreground=self.params.processing.match_on_foreground,
            foreground_completion_mode=self.params.processing.foreground_completion_mode,
            energy_drop_fraction=self.params.processing.energy_drop_fraction,
            energy_slope_threshold=self.params.processing.energy_slope_threshold,

            # Pattern 1 parameters (from central params)
            pattern_1_crop_rect=self._qrect_to_tuple(self.params.processing.crop_rect_pattern_1),
            pattern_1_mean_slope_threshold=self.params.ui.mean_pixel_slope_threshold,
            pattern_1_match_score_threshold=self.params.ui.match_score_threshold,
            pattern_1_max_pixels_threshold=self.params.ui.maximum_pixels_threshold,
            
            # Pattern 2 parameters (from central params - thresholds are global)
            pattern_2_crop_rect=self._qrect_to_tuple(self.params.processing.crop_rect_pattern_2),
            pattern_2_mean_slope_threshold=self.params.ui.mean_pixel_slope_threshold,
            pattern_2_match_score_threshold=self.params.ui.match_score_threshold,
            pattern_2_max_pixels_threshold=self.params.ui.maximum_pixels_threshold,
            
            # Monitoring parameters (from central params)
            confirmation_rounds=self.params.ui.confirmation_rounds,
            acquisition_delay_seconds=self.params.processing.acquisition_delay_seconds,
            percent_difference_threshold=self.params.processing.percent_difference_threshold,
            aspect_ratio_threshold=self.params.processing.aspect_ratio_threshold,

            # Slope calculation parameters
            slope_method=self.params.processing.slope_method,
            num_points_for_slope=self.params.processing.num_points_for_slope,
            linear_regression_fit_points=self.params.processing.linear_regression_fit_points,
    
            # Pattern matching parameters
            min_pattern_splits=self.params.processing.min_pattern_splits,
            target_tile_size=self.params.processing.target_tile_size,
            
            # RTM settings
            rtm_mode=RTMMode.LOW_RESOLUTION,

            # Window mode
            window_mode=self.params.ui.window_mode,
            analysis_interval_seconds=self.params.ui.analysis_interval_seconds,
            min_window_size=self.params.spin_box_constraints.number_of_images_min,
            max_window_size=self.params.spin_box_constraints.number_of_images_max,
            
            # Criteria enabled flags (from checkboxes)
            mean_slope_enabled=criteria_enabled['mean_slope_enabled'],
            match_score_enabled=criteria_enabled['match_score_enabled'],
            percent_pixels_enabled=criteria_enabled['percent_pixels_enabled'],
            
            # Save data settings
            save_data=self.controls_group_box.save_data_checkbox.isChecked(),
            script_root=str(self._get_script_root()),

            # Auto contrast/brightness calibration (from central params)
            auto_cb_on_start=self.params.processing.auto_cb_on_start,
            cb_white_level=self.params.processing.cb_white_level,
            cb_target_median_fraction=self.params.processing.cb_target_median_fraction,
            cb_target_contrast_span=self.params.processing.cb_target_contrast_span,
            cb_max_white_clip_fraction=self.params.processing.cb_max_white_clip_fraction,
            cb_max_black_clip_fraction=self.params.processing.cb_max_black_clip_fraction,
            cb_min_bound=self.params.processing.cb_min_bound,
            cb_max_bound=self.params.processing.cb_max_bound,
            cb_max_iterations=self.params.processing.cb_max_iterations,
            cb_settle_seconds=self.params.processing.cb_settle_seconds,
            cb_frames_per_measurement=self.params.processing.cb_frames_per_measurement,
        )
    
    def _qrect_to_tuple(self, qrect: QRect | None) -> tuple[int, int, int, int] | None:
        """Convert QRect to tuple (x, y, width, height) or None."""
        if qrect is None:
            return None
        return (qrect.x(), qrect.y(), qrect.width(), qrect.height())
    
    def _connect_worker_signals(self):
        """Connect all worker signals to appropriate GUI slots."""
        # Control Signals
        self.worker.monitoring_started.connect(self.on_monitoring_started)
        self.worker.monitoring_stopped.connect(self.on_monitoring_stopped)
        self.worker.monitoring_paused.connect(self.on_monitoring_paused)
        self.worker.monitoring_resumed.connect(self.on_monitoring_resumed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.session_reset.connect(self.on_session_reset)
        self.worker.save_plots_requested.connect(self.on_save_plots_requested)
        
        # Status Signals
        self.worker.status_update.connect(lambda text: self.status_bar.set_timed_status_text(text, 5))
        self.worker.persistent_status_update.connect(self.status_bar.set_status_text)
        
        # Image Display Signals
        self.worker.processed_image_ready.connect(self.on_processed_image_ready)

        # Display binary images
        self.worker.binary_image_ready.connect(self.on_binary_image_ready)

        # Pattern scan direction (highlights the matching crop-box edge)
        self.worker.pattern_scan_direction_ready.connect(self.on_scan_direction_ready)
        
        # Plot Data Signals
        self.worker.mean_pixel_data_ready.connect(self.on_mean_pixel_data_ready)
        self.worker.match_score_data_ready.connect(self.on_match_score_data_ready)
        self.worker.specimen_current_data_ready.connect(self.on_specimen_current_data_ready)
        self.worker.specimen_current_disabled.connect(self._on_specimen_current_disabled)

        # Calibration signal (TIME_BASED mode)
        self.worker.calibration_complete.connect(self.controls_group_box.set_auto_adjusted_interval)
        
        # Results Signals
        self.worker.pattern_results_ready.connect(self.on_pattern_results_ready)
        
        # UI Control Signals
        self.worker.show_pattern_plots.connect(self.on_show_pattern_plots)
        self.worker.set_monitoring_icon_active.connect(
            self.monitoring_icon.set_monitoring_active
        )

        # Control signals INTO the worker, wired DirectConnection on purpose.
        # run() blocks the worker thread and no longer pumps its event loop, so a
        # QueuedConnection would never be delivered. Direct slots run on the GUI
        # thread and only set thread-safe flags (stop/pause/resume bools, which
        # are GIL-atomic) or queue into the lock-protected WorkerParameterManager
        # (update_parameter). The worker loop polls those each iteration.
        _direct = Qt.ConnectionType.DirectConnection
        self.request_worker_stop.connect(self.worker.request_stop, _direct)
        self.request_worker_pause.connect(self.worker.request_pause, _direct)
        self.request_worker_resume.connect(self.worker.request_resume, _direct)
        self.request_worker_param.connect(self.worker.update_parameter, _direct)

    def _connect_live_parameter_updates(self):
        """
        Connect UI controls to the worker for live parameter updates.

        Connected ONCE (from __init__), not per-Start: handlers emit the
        request_worker_param signal, which is bound to the worker per-Start.
        With no active worker the signal has no connection and emitting is a
        no-op, so these UI controls are safe to leave connected for the app's
        lifetime (this also fixes the prior per-Start connection leak).
        """
        # Threshold updates apply to BOTH patterns (global thresholds)
        self.controls_group_box.mean_pixel_slope_threshold_spinbox.valueChanged.connect(
            lambda val: self._update_both_patterns('mean_slope_threshold', val)
        )

        self.controls_group_box.match_score_threshold_spinbox.valueChanged.connect(
            lambda val: self._update_both_patterns('match_score_threshold', val)
        )

        self.controls_group_box.maximum_pixels_threshold_spinbox.valueChanged.connect(
            lambda val: self._update_both_patterns('max_pixels_threshold', val)
        )

        # Crop rectangles are INDEPENDENT per pattern
        self.pattern_one_group_box.rtm_plot.crop_changed.connect(
            lambda x, y, w, h: self.request_worker_param.emit(
                'pattern_1_crop_rect', (x, y, w, h), 0
            )
        )

        self.pattern_two_group_box.rtm_plot.crop_changed.connect(
            lambda x, y, w, h: self.request_worker_param.emit(
                'pattern_2_crop_rect', (x, y, w, h), 1
            )
        )

        # Global Parameters
        self.controls_group_box.confirmation_rounds_spinbox.valueChanged.connect(
            lambda val: self.request_worker_param.emit('confirmation_rounds', int(val), -1)
        )

        self.controls_group_box.number_of_images_spinbox.valueChanged.connect(
            lambda val: self.request_worker_param.emit('number_of_images', int(val), -1)
        )

        self.controls_group_box.analysis_interval_seconds_spinbox.valueChanged.connect(
            lambda val: self.request_worker_param.emit('analysis_interval_seconds', int(val), -1)
        )

        # Criteria enabled checkboxes → worker (global, not per-pattern)
        self.pattern_results_group_box.criteria_enabled_changed.connect(
            self._on_criteria_enabled_changed
        )

    @Slot(dict)
    def _on_criteria_enabled_changed(self, enabled_states: dict):
        """
        Handle criteria checkbox changes from pattern results group box.

        Routes each enabled flag to the worker as a separate parameter update.
        """
        for param_name, value in enabled_states.items():
            self.request_worker_param.emit(param_name, value, -1)

    def _update_both_patterns(self, param_name: str, value: float):
        """
        Update threshold parameter for both patterns.

        Thresholds are global - they apply to both Pattern 1 and Pattern 2.
        This ensures consistency when user changes threshold values.
        """
        self.request_worker_param.emit(f'pattern_1_{param_name}', value, 0)
        self.request_worker_param.emit(f'pattern_2_{param_name}', value, 1)
    
    # ========================================
    # Catbug status message
    # ========================================

    def _update_monitoring_message(self):
        """Set the catbug status message (information_label_2) from current state.
        Precedence: stopped > paused > detecting > idle."""
        if not self._monitoring_running:
            text = AppStyles.AppText.WINDOW_INFO_LABEL_2            # "Click Start to begin monitoring."
        elif self._monitoring_paused:
            text = AppStyles.AppText.WINDOW_INFO_MONITORING_PAUSED  # "Monitoring paused"
        elif self._monitoring_detecting:
            text = AppStyles.AppText.WINDOW_INFO_MONITORING_ACTIVE  # "Monitoring active..."
        else:
            text = AppStyles.AppText.WINDOW_INFO_MONITORING_IDLE    # "Monitoring..."
        self.information_label_2.setText(text)

    @Slot(bool)
    def _on_monitoring_icon_changed(self, active: bool):
        """Catbug flipped color/grayscale -> refresh the status message."""
        self._monitoring_detecting = active
        self._update_monitoring_message()

    # ========================================
    # Worker Signal Handlers - Control
    # ========================================

    @Slot()
    def on_monitoring_started(self):
        """Handle monitoring_started signal from worker."""
        # Buttons already set when Start clicked, but confirm states
        self.start_button.setEnabled(False)
        self.pause_resume_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        logger.info("Monitoring started (worker confirmed)")
    
    @Slot(str)
    def on_monitoring_stopped(self, message: str):
        """Handle monitoring_stopped signal from worker."""
        # Reset button states
        self.start_button.setEnabled(True)
        self.pause_resume_button.setEnabled(False)
        self.pause_resume_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.controls_group_box.set_settings_button_enabled(True)
        
        # Re-enable mode switching and save data
        self.controls_group_box.set_monitoring_mode_enabled(True)
        self.controls_group_box.save_data_checkbox.setEnabled(True)

        # Run ended -> catbug message reverts to "Click Start to begin monitoring."
        self._monitoring_running = False
        self._monitoring_paused = False
        self._monitoring_detecting = False
        self._update_monitoring_message()

        # Display message
        self.status_bar.set_status_text(message)
        
        logger.info(f"Monitoring stopped: {message}")
    
    @Slot()
    def on_monitoring_paused(self):
        """Handle monitoring_paused signal from worker."""
        # Update button text to show Resume option
        self.pause_resume_button.setText("Resume")
        # Catbug message -> "Monitoring paused"
        self._monitoring_paused = True
        self._update_monitoring_message()
        logger.info("Monitoring paused (worker confirmed)")

    @Slot()
    def on_monitoring_resumed(self):
        """Handle monitoring_resumed signal from worker."""
        # Update button text to show Pause option
        self.pause_resume_button.setText("Pause")
        # Catbug message -> back to "Monitoring..." / "Monitoring active..." per icon
        self._monitoring_paused = False
        self._update_monitoring_message()
        logger.info("Monitoring resumed (worker confirmed)")
    
    @Slot(str)
    def on_error_occurred(self, error_message: str):
        """Handle error_occurred signal from worker."""
        # Show error in status bar
        self.status_bar.set_status_text(f"Error: {error_message}")
        logger.error(f"Worker error: {error_message}")
    
    @Slot()
    def on_session_reset(self):
        """Handle session_reset signal - clear plots for new patterning session."""
        # Unfreeze results panels so the new session updates them again
        # (clear_plot also resets each plot's criteria-met marker).
        self._results_frozen = [False, False]
        for group_box in self.pattern_group_boxes:
            group_box.mean_pixel_plot.clear_plot()
            group_box.match_score_plot.clear_plot()
        logger.info("Plots cleared for new patterning session")
    
    @Slot(str)
    def on_save_plots_requested(self, plots_dir: str):
        """
        Handle save_plots_requested signal - save plot snapshots as PNGs.
        
        Called from the worker thread via signal, executes on the GUI thread
        where the plot widgets live.
        
        :param plots_dir: Path to the plots directory
        """
        try:
            plots_path = Path(plots_dir)
            save_kwargs = dict(dpi=150, bbox_inches='tight', pad_inches=0.1)
            
            for i, group_box in enumerate(self.pattern_group_boxes, 1):
                group_box.mean_pixel_plot.figure.savefig(
                    plots_path / f"pattern_{i}_mean_pixel.png", **save_kwargs
                )
                group_box.match_score_plot.figure.savefig(
                    plots_path / f"pattern_{i}_match_score.png", **save_kwargs
                )
            
            logger.info(f"Plot snapshots saved to {plots_dir}")
            self.status_bar.set_timed_status_text("Plot snapshots saved", 5)
            
        except Exception as e:
            logger.error(f"Failed to save plot snapshots: {e}")
    
    # ========================================
    # Worker Signal Handlers - Images
    # ========================================
    
    @Slot(object, int)
    def on_processed_image_ready(self, image, pattern_idx):
        """Handle processed_image_ready signal - display when grayscale mode is active."""
        if not self.controls_group_box.show_grayscale_images_checkbox.isChecked():
            return  # Binary mode - skip grayscale display
        self.pattern_group_boxes[pattern_idx].rtm_plot.plot_image(
            image, title="Processed RTM Image"
        )

    @Slot(object, int)
    def on_binary_image_ready(self, image, pattern_idx):
        """Handle binary_image_ready signal - display when binary mode is active (default)."""
        if self.controls_group_box.show_grayscale_images_checkbox.isChecked():
            return  # Grayscale mode - skip binary display
        # The Top-Hat method emits a grayscale foreground map, not a binary image,
        # so title it honestly when that method is active.
        is_energy = self.params.processing.binarization_method.name == "TOPHAT_ENERGY"
        title = "Top-Hat Foreground" if is_energy else "Binary RTM Image"
        self.pattern_group_boxes[pattern_idx].rtm_plot.plot_image(
            image, title=title
        )

    @Slot(int, str, float)
    def on_scan_direction_ready(self, pattern_idx, scan_direction, rotation):
        """Highlight the crop-box edge matching the pattern's (rotated) scan direction."""
        self.pattern_group_boxes[pattern_idx].rtm_plot.set_scan_direction(scan_direction, rotation)
    
    # ========================================
    # Worker Signal Handlers - Plots
    # ========================================
    
    @Slot(list, list, int)
    def on_mean_pixel_data_ready(self, x_data, y_data, pattern_idx):
        """Handle mean_pixel_data_ready signal - update mean pixel plot."""
        self.pattern_group_boxes[pattern_idx].mean_pixel_plot.plot_pixel_data(x_data, y_data)
    
    @Slot(list, list, int)
    def on_specimen_current_data_ready(self, x_data, y_data, pattern_idx):
        """Handle specimen_current_data_ready signal - update specimen current plot."""
        self.pattern_group_boxes[pattern_idx].mean_pixel_plot.plot_current_data(x_data, y_data)
    
    @Slot()
    def _on_specimen_current_disabled(self):
        """Handle specimen_current_disabled signal - update plot titles for Arctis systems."""
        for group_box in self.pattern_group_boxes:
            group_box.mean_pixel_plot.set_specimen_current_enabled(False)
    
    @Slot(list, list, int)
    def on_match_score_data_ready(self, x_data, y_data, pattern_idx):
        """Handle match_score_data_ready signal - update match score plot."""
        self.pattern_group_boxes[pattern_idx].match_score_plot.plot_data(x_data, y_data)
    
    # ========================================
    # Worker Signal Handlers - Results
    # ========================================
    
    @Slot(dict, int)
    def on_pattern_results_ready(self, results, pattern_idx):
        """
        Handle pattern_results_ready signal - update results display.
        
        results dict contains:
        - mean_slope, match_score, white_pixels (float or 'Delay')
        - confirmation_count, confirmation_total (int)
        - is_complete, all_criteria_met, is_delay (bool)
        - slope_ok, match_ok, pixels_ok (bool, monitoring phase only) -- the
          worker's authoritative per-criterion pass/fail for the indicators
        - criteria_met_batch (int or None)
        - thresholds (if not delay)
        """
        # Once a pattern has met its criteria, freeze its results panel on the
        # met values - ignore all further updates until the next session resets
        # the flag. RTM images and plots keep updating (handled by other slots).
        if self._results_frozen[pattern_idx]:
            return

        results_widget = self.pattern_results_group_box
        
        # Update result labels
        if results['is_delay']:
            # Show delay status
            results_widget.set_mean_slope_result(str(results['mean_slope']), pattern_idx)
            results_widget.set_match_score_result(str(results['match_score']), pattern_idx)
            results_widget.set_percent_pixels_result(str(results['white_pixels']), pattern_idx)
            
            # No highlighting during delay
            results_widget.set_mean_slope_result_match(False, pattern_idx)
            results_widget.set_match_score_result_match(False, pattern_idx)
            results_widget.set_percent_pixels_result_match(False, pattern_idx)

            # Clear duration from previous session
            results_widget.clear_duration_result(pattern_idx)
        else:
            # Show actual values
            mean_slope = results['mean_slope']
            match_score = results['match_score']
            white_pixels = results['white_pixels']
            
            results_widget.set_mean_slope_result(f"{mean_slope:.4f}", pattern_idx)
            results_widget.set_match_score_result(f"{match_score:.4f}", pattern_idx)
            results_widget.set_percent_pixels_result(f"{white_pixels:.2f}", pattern_idx)

            # Highlight each criterion using the worker's authoritative pass/fail
            # (honors enabled-checkboxes and the foreground completion mode), so the
            # indicators always agree with the confirmation counter. Falls back to a
            # plain threshold compare only if a value is missing (e.g. legacy emit).
            slope_ok = results.get('slope_ok')
            match_ok = results.get('match_ok')
            pixels_ok = results.get('pixels_ok')
            if slope_ok is None:
                slope_ok = abs(mean_slope) <= results['mean_slope_threshold']
            if match_ok is None:
                match_ok = match_score <= results['match_score_threshold']
            if pixels_ok is None:
                pixels_ok = white_pixels <= results['max_pixels_threshold']

            results_widget.set_mean_slope_result_match(slope_ok, pattern_idx)
            results_widget.set_match_score_result_match(match_ok, pattern_idx)
            results_widget.set_percent_pixels_result_match(pixels_ok, pattern_idx)

            # Clear duration during active monitoring - will show once patterning complete
            results_widget.clear_duration_result(pattern_idx)
        
        # Update confirmation rounds
        count = results['confirmation_count']
        total = results['confirmation_total']
        results_widget.update_confirmation_rounds_progress(count, total, pattern_idx)
        results_widget.set_confirmation_rounds_result_match(results['is_complete'], pattern_idx)

        # Update duration if pattern completed
        if results.get('duration'):
            results_widget.set_duration_result(results['duration'], pattern_idx)

        # On the batch where criteria were met: draw the marker line on this
        # pattern's plots, then freeze the results panel. The met values rendered
        # above remain on display; subsequent emissions are ignored (see top).
        met_batch = results.get('criteria_met_batch')
        if met_batch is not None:
            group_box = self.pattern_group_boxes[pattern_idx]
            group_box.mean_pixel_plot.mark_criteria_met(met_batch)
            group_box.match_score_plot.mark_criteria_met(met_batch)
            self._results_frozen[pattern_idx] = True

    @Slot(int)
    def on_show_pattern_plots(self, pattern_idx):
        """Handle show_pattern_plots signal - make plots visible."""
        self.pattern_group_boxes[pattern_idx].set_plot_visibility(True)
    
    # ========================================
    # Utility Methods
    # ========================================
    
    @Slot(str)
    def set_status_message(self, message: str):
        """Set the status bar message."""
        self.status_bar.set_status_text(message)
    
    @Slot(bool)
    def set_monitoring_active(self, active: bool):
        """Set the monitoring status in the status bar."""
        self.monitoring_icon.set_monitoring_active(active)
    
    @Slot(bool)
    def set_plot_visibility(self, visible: bool):
        """Set the visibility of all plots in the pattern group boxes."""
        for group_box in self.pattern_group_boxes:
            group_box.set_plot_visibility(visible)
    
    @Slot(bool)
    def set_start_button_enabled(self, enabled: bool):
        """Enable or disable the start button."""
        self.start_button.setEnabled(enabled)
    
    @Slot(bool)
    def set_stop_button_enabled(self, enabled: bool):
        """Enable or disable the stop button."""
        self.stop_button.setEnabled(enabled)
    
    def _shutdown_worker(self, timeout_ms: int = 15000):
        """
        Cooperatively stop the monitoring worker and finalize its save data.

        Requests a graceful stop and waits a bounded time for the worker's
        run() loop to return -- the worker's finally/_cleanup path finalizes
        any in-progress save session before the thread exits. Falls back to
        terminate() only if the worker does not stop in time, which forfeits
        save finalization but prevents a "QThread: Destroyed while thread is
        still running" abort on teardown.

        This intentionally does NOT stop the FIB beam: a separate application
        controls the mill, so leaving milling active on close is by design.
        """
        thread = self.worker_thread
        if thread is None or not thread.isRunning():
            return

        logger.info("Shutting down monitoring worker (graceful stop requested)...")
        # Queued to the worker thread; its run() loop pumps events each iteration
        # and will see the stop request, then emit finished -> thread.quit().
        self.request_worker_stop.emit()

        if thread.wait(timeout_ms):
            logger.info("Monitoring worker stopped cleanly")
        else:
            logger.error(
                "Monitoring worker did not stop within %d ms; terminating thread "
                "(in-progress save data may not be finalized)",
                timeout_ms,
            )
            thread.terminate()
            thread.wait()

    def closeEvent(self, event):
        """Save settings and clean up when the window is closed."""
        # Stop the monitoring worker first so it can finalize any in-progress
        # save session before we tear down the window. (Does not stop the beam.)
        self._shutdown_worker()

        # The worker's finalize emits a queued save_plots_requested to this
        # (GUI) thread. We just blocked in wait(), so that request is sitting in
        # the event queue; flush it now so the plot PNGs are written before the
        # window is destroyed rather than being silently dropped on close.
        QApplication.processEvents()

        # Sync latest UI state to params before saving
        current_ui_params = self.controls_group_box.get_control_parameters()
        self.params.update_ui_from_dict(current_ui_params)
        
        # Sync criteria checkbox states
        criteria_enabled = self.pattern_results_group_box.get_criteria_enabled()
        self.params.update_ui_from_dict(criteria_enabled)
        
        # Sync crop rectangles from plot widgets (captures default 85% crop
        # even if user never manually adjusted it)
        for i, group_box in enumerate(self.pattern_group_boxes):
            crop = group_box.rtm_plot.get_crop_dimensions()
            if crop is not None:
                x, y, w, h = crop
                if i == 0:
                    self.params.update_crop_rect_pattern_1(x, y, w, h)
                else:
                    self.params.update_crop_rect_pattern_2(x, y, w, h)
        
        # Persist to QSettings
        self.params.save_settings()
        
        super().closeEvent(event)

    @staticmethod
    def _get_script_root() -> Path:
        """
        Get the script root directory for saved data.
        
        Returns the directory containing the main script file.
        """
        return Path(sys.argv[0]).resolve().parent


def _qt_message_handler(mode, context, message):
    """
    Route Qt's own diagnostics (including the qFatal() text Qt prints right
    before it calls abort()) into our log, so a Qt-triggered crash is preceded
    by the exact reason instead of a bare "Aborted".

    NOTE: this handler can only LOG a QtFatalMsg -- Qt calls abort() immediately
    after it returns, and returning does not cancel that. The actual prevention
    for the "Cannot create window: no screens available" fatal lives in
    BasePlotWidget._guarded_draw, which skips canvas draws while no display is
    attached. Do not try to "recover" here.
    """
    try:
        # Known-benign noise emitted every launch on this AutoScript env.
        if message and "Cannot find font directory" in message:
            return
        if mode == QtMsgType.QtFatalMsg:
            logger.critical(f"Qt FATAL: {message}")
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error(f"Qt CRITICAL: {message}")
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning(f"Qt WARNING: {message}")
        else:
            logger.info(f"Qt: {message}")
    except Exception:
        pass


def _qt_version_safe() -> str:
    """Return the runtime Qt version for the launch banner, or 'unknown'."""
    try:
        from PySide6.QtCore import qVersion
        return qVersion()
    except Exception:
        return "unknown"


def setup_logging():
    """
    Configure logging: stderr (as before) PLUS a durable rotating file log and
    native-fault capture to a file.

    Why the file outputs matter: this is a windowed GUI app, so when it is
    launched without an attached console, stderr goes nowhere visible. A native
    crash (e.g. a fault inside the AutoScript C transport) then closes the window
    with "nothing in the console" and no record at all. Writing the app log and
    the faulthandler traceback to files under ./logs makes such a crash leave
    durable on-disk evidence (the C+Python stack at the fault point) instead of
    dying silently.
    """
    global _FAULT_FP

    logs_dir = Path(sys.argv[0]).resolve().parent / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Never let logging setup itself crash the app; fall back to stderr-only.
        logs_dir = None

    # process id + thread name in every record so two separate-process runs
    # (launch 1 vs launch 2) and GUI vs worker thread are attributable in one
    # appended file -- the crux of diagnosing the intermittent first-Start crash.
    fmt = "%(asctime)s\t%(process)d\t%(threadName)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s"

    handlers = [logging.StreamHandler()]
    if logs_dir is not None:
        try:
            # Rotating so the log cannot grow unbounded (~25 MB cap). StreamHandler
            # (and thus FileHandler) flushes after every record, so the file
            # survives a hard crash up to the fault point.
            file_handler = RotatingFileHandler(
                logs_dir / "atc_monitor.log",
                maxBytes=5_000_000, backupCount=5, encoding="utf-8", delay=False,
            )
            file_handler.setFormatter(logging.Formatter(fmt))
            handlers.append(file_handler)
        except Exception:
            logger.warning("Could not create rotating file log handler", exc_info=True)

    logging.basicConfig(
        format=fmt,
        level=logging.INFO,
        force=True,
        handlers=handlers,
    )

    # Launch banner -- first record of each run, so runs in the appended file can
    # be told apart and the environment (Python/Qt/OS) is captured for triage.
    logger.info(
        "LAUNCH app=3.3.0 pid=%s exe=%s py=%s qt=%s platform=%s",
        os.getpid(), sys.executable,
        sys.version.replace("\n", " "), _qt_version_safe(), platform.platform(),
    )

    # Capture Qt's own fatal/critical messages (names the exact qFatal reason if
    # a Qt assertion ever aborts the process).
    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        logger.warning("Could not install Qt message handler", exc_info=True)

    # Dump a Python+C traceback (all threads) if the process is killed by a native
    # fault (e.g. inside the AutoScript C transport), which a normal try/except
    # cannot catch. Direct it to a SEPARATE, append-mode, kept-open file (NOT the
    # logging handler's file): faulthandler writes through the raw fd at crash
    # time and must not interleave with the logging buffer. Append so launch 2
    # never overwrites launch 1's fault. This is the single highest-value
    # diagnostic -- it converts a silent native death into a stack trace on disk.
    try:
        if logs_dir is not None:
            _FAULT_FP = open(
                logs_dir / "faulthandler.log", "a", buffering=1, encoding="utf-8"
            )
            _FAULT_FP.write(f"\n==== faulthandler armed pid={os.getpid()} ====\n")
            _FAULT_FP.flush()
            faulthandler.enable(file=_FAULT_FP, all_threads=True)
        else:
            faulthandler.enable(all_threads=True)  # fall back to stderr
    except Exception:
        logger.warning("Could not enable faulthandler-to-file", exc_info=True)
        try:
            faulthandler.enable(all_threads=True)  # last-resort fallback
        except Exception:
            pass


def _install_excepthook(window: "MainWindow"):
    """
    Install a global exception handler for unhandled exceptions.

    PySide6 routes unhandled exceptions raised inside slots/event handlers
    through sys.excepthook. Without this, such a crash would tear the app down
    with the monitoring worker still running and its save session unfinalized.

    The handler logs the traceback, then best-effort stops the worker and
    finalizes its saved data (it does NOT stop the beam -- a separate app owns
    the mill). Finally it chains to the default hook so the traceback still
    reaches stderr.
    """
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )
        try:
            window._shutdown_worker()
        except Exception:
            logger.error(
                "Error during emergency worker shutdown", exc_info=True
            )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook


def main():
    """Main function to run the application."""

    # Setup logging
    setup_logging()

    app = QApplication(sys.argv)
    font = app.font()
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

    # Construct the main window; a failure here (e.g. corrupt QSettings during
    # load_parameters) must be logged rather than producing a bare traceback.
    try:
        window = MainWindow()
    except Exception:
        logger.critical("Failed to construct main window", exc_info=True)
        raise

    # Install the global handler now that we have a window to clean up through.
    _install_excepthook(window)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()