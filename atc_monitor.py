"""
ATC Monitor
.
This UI is designed to analyze real-time monitor (RTM) data generated from a Thermo Scientific FIB-SEM microscope.
Specifically, it processes RTM images while Thermo Scientific AutoTEM Cryo performs rough milling with one or two rectangle patterns.
The script supports rectangle patterns only (regular cross-section and cleaning cross-section patterns are rejected during validation).
.
The application runs in the background while FIB patterns are generating RTM data. Based on analysis of the images, the application
will stop FIB patterning if certain criteria are met.
.
Thermo Scientific AutoScript 4.13+ is required, and the application uses only modules included with AutoScript (no extra dependencies).
Script created with the assistance of Claude Code.
.
If you have any questions or suggestions for improvements, please contact me (Chris Thompson on GitHub: ChrisLeeThompson).
.
Thank you,
Chris Thompson
.
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

import atexit
import gc
import logging
import os
import queue
import sys
import platform
import threading
import time
import faulthandler
import weakref

import shiboken6
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QWidget,
    QHBoxLayout, QSizePolicy, QLabel
)
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Slot, Signal, QRect, QThread, QTimer, Qt, qInstallMessageHandler, QtMsgType
from script_modules import app_paths
from script_modules import crash_breadcrumbs
from script_modules.gui_watchdog import GuiWatchdog
from script_modules.app_styles import AppStyles, APP_VERSION
from script_modules.monitoring_icon import CatbugMonitoringIcon
from script_modules.pattern_one_group_box import PatternOneGroupBox
from script_modules.pattern_two_group_box import PatternTwoGroupBox
from script_modules.controls_group_box import ControlsGroupBox
from script_modules.pattern_results_group_box import PatternResultsGroupBox
from script_modules.button_widgets import StartButton, StopButton, PauseResumeButton
from script_modules.status_bar_widget import StatusBarWidget
from script_modules.atc_monitor_parameters import (
    load_parameters, RTMMode
)
from script_modules.workflow_worker import WorkflowWorker, WorkerParameters
from script_modules.settings_dialog import SettingsDialog


logger = logging.getLogger(__name__)

# Held module-global for the process lifetime so the file object backing
# faulthandler is NEVER garbage-collected/closed. faulthandler writes to the
# raw file descriptor at crash time; a closed fd would mean the native-fault
# traceback is silently lost -- the exact failure this is meant to capture.
_FAULT_FP = None

# The QueueListener that owns the log sink handlers, kept module-global so
# tests can stop/flush it deterministically (mirrors the _FAULT_FP pattern).
_LOG_LISTENER = None


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

        # GUI responsiveness watchdog (created/started in main(); None when
        # the crash fd is unavailable).
        self._gui_watchdog = None

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
        # Read back the spinbox's QUANTIZED value and use it everywhere: the
        # raw drag position carries more decimals than the spinbox displays or
        # persists, and gating the worker at an invisible value means the same
        # displayed configuration behaves differently after a restart.
        threshold = spinbox.value()
        spinbox.blockSignals(False)
        # Update central parameters
        self.params.ui.match_score_threshold = threshold
        # Snap BOTH plots' lines to the quantized value (the source line sits
        # at the raw drag position otherwise; set_* does not re-emit).
        other_idx = 1 - source_idx
        self.pattern_group_boxes[other_idx].set_match_score_threshold(threshold)
        self.pattern_group_boxes[source_idx].set_match_score_threshold(threshold)
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
            "Maximum Foreground Energy" if is_energy else "Maximum Pixels Threshold"
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
            crash_breadcrumbs.drop("settings-saved")

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

        # Run begins -> catbug message shows "Monitoring (waiting for patterning)..." (gray until first detection)
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

        crash_breadcrumbs.drop("start-clicked")
        logger.info("Start button clicked - worker thread started")

    def _reap_worker(self):
        """
        Destroy a previously finished worker/thread deterministically.

        The 2026-08-18 on-tool abort traced to GC-timed destruction: the old
        worker/QThread wrappers survived the plain ref-drop via PySide's cached
        SignalInstance self-cycle, so their Qt C++ destructors ran inside a
        later cyclic-GC pass -- on the main thread, mid-signal-dispatch, on a
        QObject whose affinity thread was dead. Qt object lifetime must never
        depend on GC timing: disconnect everything and delete the C++ objects
        explicitly, here, where no signal delivery is on the stack.
        """
        if self.worker is None and self.worker_thread is None:
            return

        # Never destroy a running thread's objects. Dropping the refs after a
        # timed-out wait would run ~QThread on a running thread -> qFatal ->
        # abort on the main thread. Keep the refs instead; the on_start_clicked
        # isRunning guard keeps blocking the next Start.
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(5000):
                logger.error(
                    "Reap refused: worker thread still running after quit+5s "
                    "wait; keeping references (no destruction of a live thread)"
                )
                return

        if self.worker is not None:
            # Drop the GUI->worker control connections BEFORE destroying: their
            # sender (this window) outlives the worker, so without this they
            # accumulate per Start and would deliver into a dead worker's
            # slots. Worker-as-sender connections need no explicit disconnect;
            # the shiboken6.delete below destroys them with the C++ object
            # (PySide 6.7.1 has no blanket QObject.disconnect() anyway).
            for sig, slot in (
                (self.request_worker_stop, self.worker.request_stop),
                (self.request_worker_pause, self.worker.request_pause),
                (self.request_worker_resume, self.worker.request_resume),
                (self.request_worker_param, self.worker.update_parameter),
            ):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass

            # Field probe (2026-08-18 crash follow-up): record whether the
            # wrapper carries cached SignalInstance attrs and whether anything
            # keeps the wrapper shells alive past this reap. One on-tool
            # session's log settles whether GC-deferred destruction (the
            # crash hypothesis) actually occurs there; not reproducible on
            # the dev box.
            sig_keys = [k for k, v in self.worker.__dict__.items()
                        if type(v).__name__ == "SignalInstance"]
            logger.info(f"Reap: cached SignalInstance attrs on worker: {sig_keys}")
            wr = weakref.ref(self.worker)
            wt = (weakref.ref(self.worker_thread)
                  if self.worker_thread is not None else None)
            QTimer.singleShot(0, lambda: logger.info(
                "Post-reap wrappers alive: worker=%s, thread=%s" % (
                    wr() is not None,
                    "n/a" if wt is None else (wt() is not None),
                )
            ))

            # Deterministic C++ destruction, now, on the main thread (the
            # worker re-homed its affinity here at the end of run()).
            # deleteLater is not used: it would leave a window in which a
            # cyclic wrapper's GC dealloc deletes the C++ object anyway.
            if shiboken6.isValid(self.worker):
                shiboken6.delete(self.worker)
        if self.worker_thread is not None and shiboken6.isValid(self.worker_thread):
            shiboken6.delete(self.worker_thread)

        self.worker = None
        self.worker_thread = None
        crash_breadcrumbs.drop("reaped")
        # Sweep the now destructor-free wrapper shells (and any other cyclic
        # garbage) at a known-idle main-loop point instead of inside an
        # arbitrary allocation.
        QTimer.singleShot(0, gc.collect)

    def _check_worker_teardown(self, thread, attempt: int = 1):
        """
        Crash-diagnosis instrumentation (2026-08-14 heap-corruption follow-up):
        the on-tool log showed the 'Post-connect: N live threads' count growing
        2 -> 6 across Start/Stop cycles. That is EITHER worker threads genuinely
        outliving Stop (a real leak, and a prime suspect for the 0xc0000374
        heap corruption) OR stale threading._DummyThread bookkeeping for
        already-exited Qt threads (CPython cannot observe a foreign thread's
        death). This check runs shortly after each stop and logs the QThread's
        OWN state, which is authoritative -- the next field session's log
        settles the question. When the thread has genuinely finished, its
        objects are reaped immediately instead of at the next Start.

        ``thread`` is bound at schedule time so a quick Start of a NEW run
        cannot be mistaken for the old thread failing to exit.
        """
        if thread is None or thread is not self.worker_thread:
            return  # a new run has replaced it; the old refs are already gone
        running = thread.isRunning()
        finished = thread.isFinished()
        py_threads = sorted(t.name for t in threading.enumerate())
        logger.info(
            f"Worker teardown check #{attempt}: qthread.isRunning={running}, "
            f"qthread.isFinished={finished}, python-visible threads={py_threads}"
        )
        if finished:
            self._reap_worker()
        elif attempt < 3:
            QTimer.singleShot(
                700, lambda: self._check_worker_teardown(thread, attempt + 1)
            )
        else:
            logger.warning(
                "Worker thread still running ~2s after monitoring stopped -- "
                "possible zombie worker (crash-diagnosis evidence; see "
                "_check_worker_teardown)"
            )
    
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
            stall_window=self.params.processing.stall_window,
            stall_drop_fraction=self.params.processing.stall_drop_fraction,
            stall_rel_tolerance=self.params.processing.stall_rel_tolerance,
            stall_abs_tolerance=self.params.processing.stall_abs_tolerance,

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
        
        # Status Signals. _on_status_update is a real @Slot so this is a native
        # meta-connection: a lambda here would be the app's only Python-callable
        # (GlobalReceiver) connection -- the machinery on the 2026-08-18 crash
        # stack.
        self.worker.status_update.connect(self._on_status_update)
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
            text = AppStyles.AppText.WINDOW_INFO_MONITORING_ACTIVE  # "Monitoring (patterning detected)..."
        else:
            text = AppStyles.AppText.WINDOW_INFO_MONITORING_IDLE    # "Monitoring (waiting for patterning)..."
        self.information_label_2.setText(text)

    @Slot(bool)
    def _on_monitoring_icon_changed(self, active: bool):
        """Catbug flipped color/grayscale -> refresh the status message."""
        self._monitoring_detecting = active
        self._update_monitoring_message()

    @Slot()
    def _on_watchdog_heartbeat(self):
        """1 s GUI heartbeat -> GuiWatchdog (native slot per house rule)."""
        if self._gui_watchdog is not None:
            self._gui_watchdog.beat()

    @Slot()
    def _stop_gui_watchdog(self):
        """aboutToQuit: stop the watchdog thread (daemon exit is safe anyway)."""
        if self._gui_watchdog is not None:
            self._gui_watchdog.stop()

    # ========================================
    # Worker Signal Handlers - Control
    # ========================================

    @Slot(str)
    def _on_status_update(self, text: str):
        """Timed status-bar text from the worker (native-slot connection)."""
        self.status_bar.set_timed_status_text(text, 5)

    @Slot()
    def on_monitoring_started(self):
        """Handle monitoring_started signal from worker."""
        # First queued worker->GUI delivery of a run: its breadcrumb missing
        # after a "start-clicked" is the exact signature of the 2026-08-19
        # main-thread freeze (frozen before the first delivery).
        crash_breadcrumbs.drop("monitoring-confirmed")
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

        # Diagnose (and, once finished, reap) the worker thread shortly after
        # the stop instead of waiting for the next Start -- see
        # _check_worker_teardown for the crash-evidence rationale.
        QTimer.singleShot(
            700, lambda t=self.worker_thread: self._check_worker_teardown(t)
        )
    
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
        # Catbug message -> back to "Monitoring (waiting for patterning)..." / "Monitoring (patterning detected)..." per icon
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
                # No foreground snapshot: the live plot was removed in v3.3.5;
                # the full per-batch foreground/stall trace persists in
                # metrics.csv (white_pixel_percentage, smoothed_foreground,
                # running_peak, stall_latched).

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
        title = "Top-Hat Foreground Map" if is_energy else "Binary RTM Image"
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
        - pixels_via ("ABS"/"STALL"/None) -- which path satisfied the
          foreground criterion on THIS round; drives the live "(stall)"
          annotation
        - stall_in_streak (bool) -- sticky: True if ANY round of the current
          confirmation streak passed via the stall latch; makes the
          annotation durable on the frozen completion panel
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
            results_widget.set_match_score_result(
                "--" if match_score is None else f"{match_score:.4f}", pattern_idx
            )
            # Annotate the foreground criterion while it is passing via the
            # stall latch (the trace has floored above the absolute threshold),
            # so the operator can see WHICH path is satisfying it - shown from
            # the first stall-latched round, not just at completion. On the
            # completion round the sticky streak flag takes over: the panel
            # freezes with THIS text, and a streak can mix ABS and STALL
            # rounds, so keying the final render on pixels_via alone would
            # leave the durable panel unannotated whenever the arbitrary last
            # round happened to pass via ABS (UI review 2026-08-18) - the
            # 5-second status message below is not a durable surface.
            via_stall = (results.get('pixels_via') == "STALL"
                         or (results.get('is_complete')
                             and results.get('stall_in_streak')))
            # "(stall)" renders on a SECOND line: the results panel reserves
            # the two-line cell size at construction, so the annotation costs
            # no window width (the right column absorbs the height in its
            # existing stretch) and neither dimension moves mid-run.
            results_widget.set_percent_pixels_result(
                f"{white_pixels:.2f}\n(stall)" if via_stall
                else f"{white_pixels:.2f}",
                pattern_idx
            )

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
                match_ok = (match_score is not None
                            and match_score <= results['match_score_threshold'])
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
            # A stall-flagged completion means the stall latch was load-bearing
            # on at least one round of the confirmation streak (the trace
            # floored at or above the absolute threshold, e.g. an exposed grid
            # bar) - say so explicitly. stall_in_streak is sticky across the
            # streak, unlike pixels_via which reflects only the final round (a
            # streak can mix ABS and STALL rounds when the raw value straddles
            # the threshold). This block runs once per completion (the panel
            # freezes right below), so the status fires exactly once.
            if (results.get('is_complete')
                    and results.get('stall_in_streak')):
                self.status_bar.set_timed_status_text(
                    f"Pattern {pattern_idx + 1}: Complete (foreground stalled)", 5
                )
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
        # Breadcrumb first: a WATCHDOG dump during the shutdown wait below is
        # then self-attributing (hang-on-close vs hang-mid-run).
        crash_breadcrumbs.drop("close-event")
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

        Returns the directory containing the main script file (also the root
        of the worker's cb_plant.json -- script_root is passed into
        WorkflowWorker at Start). Single source:
        script_modules.app_paths.
        """
        return app_paths.script_root()


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
            # The logger call above is only an ENQUEUE now (QueueHandler):
            # Qt aborts the process the moment this handler returns, atexit
            # never runs, and the listener thread may never drain the queue --
            # so the fatal reason would be racy-lost, the exact blindness this
            # handler exists to prevent. Write it synchronously through the
            # kept-open raw faulthandler fd, which is the designated
            # crash-time record.
            if _FAULT_FP is not None:
                try:
                    _FAULT_FP.write(f"Qt FATAL: {message}\n")
                    _FAULT_FP.flush()
                except Exception:
                    pass
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error(f"Qt CRITICAL: {message}")
            # Criticals often immediately precede an abort, and the queued-
            # logger copy is exactly what is lost when the listener dies with
            # the process (2026-08-18: the last enqueued records never reached
            # the app log). Mirror them synchronously like fatals.
            if _FAULT_FP is not None:
                try:
                    _FAULT_FP.write(f"Qt CRITICAL: {message}\n")
                    _FAULT_FP.flush()
                except Exception:
                    pass
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning(f"Qt WARNING: {message}")
        else:
            logger.info(f"Qt: {message}")
    except Exception:
        pass


def _gc_marker(phase, info):
    """
    Crash forensics (2026-08-18 abort follow-up): an unmatched "GC start gen=2"
    line adjacent to a faulthandler dump proves the abort ran inside a full
    cyclic collection (the GC-timed Qt-destructor hypothesis); its absence at
    the next crash refutes that. Gen-2 collections are rare, so two buffered
    local-disk writes per collection are negligible. Keep this allocation-light
    and never raise -- it runs inside the collector.
    """
    try:
        if info.get("generation") == 2 and _FAULT_FP is not None:
            _FAULT_FP.write(
                "GC start gen=2\n" if phase == "start" else "GC stop gen=2\n"
            )
            _FAULT_FP.flush()
    except Exception:
        pass


def _install_unraisable_hook():
    """
    Mirror unraisable errors (destructor/__del__ exceptions -- the kind raised
    during GC-time finalization) synchronously into the crash record, then
    chain to the previous hook. A future abort preceded by an UNRAISABLE line
    names the exact object being finalized.
    """
    prev = sys.unraisablehook

    def _unraisable_to_fault_fd(unraisable):
        if _FAULT_FP is not None:
            try:
                _FAULT_FP.write(
                    f"UNRAISABLE: {unraisable.exc_value!r} "
                    f"in {unraisable.object!r}\n"
                )
                _FAULT_FP.flush()
            except Exception:
                pass
        prev(unraisable)

    sys.unraisablehook = _unraisable_to_fault_fd


def _qt_version_safe() -> str:
    """Return the runtime Qt version for the launch banner, or 'unknown'."""
    try:
        from PySide6.QtCore import qVersion
        return qVersion()
    except Exception:
        return "unknown"


def _local_logs_dir() -> Path:
    """
    Local-disk directory (%LOCALAPPDATA%/ATC_Monitor/logs). Two roles:

    1. Home of faulthandler.log, ALWAYS. It is written through a raw kept-open
       fd at crash time; a hung SMB write at that moment would hang the crash
       dump itself, so the crash-time record must never live on the network
       share. Non-trivial content is swept into <script dir>/logs/
       faulthandler.log at the next startup (see _mirror_fault_log).
    2. Fallback for the rotating app log when <script dir>/logs (the primary
       since v3.3.2, see _script_logs_dir) cannot be created or written.

    History: pre-3.3.2 the rotating app log lived here too, as part of the fix
    for the 2026-08-14 heap-corruption crash (0xc0000374) that died in
    worker-thread RotatingFileHandler.shouldRollover -> os.path.exists on the
    UNC share. The queue indirection (a single listener thread owns ALL file
    I/O, see setup_logging) is what actually removes that stack from worker
    threads, so the app log could move back beside the script; local disk
    remains the crash-time and fallback location.

    Single source: script_modules.app_paths.
    """
    return app_paths.local_logs_dir()


# Cap on how much crash residue one startup sweep copies to the script dir; a
# fault log this large means many unswept crashes -- the newest tail is the
# diagnostic that matters, and an unbounded SMB write at startup is not.
_FAULT_MIRROR_MAX_BYTES = 1_000_000

# Cap on the script-dir mirror FILE itself -- the one append-only log with no
# rotation. Since the v3.3.6 breadcrumbs every session appends a trail (a few
# KB), not just crashes, so without this the mirror grows forever.
_FAULT_MIRROR_FILE_CAP_BYTES = 5_000_000


def _trim_fault_mirror(mirror_path):
    """
    Keep the fault-log mirror bounded by ROTATING it (rename to .1) when it
    outgrows the cap; the caller's append then recreates a fresh file. A
    rotation never destroys bytes -- the mirror is shared by every machine
    launched from one deployment, and an in-place read-trim-rewrite could
    silently discard a sweep another host appended mid-rewrite (the exact
    loss mode the local-file sweep is engineered against). os.replace is
    atomic; if another host holds the file open the rename fails and the
    file is simply left over-cap until a quieter launch (an over-cap mirror
    beats a lost sweep). Retention: newest cycle in faulthandler.log,
    previous cycle in faulthandler.log.1.
    """
    try:
        if (not mirror_path.exists()
                or mirror_path.stat().st_size <= _FAULT_MIRROR_FILE_CAP_BYTES):
            return
        mirror_path.replace(mirror_path.with_name(mirror_path.name + ".1"))
    except Exception:
        logger.warning("Fault-log mirror rotation failed; appending anyway",
                       exc_info=True)


def _fault_log_has_content(text: str) -> bool:
    """
    True iff the fault log holds anything beyond armed banners and blank
    lines -- i.e. actual crash residue worth mirroring. Every launch appends
    one "==== faulthandler armed pid=N ====" banner (see setup_logging), so a
    file of nothing but banners means the prior runs exited cleanly.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (stripped.startswith("==== faulthandler armed pid=")
                and stripped.endswith("====")):
            continue
        return True
    return False


def _mirror_fault_log(local_logs, script_logs) -> None:
    """
    One-shot startup sweep of crash residue from the LOCAL faulthandler.log
    into <script dir>/logs/faulthandler.log, putting native-crash evidence
    beside the app log that gets read in the field. Crash-time writes stay
    local (see _local_logs_dir); only this after-the-fact copy touches the
    share, at startup, where a slow SMB write is harmless.

    Must run BEFORE the faulthandler-open block in setup_logging: a
    successful mirror truncates the local file, and that has to happen before
    _FAULT_FP reopens it for append and writes this run's armed banner.

    Never raises -- a failed mirror must not abort startup, and on any mirror
    failure the local copy is left untouched so the residue survives for the
    next attempt. The reverse failure (mirrored but could not truncate) is a
    warning only and keeps the mirror: a duplicated block on the next sweep
    beats a lost one (duplicate-tolerant by design).

    Only the bytes actually read are removed afterwards, never the whole file:
    a second instance may be running with this file open for append, and its
    crash dump must survive this sweep.
    """
    try:
        if local_logs is None:
            return
        local_path = local_logs / "faulthandler.log"
        if not local_path.exists():
            return
        try:
            swept = local_path.read_text(encoding="utf-8", errors="replace")
            text = swept
            if not _fault_log_has_content(text):
                # Banner-only residue (~45 bytes per launch): nothing to say,
                # so write nothing -- and do NOT truncate, the banners are
                # the per-launch arming record.
                return
            if script_logs is None:
                logger.warning(
                    "Fault-log residue in %s could not be mirrored: no "
                    "script-dir logs location", local_path)
                return
            truncated = len(text) > _FAULT_MIRROR_MAX_BYTES
            if truncated:
                text = text[-_FAULT_MIRROR_MAX_BYTES:]
            script_logs.mkdir(parents=True, exist_ok=True)
            _trim_fault_mirror(script_logs / "faulthandler.log")
            # host= and app= for the same reason as the LAUNCH banner: several
            # machines launched from one share directory append to one file.
            header = (
                f"\n==== mirrored from {local_path} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"host={platform.node()} app={APP_VERSION}"
                + (f" (truncated to last {_FAULT_MIRROR_MAX_BYTES} bytes)"
                   if truncated else "")
                + " ====\n"
            )
            with open(script_logs / "faulthandler.log", "a",
                      encoding="utf-8") as mirror_fp:
                mirror_fp.write(header + text)
                mirror_fp.flush()
        except Exception:
            logger.warning(
                "Fault-log mirror to script dir failed; local copy kept: %s",
                local_path, exc_info=True)
            return
        # Reset the local file ONLY now that the mirror fully succeeded, and
        # drop ONLY the bytes actually mirrored: another instance of the app
        # can be running with this file open for append (it is shared per
        # account), and a blind truncate would silently destroy a crash dump it
        # wrote while the SMB append above was in flight -- the single
        # highest-value diagnostic. Anything appended since is carried forward
        # for the next sweep instead.
        try:
            with open(local_path, "r+", encoding="utf-8", errors="replace") as fp:
                current = fp.read()
                remainder = (current[len(swept):]
                             if current.startswith(swept) else current)
                fp.seek(0)
                fp.write(remainder)
                fp.truncate()
        except Exception:
            logger.warning(
                "Mirrored fault log but could not reset local copy: %s",
                local_path, exc_info=True)
        logger.info("Mirrored %d bytes of fault-log residue to %s",
                    len(text), script_logs / "faulthandler.log")
    except Exception:
        logger.warning("Fault-log mirror failed", exc_info=True)


def _script_logs_dir() -> Path:
    """
    Primary app-log directory: <script dir>/logs (usually on the UNC share the
    app is launched from). Safe for the rotating app log since v3.3.1 because
    only the single QueueListener thread ever touches the file -- worker
    threads only enqueue, so the 2026-08-14 crash stack (worker-thread SMB
    I/O) cannot recur. Matches the cryo-utilities convention of file output
    beside the script and puts the log next to the Saved_Data folder.

    Single source: script_modules.app_paths (same sys.argv[0]-based root as
    MainWindow._get_script_root, used for saved data and the learned CB plant
    file).
    """
    return app_paths.script_logs_dir()


# ---- Console policy --------------------------------------------------------
# Applies to the stderr sink ONLY; the rotating file log always receives full
# INFO telemetry (Auto CB traces, match scores -- the field-diagnosis record).
#   quiet console:    _CONSOLE_LEVEL = logging.WARNING
#   key events only:  _CONSOLE_LEVEL = logging.INFO (default, with noise filter)
#   silent console:   _CONSOLE_LEVEL = logging.ERROR
_CONSOLE_LEVEL = logging.INFO
_CONSOLE_FMT = "%(asctime)s %(levelname)-8s %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"


class _ConsoleNoiseFilter(logging.Filter):
    """
    Console-side quieting only: attached to the stderr StreamHandler, never to
    the file handler or the root logger, so every category below still reaches
    the file log at INFO (needed for field diagnosis).

    Blocklist semantics -- allow everything except known-noisy message
    families, matched by substring (several carry a variable "Pattern N: "
    prefix, so startswith would miss them). Fails open, and WARNING and above
    ALWAYS pass, so the filter can never hide a warning/error.
    """

    NOISY_SUBSTRINGS = (
        "sparse-fallback:",            # per-tile score dumps (image_pattern_matching)
        "Microscope data read:",       # telemetry dict dump (workflow_worker)
        "Updated global parameters:",  # per-tick params echo (atc_monitor)
        "crop updated:",               # per-drag crop echo (atc_monitor)
        "Scan direction set to",       # RTM overlay update (rtm_plot_widget)
        "Auto CB:",                    # calibration step notes (full trace in file)
        "Qt: ",                        # Qt info catch-all (_qt_message_handler)
        "Worker teardown check",       # crash-diagnosis residue (atc_monitor)
        "Post-connect:",               # thread-list dump (fib_patterning_monitor)
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.levelno >= logging.WARNING:
                return True
            msg = record.getMessage()
            return not any(s in msg for s in self.NOISY_SUBSTRINGS)
        except Exception:
            return True


def setup_logging():
    """
    Configure logging: a concise stderr console PLUS a durable rotating file
    log (in <script dir>/logs, falling back to %LOCALAPPDATA%/ATC_Monitor/logs,
    fed through a queue so no worker thread ever performs file I/O) and
    native-fault capture to a LOCAL file. Crash residue a previous run left
    in that local file is swept into <script dir>/logs here at startup, so
    all field-readable evidence ends up in one place (see _mirror_fault_log).

    Why the file outputs matter: this is a windowed GUI app, so when it is
    launched without an attached console, stderr goes nowhere visible. A native
    crash (e.g. a fault inside the AutoScript C transport) then closes the window
    with "nothing in the console" and no record at all. Writing the app log and
    the faulthandler traceback to files makes such a crash leave durable
    on-disk evidence instead of dying silently.

    Why the queue: emitting a record must never do file (SMB!) I/O on the
    calling thread. A QueueHandler makes every logger call a lock-free enqueue;
    a single QueueListener thread owns the actual stderr + rotating-file
    handlers -- which is what makes a rotating log on the launch share safe.
    Trade-offs (accepted): records still in the queue at a HARD crash are lost
    -- the faulthandler file, written through a raw kept-open fd on LOCAL
    disk, is the crash-time record. An SMB stall pauses only the listener
    thread while records buffer in the unbounded in-memory queue; do NOT
    bound the queue -- a blocking put would reintroduce worker-thread stalls.
    """
    global _FAULT_FP, _LOG_LISTENER

    # Resolve both candidate directories defensively -- logging setup must
    # never crash the app, whatever sys.argv[0] or the environment look like.
    try:
        script_logs = _script_logs_dir()
    except Exception:
        script_logs = None
    try:
        local_logs = _local_logs_dir()
    except Exception:
        local_logs = None

    # process id + thread name in every record so two separate-process runs
    # (launch 1 vs launch 2 -- possibly on different machines now that the log
    # can live on a shared script directory; see host= in the banner) and GUI
    # vs worker thread are attributable in one appended file -- the crux of
    # diagnosing the intermittent first-Start crash.
    fmt = "%(asctime)s\t%(process)d\t%(threadName)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s"

    # Console (stderr) sink: concise format, own level + noise filter.
    # Console policy never affects the file sink.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_CONSOLE_LEVEL)
    console_handler.addFilter(_ConsoleNoiseFilter())
    console_handler.setFormatter(
        logging.Formatter(_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT))

    sink_handlers: list[logging.Handler] = [console_handler]
    app_log_path = None
    # delay=False on purpose: the file open happens HERE, once, at startup --
    # an unreachable share fails fast and falls through to the local fallback,
    # instead of failing later on the listener thread mid-run.
    for candidate_dir in (script_logs, local_logs):
        if candidate_dir is None:
            continue
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            # Rotating so the log cannot grow unbounded (~25 MB cap).
            file_handler = RotatingFileHandler(
                candidate_dir / "atc_monitor.log",
                maxBytes=5_000_000, backupCount=5, encoding="utf-8", delay=False,
            )
            file_handler.setFormatter(logging.Formatter(fmt))
            sink_handlers.append(file_handler)
            app_log_path = candidate_dir / "atc_monitor.log"
            break
        except Exception:
            continue  # try the next candidate; stderr-only if all fail

    try:
        log_queue = queue.SimpleQueue()
        listener = QueueListener(log_queue, *sink_handlers,
                                 respect_handler_level=True)
        listener.start()
        atexit.register(listener.stop)
        _LOG_LISTENER = listener
        root_handlers = [QueueHandler(log_queue)]
    except Exception:
        # Fall back to direct handlers rather than lose logging entirely.
        root_handlers = sink_handlers

    logging.basicConfig(
        format=fmt,
        level=logging.INFO,
        force=True,
        handlers=root_handlers,
    )
    # basicConfig stamps `fmt` onto the handlers it is given -- but a
    # QueueHandler must NOT carry the full formatter: its prepare() bakes the
    # formatted text into record.msg BEFORE the listener's sinks format the
    # record again (observed as double-prefixed lines). Message-only here;
    # the sinks own the real format. (Exception text is still serialized into
    # the message by prepare(), which is exactly what we want off-thread.)
    for handler in root_handlers:
        if isinstance(handler, QueueHandler):
            handler.setFormatter(logging.Formatter("%(message)s"))

    # faulthandler ALWAYS lives on local disk: it writes through the raw fd at
    # crash time, and a hung SMB write at that moment would hang the crash
    # dump itself. Residue is mirrored to the script dir at next startup
    # (_mirror_fault_log below).
    fault_dir = local_logs
    if fault_dir is not None:
        try:
            fault_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            fault_dir = None

    # Launch banner -- first record of each run, so runs in the appended file
    # can be told apart and the environment (Python/Qt/OS) is captured for
    # triage. host= because two machines launched from the same share directory
    # append to the same file (pid alone no longer disambiguates).
    logger.info(
        "LAUNCH app=%s pid=%s host=%s exe=%s py=%s qt=%s platform=%s",
        APP_VERSION, os.getpid(), platform.node(), sys.executable,
        sys.version.replace("\n", " "), _qt_version_safe(), platform.platform(),
    )
    logger.info("App log file: %s",
                app_log_path if app_log_path is not None else "(stderr only)")
    logger.info("Fault log file: %s",
                (fault_dir / "faulthandler.log") if fault_dir is not None
                else "(stderr)")
    if app_log_path is None:
        logger.warning("No writable log directory (script dir or local app "
                       "data); file logging disabled")
    elif script_logs is None or app_log_path.parent != script_logs:
        logger.warning("Script-directory logs unavailable; using local "
                       "fallback: %s", app_log_path)

    # Capture Qt's own fatal/critical messages (names the exact qFatal reason if
    # a Qt assertion ever aborts the process).
    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        logger.warning("Could not install Qt message handler", exc_info=True)

    # Sweep the previous run's crash residue into the script directory BEFORE
    # arming faulthandler: a successful sweep truncates the local file, which
    # must happen before _FAULT_FP reopens it for append and writes this
    # run's armed banner. Ordering is load-bearing.
    _mirror_fault_log(fault_dir, script_logs)

    # Dump a Python+C traceback (all threads) if the process is killed by a native
    # fault (e.g. inside the AutoScript C transport), which a normal try/except
    # cannot catch. Direct it to a SEPARATE, append-mode, kept-open file (NOT the
    # logging handler's file): faulthandler writes through the raw fd at crash
    # time and must not interleave with the logging buffer. Append so launch 2
    # never overwrites launch 1's fault. This is the single highest-value
    # diagnostic -- it converts a silent native death into a stack trace on disk.
    try:
        if fault_dir is not None:
            _FAULT_FP = open(
                fault_dir / "faulthandler.log", "a", buffering=1, encoding="utf-8"
            )
            _FAULT_FP.write(f"\n==== faulthandler armed pid={os.getpid()} ====\n")
            _FAULT_FP.flush()
            faulthandler.enable(file=_FAULT_FP, all_threads=True)
            # Run-lifecycle breadcrumbs share the crash fd: the queued app log
            # loses its final records on a hard crash (2026-08-19 left a
            # 1-37 s blind spot), the synchronous breadcrumb trail does not.
            crash_breadcrumbs.set_fp(_FAULT_FP)
        else:
            faulthandler.enable(all_threads=True)  # fall back to stderr
    except Exception:
        logger.warning("Could not enable faulthandler-to-file", exc_info=True)
        try:
            faulthandler.enable(all_threads=True)  # last-resort fallback
        except Exception:
            pass

    # Crash forensics companions to faulthandler (2026-08-18 abort follow-up):
    # mark full GC collections and unraisable finalizer errors in the same
    # synchronous crash record. Both are no-ops on the happy path.
    try:
        gc.callbacks.append(_gc_marker)
    except Exception:
        logger.warning("Could not register GC crash marker", exc_info=True)
    try:
        _install_unraisable_hook()
    except Exception:
        logger.warning("Could not install unraisable-error hook", exc_info=True)


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

    # GUI responsiveness watchdog (2026-08-19 hang-then-kill follow-up): a 1 s
    # heartbeat on the GUI thread + a daemon poller that writes an all-thread
    # stack dump to the crash fd if the heartbeat stalls >10 s -- the only
    # instrument that survives a TerminateProcess kill of a ghosted window.
    # Gated on the crash fd: a watchdog dumping to a windowed app's stderr is
    # worthless. Started here, NOT in setup_logging (tests re-run that against
    # temp dirs and close _FAULT_FP in tearDown).
    if _FAULT_FP is not None:
        window._gui_watchdog = GuiWatchdog(_FAULT_FP)
        window._heartbeat_timer = QTimer(window)
        window._heartbeat_timer.setInterval(1000)
        window._heartbeat_timer.timeout.connect(window._on_watchdog_heartbeat)
        window._heartbeat_timer.start()
        window._gui_watchdog.start()
        app.aboutToQuit.connect(window._stop_gui_watchdog)
    else:
        logger.warning("GUI watchdog disabled: no crash fd available")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()