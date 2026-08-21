"""
Module for managing the pattern two group box.
"""
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from script_modules.app_styles import AppStyles
from script_modules.rtm_plot_widget import RTMPlotWidget
from script_modules.mean_pixel_plot_widget import PixelCurrentPlotWidget
from script_modules.pattern_match_plot_widget import PatternMatchPlotWidget


logger = logging.getLogger(__name__)


class PatternTwoGroupBox(QGroupBox):

    # Signals
    crop_changed = Signal(int, int, int, int)
    threshold_changed = Signal(float)   # Forward threshold changes from match score plot widget

    def __init__(self, parent=None, initial_threshold=0.7, threshold_min=0.0, threshold_max=1.0):
        super().__init__(parent)
        # Set title
        self.setTitle("Pattern 2")
        # Set style
        self.setStyleSheet(AppStyles.GroupBox.with_title())
        # Set focus policy
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Set minimum width
        self.setMinimumWidth(150)

        # Store threshold parameters
        self._initial_threshold = initial_threshold
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max

        # Create components
        self._create_components()
        # Create connections
        self._setup_connections()
        # Setup layout
        self._setup_layout()
    
    def _create_components(self):
        # RTM plot
        self.rtm_plot = RTMPlotWidget(parent=self, min_height=200)
        self.rtm_plot.setObjectName("Pattern 2 RTM Plot")
        # Mean pixel value / specimen current plot
        self.mean_pixel_plot = PixelCurrentPlotWidget(parent=self, min_height=200)
        self.mean_pixel_plot.setObjectName("Pattern 2 Pixel Current Plot")
        # Pattern match score plot
        self.match_score_plot = PatternMatchPlotWidget(
            parent=self, 
            min_height=200,
            initial_threshold=self._initial_threshold,
            threshold_min=self._threshold_min,
            threshold_max=self._threshold_max
            )
        self.match_score_plot.setObjectName("Pattern 2 Match Score Plot")

    def _setup_connections(self):
        """Setup signal connections."""
        # RTM crop rectangle changes
        self.rtm_plot.crop_changed.connect(self.on_crop_changed)
        # Pattern match score threshold changes
        self.match_score_plot.threshold_changed.connect(self._on_threshold_changed)

    @Slot(int, int, int, int)
    def on_crop_changed(self, x: int, y: int, width: int, height: int):
        """
        Handle crop rectangle changes from RTM plot.

        :param x: X-coordinate of crop rectangle
        :param y: Y-coordinate of crop rectangle
        :param width: Width of crop rectangle
        :param height: Height of crop rectangle
        """
        sender = self.sender()  # Get the widget that sent the signal
        sender_name = sender.objectName() if sender else "Unknown"
        # DEBUG: widget-level duplicate of the authoritative "crop updated"
        # record logged by MainWindow in model space.
        logger.debug(f"{sender_name}: New crop -> Width: {width}, Height: {height} at ({x}, {y})")
        self.crop_changed.emit(x, y, width, height)
    
    @Slot(float)
    def _on_threshold_changed(self, threshold: float):
        """Handle threshold changes from the match score plot widget."""
        sender = self.sender()
        sender_name = sender.objectName() if sender else "Unknown"
        # logger.info(f"{sender_name}: Threshold changed to {threshold}")
        
        # Forward signal to main window
        self.threshold_changed.emit(threshold)
    
    @Slot(float)
    def set_match_score_threshold(self, threshold: float):
        """Set the match score threshold from the controls spinbox."""
        self.match_score_plot.set_threshold(threshold)
    
    @Slot(float, float)
    def set_match_score_threshold_bounds(self, min_value: float, max_value: float):
        """
        Update threshold bounds if spinbox bounds change.
        
        :param min_value: New minimum threshold value
        :param max_value: New maximum threshold value
        """
        self._threshold_min = min_value
        self._threshold_max = max_value
        self.match_score_plot.set_threshold_bounds(min_value, max_value)

    def _setup_layout(self):
        """Setup the layout with proper spacing."""
        layout = QVBoxLayout()

        # Small margins around the container
        # layout.setContentsMargins(4, 4, 4, 4)

        # Spacing between plots
        layout.setSpacing(10)

        # Add plots to layout
        layout.addWidget(self.rtm_plot)
        layout.addWidget(self.mean_pixel_plot)
        layout.addWidget(self.match_score_plot)
        self.setLayout(layout)

    @Slot(bool)
    def set_plot_visibility(self, visible: bool):
        """
        Set the visibility of all plots in the group box.

        :param visible: Show/hide all plots.
        """
        plots = [self.rtm_plot, self.mean_pixel_plot, self.match_score_plot]
        for plot in plots:
            plot.setVisible(visible)