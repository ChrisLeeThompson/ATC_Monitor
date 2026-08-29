"""
Pattern match score plot widget.
"""
import matplotlib
matplotlib.set_loglevel("WARNING")  # Suppress matplotlib debug messages
import logging
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCursor
from script_modules.base_plot_widget import BasePlotWidget
from script_modules.app_styles import AppStyles


logger = logging.getLogger(__name__)


class PatternMatchPlotWidget(BasePlotWidget):

    # Signals
    threshold_changed = Signal(float)  # Emitted when the threshold line is moved.

    def __init__(self, parent=None, min_height=200, initial_threshold=0.7, 
                 threshold_min=0.0, threshold_max=1.0):
        """
        Initialize the pattern match plot widget.

        :param min_height: Minimum height in pixels for the widget.
        :param initial_threshold: Initial value for the threshold line.
        :param threshold_min: Minimum value for the threshold line.
        :param threshold_max: Maximum value for the threshold line.
        """
        # Store threshold constraints from spinbox
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        # Store threshold value before calling super().__init__()
        self._threshold_value = initial_threshold
        self._dragging_threshold = False
        self._hover_threshold = False

        super().__init__(parent, min_height=min_height)
        
        # Set title
        self.ax.set_title("Match Score", fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        
        # Styling
        self.ax.set_facecolor(AppStyles.Colors.GROUPBOX_BG)
        self.ax.spines["left"].set_visible(True)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color(AppStyles.Colors.TEXT_PRIMARY)
        self.ax.spines["bottom"].set_color(AppStyles.Colors.TEXT_PRIMARY)
        self.ax.tick_params(colors=AppStyles.Colors.TEXT_PRIMARY)
        
        # Store the line object for updates
        self.line = None
        
        # Create the threshold line
        self._create_match_score_threshold_line()

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self._on_match_score_threshold_press)
        self.canvas.mpl_connect("button_release_event", self._on_match_score_threshold_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_match_score_threshold_motion)

        # Draw
        self.canvas.draw()
    
    def _create_match_score_threshold_line(self):
        """Create the interactive threshold line."""
        # Create horizontal dashed line
        self.threshold_line = self.ax.axhline(
            y=self._threshold_value, 
            color=AppStyles.Colors.PLOT_INTERACTIVE_LINE_COLOR, 
            linestyle='--', 
            linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
            picker=True,
            pickradius=5
        )
        logger.debug(f"Created threshold line at y={self._threshold_value} "
                     f"(bounds: {self._threshold_min} to {self._threshold_max})")
    
    def _is_near_match_score_threshold_line(self, event):
        """Check if mouse is near the threshold line."""
        if event.inaxes != self.ax or event.ydata is None:
            return False
        
        # Get the threshold y-value in data coordinates
        threshold_y = self._threshold_value

        # Get the y-axis range to calculate pixel tolerance
        y_min, y_max = self.ax.get_ylim()
        y_range = y_max - y_min

        # Define tolerance as a fraction of the visible range (about 5 pixels worth)
        tolerance = y_range * 0.02  # 2% of visible range
        
        # Check if mouse y is within tolerance of threshold line
        return abs(event.ydata - threshold_y) < tolerance
    
    def _on_match_score_threshold_press(self, event):
        """Handle mouse press on threshold line."""
        if event.button != 1:  # Left click only
            return
        
        if self._is_near_match_score_threshold_line(event):
            self._dragging_threshold = True
            # logger.debug(f"Started dragging threshold line at y={event.ydata}")
    
    def _on_match_score_threshold_release(self, event):
        """Handle mouse release after dragging threshold line."""
        if self._dragging_threshold:
            self._dragging_threshold = False
            logger.info(f"Match score threshold line set to: {self._threshold_value}")
            # Emit final value
            self.threshold_changed.emit(self._threshold_value)
    
    def _on_match_score_threshold_motion(self, event):
        """Handle mouse motion for threshold line interaction."""
        if event.inaxes != self.ax:
            # Reset cursor when leaving plot
            if self._hover_threshold:
                self.canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                self._hover_threshold = False
            return
        
        # Check if dragging
        if self._dragging_threshold and event.ydata is not None:
            # Constrain using spinbox bounds (single source of truth!)
            new_threshold = max(self._threshold_min, 
                              min(self._threshold_max, event.ydata))
            self._update_threshold_line_position(new_threshold)
            
            # Emit value change while dragging (for real-time updates)
            self.threshold_changed.emit(new_threshold)
            
            # logger.debug(f"Dragging threshold to: {new_threshold}")
        
        # Check if hovering (for cursor change)
        elif self._is_near_match_score_threshold_line(event):
            if not self._hover_threshold:
                self.canvas.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
                self._hover_threshold = True
        else:
            if self._hover_threshold:
                self.canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                self._hover_threshold = False
    
    def _update_threshold_line_position(self, new_threshold):
        """Update the position of the threshold line."""
        self._threshold_value = new_threshold
        self.threshold_line.set_ydata([new_threshold, new_threshold])
        self.canvas.draw_idle()
    
    @Slot(float)
    def set_threshold(self, threshold):
        """Set the threshold line position from external source (e.g., spinbox)."""
        # Constrain using bounds
        threshold = max(self._threshold_min, min(self._threshold_max, threshold))
        
        # Only update if value actually changed to avoid unnecessary redraws
        if abs(self._threshold_value - threshold) > 0.0001:
            # logger.info(f"Setting threshold from external source: {threshold}")
            self._update_threshold_line_position(threshold)
    
    @Slot(float, float)
    def set_threshold_bounds(self, min_value, max_value):
        """Update the threshold bounds (e.g., if spinbox bounds change)."""
        self._threshold_min = min_value
        self._threshold_max = max_value
        
        # Clamp current threshold to new bounds
        if self._threshold_value < min_value:
            self.set_threshold(min_value)
        elif self._threshold_value > max_value:
            self.set_threshold(max_value)
        
        # logger.info(f"Threshold bounds updated: {min_value:.4f} to {max_value}")
    
    def get_threshold(self):
        """Get the current threshold value."""
        return self._threshold_value
    
    def plot_data(self, x_data, y_data):
        """
        Plot pattern match score data.
        
        Uses set_data() on subsequent calls to avoid recreating Line2D artists,
        which is significantly cheaper than remove + plot.
        
        :param x_data: X-axis data (e.g., image numbers)
        :param y_data: Y-axis data (pattern match scores)
        """
        if self.line is None:
            # First call - create the line
            self.line, = self.ax.plot(
                x_data, 
                y_data,
                color=AppStyles.Colors.PLOT_LINE_COLOR,
                linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
                marker='',
                linestyle='-'
            )
        else:
            # Subsequent calls - update existing line data
            self.line.set_data(x_data, y_data)
        
        # Auto-scale axes to fit data
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Force x-axis to integer ticks (batch numbers)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Ensure threshold line stays visible and on top
        if hasattr(self, 'threshold_line'):
            self.threshold_line.set_zorder(10)  # Bring to front
        
        # Redraw canvas
        self.canvas.draw_idle()
    
    def clear_plot(self):
        """Clear the plot and reset."""
        self._criteria_met_batch = None
        if self.line is not None:
            self.line.remove()
            self.line = None

        self.ax.clear()
        self._style_axes(self.ax)
        
        # Restore labels and title
        self.ax.set_title("Match Score", fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        
        # Recreate threshold line after clearing
        self._create_match_score_threshold_line()
        
        self.canvas.draw_idle()