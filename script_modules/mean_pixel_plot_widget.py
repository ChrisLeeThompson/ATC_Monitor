"""
Mean pixel value plot widget.
"""
import matplotlib
matplotlib.set_loglevel("WARNING")  # Suppress matplotlib debug messages
import logging
import numpy as np
from matplotlib.ticker import AutoLocator, MaxNLocator
from PySide6.QtCore import Qt
from script_modules.base_plot_widget import BasePlotWidget
from script_modules.app_styles import AppStyles


logger = logging.getLogger(__name__)


class MeanPixelPlotWidget(BasePlotWidget):

    def __init__(self, parent=None, min_height=200):
        """
        Initialize the mean pixel plot widget.

        :param min_height: Minimum height in pixels for the widget.
        """
        super().__init__(parent, min_height=min_height)
        
        # Set title
        self._title = "Mean Pixel Value / Specimen Current"
        self.ax.set_title(self._title, fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        
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
        
        # Plot test data for design purposes
        # self._plot_test_data()

        # Draw
        self.canvas.draw()
    
    def _plot_test_data(self):
        """Plot test data for design purposes."""
        # Generate test data - simulate mean pixel values over frames
        frames = np.arange(0, 100)
        # Create realistic-looking mean pixel values with some variation
        base_value = 128
        noise = np.random.normal(0, 5, len(frames))
        drift = np.sin(frames / 20) * 10  # Slow drift
        mean_values = base_value + noise + drift
        
        self.plot_data(frames, mean_values)
    
    def plot_data(self, x_data, y_data):
        """
        Plot mean pixel value data.
        
        Uses set_data() on subsequent calls to avoid recreating Line2D artists,
        which is significantly cheaper than remove + plot.
        
        :param x_data: X-axis data (e.g., image numbers)
        :param y_data: Y-axis data (mean pixel values)
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
        
        # Restore title
        self.ax.set_title("Mean Pixel Value", fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        
        self.canvas.draw_idle()


class PixelCurrentPlotWidget(BasePlotWidget):

    def __init__(self, parent=None, min_height=200):
        """
        Dual-axis plot widget showing mean pixel value (left axis)
        and specimen current (right axis) vs batch number.

        The right axis is hidden at construction and only revealed on the first
        call to plot_current_data(). On microscopes where specimen current is
        unavailable the widget renders as a clean single-axis mean pixel chart.

        :param min_height: Minimum height in pixels for the widget.
        """
        super().__init__(parent, min_height=min_height)

        # Set title. _title must be initialized here (not only in
        # set_specimen_current_enabled, which is reached only on Arctis systems)
        # because clear_plot() reads it on every session reset -- omitting it
        # raised AttributeError on the first reset of every non-Arctis run.
        self._title = "Mean Pixel Value / Specimen Current"
        self.ax.set_title(
            self._title,
            fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE
        )

        # Left axis styling (mean pixel value)
        self.ax.set_facecolor(AppStyles.Colors.GROUPBOX_BG)
        self.ax.spines["left"].set_visible(True)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color(AppStyles.Colors.TEXT_PRIMARY)
        self.ax.spines["bottom"].set_color(AppStyles.Colors.TEXT_PRIMARY)
        self.ax.tick_params(colors=AppStyles.Colors.TEXT_PRIMARY)

        # Right axis (specimen current) - hidden until first data arrives
        self.ax_current = self.ax.twinx()
        self.ax_current.spines["left"].set_visible(False)
        self.ax_current.spines["bottom"].set_visible(False)
        self.ax_current.spines["top"].set_visible(False)
        self.ax_current.spines["right"].set_visible(False)  # Hidden until data arrives
        self.ax_current.tick_params(
            axis='y',
            colors=AppStyles.Colors.PLOT_SPECIMEN_CURRENT_COLOR
        )
        self.ax_current.set_yticks([])  # Hide ticks until data arrives

        # Track whether the right axis has been revealed
        self._current_axis_visible = False

        # Line references
        self.pixel_line = None
        self.current_line = None

        # Draw
        self.canvas.draw()

    def _show_current_axis(self):
        """Reveal the right axis spine and style it. Called on first specimen current data."""
        self.ax_current.spines["right"].set_visible(True)
        self.ax_current.spines["right"].set_color(AppStyles.Colors.TEXT_PRIMARY)
        self.ax_current.yaxis.set_major_locator(AutoLocator())
        self._current_axis_visible = True
    
    def set_specimen_current_enabled(self, enabled: bool):
        """
        Configure whether specimen current is available on this system.

        When disabled (e.g. Arctis), the title shows only 'Mean Pixel Value'
        and the right axis remains hidden regardless of data calls.

        :param enabled: True if specimen current is available, False otherwise.
        """
        self._title = (
            "Mean Pixel Value / Specimen Current" if enabled
            else "Mean Pixel Value"
        )
        self.ax.set_title(self._title, fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        self.canvas.draw_idle()

    def plot_pixel_data(self, x_data, y_data):
        """
        Plot mean pixel value data on the left y-axis.

        :param x_data: X-axis data (batch numbers)
        :param y_data: Y-axis data (mean pixel values)
        """
        if self.pixel_line is None:
            self.pixel_line, = self.ax.plot(
                x_data,
                y_data,
                color=AppStyles.Colors.PLOT_LINE_COLOR,
                linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
                marker='',
                linestyle='-'
            )
        else:
            self.pixel_line.set_data(x_data, y_data)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.canvas.draw_idle()

    def plot_current_data(self, x_data, y_data):
        """
        Plot specimen current data on the right y-axis.

        Reveals the right axis spine on the first call, so microscopes that do
        not provide specimen current leave the axis hidden entirely.

        :param x_data: X-axis data (batch numbers)
        :param y_data: Y-axis data (specimen current values)
        """
        # Defensive guard: matplotlib's set_data raises if x and y differ in
        # length. Truncate to the shorter series rather than crash the GUI
        # thread (the worker emits aligned lists, but stay robust to drift).
        if len(x_data) != len(y_data):
            n = min(len(x_data), len(y_data))
            x_data, y_data = x_data[:n], y_data[:n]

        if not self._current_axis_visible:
            self._show_current_axis()

        if self.current_line is None:
            self.current_line, = self.ax_current.plot(
                x_data,
                y_data,
                color=AppStyles.Colors.PLOT_SPECIMEN_CURRENT_COLOR,
                linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
                marker='',
                linestyle='-'
            )
        else:
            self.current_line.set_data(x_data, y_data)

        self.ax_current.relim()
        self.ax_current.autoscale_view()
        self.canvas.draw_idle()

    def clear_plot(self):
        """Clear both axes and reset."""
        self._criteria_met_batch = None
        if self.pixel_line is not None:
            self.pixel_line.remove()
            self.pixel_line = None
        if self.current_line is not None:
            self.current_line.remove()
            self.current_line = None

        self.ax.clear()
        self._style_axes(self.ax)

        # Reset right axis - hide spine and clear flag until new data arrives
        self.ax_current.clear()
        self.ax_current.spines["left"].set_visible(False)
        self.ax_current.spines["bottom"].set_visible(False)
        self.ax_current.spines["top"].set_visible(False)
        self.ax_current.spines["right"].set_visible(False)
        self.ax_current.tick_params(
            axis='y',
            colors=AppStyles.Colors.PLOT_SPECIMEN_CURRENT_COLOR
        )
        self.ax_current.set_yticks([])  # Hide ticks until new data arrives
        self._current_axis_visible = False

        # Restore title
        self.ax.set_title(
            self._title,
            fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE
        )

        self.canvas.draw_idle()