"""
Base module for creating matplotlib plot widgets.
"""
import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QSizePolicy
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from script_modules.app_styles import AppStyles


logger = logging.getLogger(__name__)


# Cached display presence for the draw guard. QGuiApplication.screens()
# rebuilds a wrapper list on every call, and the guard used to run it on
# every canvas draw -- the GUI thread's hottest native call, exercised
# hardest during RDP screen churn (the 2026-08-25 0xc0000374 heap corruption
# was detected inside this enumeration during a session-reset draw burst).
# Presence only changes on screen-topology events, so enumerate only then:
# the cache is invalidated by the screenRemoved/screenAdded/
# primaryScreenChanged connections made in __init__, and keyed on the app
# instance so a swapped instance (tests use fakes) re-reads. Main-thread
# only, like every draw.
_screens_cache_app = None      # the app instance the cache was read from
_screens_cache_present = True  # cached bool(app.screens())


def _invalidate_screens_cache(*_args):
    """Force the next _screens_available() to re-enumerate."""
    global _screens_cache_app
    _screens_cache_app = None


class BasePlotWidget(QWidget):
    """
    Base class for matplotlib plot widgets.
    Handles common setup: figure, canvas, and layout.
    """

    # Signals
    plot_clicked = Signal(float, float)  # x, y coordinates of the click

    def __init__(self, parent=None, min_height=None):
        """
        Initialize the base plot widget.
        
        :param parent: Parent widget
        :param min_height: Minimum height in pixels for the widget
        """
        super().__init__(parent)

        # Set minimum height if specified
        if min_height:
            self.setMinimumHeight(min_height)

        # Batch number at which the criteria-met marker line has been drawn
        # (None until drawn; reset by subclass clear_plot on session reset)
        self._criteria_met_batch = None

        # Screen-availability guard state (see _guarded_draw). Initialized before
        # the canvas is created so the wrapped draw methods can rely on them.
        self._redraw_pending = False     # a draw was skipped while no screens
        self._no_screens_warned = False  # rate-limit the "no screens" log

        # Create matplotlib components
        self._create_matplotlib_components()
        # Setup layout
        self._setup_layout()

        # Auto-resume: repaint any deferred draw as soon as a display returns.
        # screenRemoved must invalidate the presence cache, or a draw after a
        # disconnect would trust a stale "screens present" and reach the Qt
        # fatal the guard exists to prevent (added/primary invalidate inside
        # _on_screens_changed).
        app = QGuiApplication.instance()
        if app is not None:
            app.screenRemoved.connect(_invalidate_screens_cache)
            app.screenAdded.connect(self._on_screens_changed)
            app.primaryScreenChanged.connect(self._on_screens_changed)
    
    def _create_matplotlib_components(self):
        """
        Create matplotlib figure, canvas, and axes.
        """
        # Create figure with constrained_layout
        # This is better than tight_layout() for dynamic resizing
        self.figure = Figure(
            facecolor=AppStyles.Colors.GROUPBOX_BG,
            constrained_layout=True
        )
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Guard every canvas draw against "no screens available". A redraw issued
        # while the session has lost its display (RDP disconnect, screen lock,
        # monitor sleep) makes Qt call qFatal("Cannot create window: no screens
        # available") -> abort(). Wrapping the canvas's own draw methods covers
        # every draw path (ours and matplotlib-internal) without chasing the many
        # call sites; when no screen is present we skip the draw, remember it, and
        # repaint on reconnect (_on_screens_changed). Monitoring is unaffected --
        # the worker thread never touches the screen.
        self._raw_canvas_draw = self.canvas.draw
        self._raw_canvas_draw_idle = self.canvas.draw_idle
        self.canvas.draw = self._guarded_draw
        self.canvas.draw_idle = self._guarded_draw_idle

        # Create axes
        self.ax = self.figure.add_subplot(111)

        # Apply styling
        self._style_axes(self.ax)

        # Connect click event
        self.canvas.mpl_connect("button_press_event", self._on_click)

    @staticmethod
    def _screens_available():
        """True when at least one display is attached (a canvas draw is safe).

        Reads the module-level cache; screens() is enumerated only when the
        cache is invalid (first use, topology change, app instance change).
        """
        global _screens_cache_app, _screens_cache_present
        app = QGuiApplication.instance()
        # No application yet (shouldn't happen post-construction) -> assume safe.
        if app is None:
            return True
        if app is not _screens_cache_app:
            _screens_cache_present = bool(app.screens())
            _screens_cache_app = app
        return _screens_cache_present

    def _guarded_draw(self):
        """Screen-guarded replacement for canvas.draw (see _create_matplotlib_components)."""
        if self._screens_available():
            self._redraw_pending = False
            self._no_screens_warned = False
            self._raw_canvas_draw()
        else:
            self._defer_draw()

    def _guarded_draw_idle(self):
        """Screen-guarded replacement for canvas.draw_idle."""
        if self._screens_available():
            self._redraw_pending = False
            self._no_screens_warned = False
            self._raw_canvas_draw_idle()
        else:
            self._defer_draw()

    def _defer_draw(self):
        """Remember a redraw is owed and log once until a display returns."""
        self._redraw_pending = True
        if not self._no_screens_warned:
            logger.warning(
                "No screens available - skipping plot redraw to avoid a Qt fatal; "
                "will repaint when a display returns"
            )
            self._no_screens_warned = True

    def _on_screens_changed(self, *args):
        """Auto-resume: repaint a deferred draw once a display is (re)connected."""
        _invalidate_screens_cache()
        if self._redraw_pending and self._screens_available():
            self._redraw_pending = False
            self._no_screens_warned = False
            self._raw_canvas_draw_idle()

    def _style_axes(self, ax):
        """
        Apply styling to axes.
        Can be overridden by subclasses for custom styling.
        """
        # Set background
        ax.set_facecolor(AppStyles.Colors.GROUPBOX_BG)

        # Styling spines
        self.ax.spines["left"].set_color(AppStyles.Colors.PLOT_SPINE_COLOR)
        self.ax.spines["bottom"].set_color(AppStyles.Colors.PLOT_SPINE_COLOR)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        
        # Style labels
        ax.xaxis.label.set_color(AppStyles.Colors.TEXT_PRIMARY)
        ax.yaxis.label.set_color(AppStyles.Colors.TEXT_PRIMARY)
        ax.title.set_color(AppStyles.Colors.TEXT_PRIMARY)

        # Attempt to disable render artifacts at edges of the plots
        self.figure.patch.set_antialiased(False)                                        # Disable anti-aliasing
        ax.patch.set_antialiased(False)                                                 # Disable anti-aliasing for axes background
        self.figure.set_dpi(110)                                                        # Set figure DPI to 110 for better pixel alignment
        self.canvas.setStyleSheet(f"background-color: {AppStyles.Colors.GROUPBOX_BG};") # Set canvas background to match figure background
    
    def _setup_layout(self):
        """
        Setup widget layout with groupbox container.
        """
        # Create groupbox container
        group_box = QGroupBox()
        group_box_layout = QVBoxLayout()
        group_box_layout.setContentsMargins(0, 0, 0, 0)
        group_box_layout.addWidget(self.canvas)
        group_box.setLayout(group_box_layout)
        group_box.setStyleSheet(AppStyles.GroupBox.plot())

        # Main widget layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group_box)
        self.setLayout(layout)
    
    def _on_click(self, event):
        """Handle click events on the plot."""
        if event.inaxes == self.ax and event.xdata and event.ydata:
            self.plot_clicked.emit(event.xdata, event.ydata)
    
    def mark_criteria_met(self, batch_number):
        """
        Draw a vertical marker line at the batch where all criteria were met.

        Idempotent: only the first call draws the line. The line is a standalone
        artist, so subsequent plot updates (set_data/relim/autoscale) leave it in
        place. Subclass clear_plot resets the guard for the next session.

        :param batch_number: Batch number (plot x-coordinate) to mark
        """
        if self._criteria_met_batch is not None:
            return
        self.ax.axvline(
            batch_number,
            color=AppStyles.Colors.RESULT_MATCH,
            linestyle='--',
            linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
            zorder=5,
        )
        self._criteria_met_batch = batch_number
        self.canvas.draw_idle()

    def clear_plot(self):
        """Clear the plot. Should be overridden by subclasses."""
        self._criteria_met_batch = None
        self.ax.clear()
        self._style_axes(self.ax)
        self.canvas.draw_idle()
    
    def get_figure(self):
        """Return the matplotlib figure."""
        return self.figure
    
    def get_canvas(self):
        """Return the matplotlib canvas."""
        return self.canvas