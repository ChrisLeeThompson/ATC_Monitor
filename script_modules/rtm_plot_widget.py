"""
Real-time monitor (RTM) plot widget.
"""
import matplotlib
matplotlib.set_loglevel("WARNING")  # Suppress matplotlib debug messages
import logging
import numpy as np
from matplotlib.patches import Rectangle
from PySide6.QtCore import Signal, Qt
from script_modules.base_plot_widget import BasePlotWidget
from script_modules.app_styles import AppStyles


logger = logging.getLogger(__name__)


_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_ROTATION_SIGN = +1  # display handedness; only matters for non-0/180 rotations


def _rotate_to_edge(vx, vy, rotation_rad):
    """Rotate a y-up unit vector (clockwise-positive) and snap to the nearest cardinal edge.

    :param vx: x component in visual y-up coords (right = +x)
    :param vy: y component in visual y-up coords (up = +y)
    :param rotation_rad: clockwise-positive rotation in radians (AutoScript pattern.rotation)
    :return: one of "top", "bottom", "left", "right"
    """
    theta = _ROTATION_SIGN * rotation_rad
    c, s = np.cos(theta), np.sin(theta)
    rx = vx * c + vy * s        # clockwise-positive rotation of (vx, vy)
    ry = -vx * s + vy * c
    if abs(rx) >= abs(ry):      # exact diagonals (never produced by 0/+-180) snap horizontal
        return "right" if rx > 0 else "left"
    return "top" if ry > 0 else "bottom"


def _scan_direction_to_edges(direction, rotation_rad=0.0):
    """Map an AutoScript PatternScanDirection string to the crop-box edge(s) to emphasize,
    accounting for the pattern's on-screen rotation.

    The local scan direction is a *destination* vector in visual y-up coords (a bottom-to-top
    scan points up). That vector is rotated by ``rotation_rad`` (radians, clockwise-positive,
    matching AutoScript ``pattern.rotation``) and snapped to the nearest cardinal edge -- so a
    bottom-to-top scan rotated by -180 deg (as AutoTEM Cryo does) highlights the *bottom* edge.
    At ``rotation_rad == 0.0`` this reproduces the original mapping exactly:

      BottomToTop -> {top}        TopToBottom -> {bottom}
      LeftToRight -> {right}      RightToLeft -> {left}
      DynamicLeftToRight -> {left, right}     DynamicTopToBottom -> {top, bottom}
      DynamicAllDirections -> {top, bottom, left, right}
      InnerToOuter / OuterToInner / DynamicInnerToOuter -> set()

    "Dynamic" axis scans are bidirectional: a representative vector is rotated and snapped,
    then that edge and its opposite are emphasized (so e.g. DynamicTopToBottom + 90 deg ->
    {left, right}). DynamicAllDirections and the radial directions are rotation-invariant.
    Radial directions (Inner/Outer) and unknown/empty values map to no emphasis (a box edge
    can't represent a radial scan; these apply to circle patterns anyway).

    :param direction: e.g. "BottomToTop", "DynamicAllDirections", "" or None
    :param rotation_rad: pattern rotation in radians, clockwise-positive (default 0.0)
    :return: a subset of {"top", "bottom", "left", "right"} (possibly empty)
    """
    if not direction:
        return set()
    key = str(direction).lower().replace(" ", "").replace("_", "")

    if "alldirections" in key:
        return {"top", "bottom", "left", "right"}
    # Radial scans have no box-edge representation (rotation-invariant).
    if "innertoouter" in key or "outertoinner" in key:
        return set()

    is_dynamic = "dynamic" in key

    # Destination unit vector in visual y-up coords (right = +x, up = +y).
    if "lefttoright" in key:
        vx, vy = 1.0, 0.0
    elif "righttoleft" in key:
        vx, vy = -1.0, 0.0
    elif "bottomtotop" in key:
        vx, vy = 0.0, 1.0
    elif "toptobottom" in key:
        vx, vy = 0.0, -1.0
    else:
        return set()

    edge = _rotate_to_edge(vx, vy, rotation_rad)
    if is_dynamic:
        # Bidirectional: emphasize the rotated axis (edge + its opposite).
        return {edge, _OPPOSITE[edge]}
    return {edge}


class InteractiveCropRectangle:
    """Manages interactive crop rectangle with draggable edges."""
    
    def __init__(self, ax, canvas, rect_params, on_change_callback=None, emphasis_edges=None):
        """
        Initialize interactive crop rectangle.

        :param ax: The axes to draw on
        :param canvas: The matplotlib canvas
        :param rect_params: Dictionary with 'x', 'y', 'width', 'height'
        :param on_change_callback: Function called when rectangle changes: callback(x, y, width, height)
        :param emphasis_edges: Optional set of edges ("top"/"bottom"/"left"/"right") to draw
                               as a solid line (scan-direction indicator).
        """
        self.ax = ax
        self.canvas = canvas
        self.on_change_callback = on_change_callback

        # Create rectangle
        self.rect = Rectangle(
            (rect_params['x'], rect_params['y']),
            rect_params['width'],
            rect_params['height'],
            linewidth=1.0,
            linestyle='dashed',
            edgecolor="#04f5ff",
            facecolor='none',
            antialiased=True,
            picker=True
        )
        self.ax.add_patch(self.rect)

        # Scan-direction emphasis: solid overlay line(s) on selected edge(s),
        # kept in sync with the rectangle geometry. See _scan_direction_to_edges.
        self.emphasis_edges = set(emphasis_edges) if emphasis_edges else set()
        self.emphasis_lines = []  # list of (edge, Line2D)
        self._build_emphasis()

        # Interaction state
        self.active_edge = None
        self.press_data = None
        self.hovering_edge = None
        
        # Colors
        self.edge_color_normal = "#04f5ff"
        self.edge_color_hover = "#04f5ff"
        self.edge_color_active = "#04f5ff"
        
        # Edge detection threshold (in pixels)
        self.edge_threshold = 15
        
        # Connect events
        self.cid_press = canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.cid_leave = canvas.mpl_connect('axes_leave_event', self._on_leave_axes)
    
    def _pixels_to_data_units(self, pixels, axis='x'):
        """Convert pixel distance to data coordinates."""
        bbox = self.ax.get_window_extent()
        
        if axis == 'x':
            width_pixels = bbox.width
            xlim = self.ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            return pixels * (x_range / width_pixels)
        else:  # y-axis
            height_pixels = bbox.height
            ylim = self.ax.get_ylim()
            y_range = ylim[0] - ylim[1]  # Note: y is inverted
            return pixels * (y_range / height_pixels)
    
    def _get_corner_or_edge_at_position(self, x, y):
        """
        Check if mouse position is near any corner or edge.
        Corners take priority over edges.
        Returns: 'top_left', 'top_right', 'bottom_left', 'bottom_right',
                 'left', 'right', 'top', 'bottom', or None
        """
        if x is None or y is None:
            return None
        
        rect_x, rect_y = self.rect.get_xy()
        width = self.rect.get_width()
        height = self.rect.get_height()
        
        # Convert threshold to data units
        threshold_x = self._pixels_to_data_units(self.edge_threshold, axis='x')
        threshold_y = self._pixels_to_data_units(self.edge_threshold, axis='y')
        
        # Check if within rectangle area (with threshold)
        in_x_range = (rect_x - threshold_x <= x <= rect_x + width + threshold_x)
        in_y_range = (rect_y - threshold_y <= y <= rect_y + height + threshold_y)
        
        if not (in_x_range and in_y_range):
            return None
        
        # Calculate distances
        dist_left = abs(x - rect_x)
        dist_right = abs(x - (rect_x + width))
        dist_top = abs(y - rect_y)
        dist_bottom = abs(y - (rect_y + height))
        
        # Check for corners FIRST (corners take priority)
        # Top-left corner
        if dist_left < threshold_x and dist_top < threshold_y:
            return 'top_left'
        
        # Top-right corner
        if dist_right < threshold_x and dist_top < threshold_y:
            return 'top_right'
        
        # Bottom-left corner
        if dist_left < threshold_x and dist_bottom < threshold_y:
            return 'bottom_left'
        
        # Bottom-right corner
        if dist_right < threshold_x and dist_bottom < threshold_y:
            return 'bottom_right'
        
        # If not a corner, check for edges
        edges = {
            'left': dist_left,
            'right': dist_right,
            'top': dist_top,
            'bottom': dist_bottom
        }
        
        closest_edge = min(edges, key=lambda k: edges[k])
        closest_dist = edges[closest_edge]
        
        # Check if close enough
        if closest_edge in ['left', 'right']:
            if closest_dist < threshold_x:
                return closest_edge
        else:  # top or bottom
            if closest_dist < threshold_y:
                return closest_edge
        
        return None
    
    def _on_press(self, event):
        """Handle mouse press event."""
        if event.inaxes != self.ax:
            return
        
        # Check if clicking near a corner or edge
        edge = self._get_corner_or_edge_at_position(event.xdata, event.ydata)
        
        if edge:
            self.active_edge = edge
            x, y = self.rect.get_xy()
            width = self.rect.get_width()
            height = self.rect.get_height()
            self.press_data = {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'mouse_x': event.xdata,
                'mouse_y': event.ydata
            }
            
            # Change color to active
            self.rect.set_edgecolor(self.edge_color_active)
            # self.rect.set_linewidth(3.0)
            self.canvas.draw_idle()
    
    def _on_release(self, event):
        """Handle mouse release event."""
        if self.active_edge:
            # Reset color based on hover state
            if self.hovering_edge:
                self.rect.set_edgecolor(self.edge_color_hover)
                # self.rect.set_linewidth(2.5)
            else:
                self.rect.set_edgecolor(self.edge_color_normal)
                # self.rect.set_linewidth(2.0)
            
            self.active_edge = None
            self.press_data = None
            self.canvas.draw_idle()
            
            # Call callback with final dimensions
            if self.on_change_callback:
                x, y = self.rect.get_xy()
                width = self.rect.get_width()
                height = self.rect.get_height()
                self.on_change_callback(int(x), int(y), int(width), int(height))
    
    def _on_motion(self, event):
        """Handle mouse motion event."""
        if event.inaxes != self.ax:
            return
        
        # Handle dragging
        if self.active_edge and self.press_data:
            self._drag_edge(event)
            return
        
        # Handle hovering
        edge = self._get_corner_or_edge_at_position(event.xdata, event.ydata)
        
        if edge != self.hovering_edge:
            # Update hover state
            if edge:
                # Mouse over edge or corner
                self.rect.set_edgecolor(self.edge_color_hover)
                self.rect.set_linewidth(1.0)
                
                # Set cursor based on edge/corner
                if edge in ['left', 'right']:
                    self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
                elif edge in ['top', 'bottom']:
                    self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)
                elif edge in ['top_left', 'bottom_right']:
                    # Diagonal cursor: ╲ (top-left to bottom-right)
                    self.canvas.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif edge in ['top_right', 'bottom_left']:
                    # Diagonal cursor: ╱ (top-right to bottom-left)
                    self.canvas.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                # Mouse not over edge
                self.rect.set_edgecolor(self.edge_color_normal)
                self.rect.set_linewidth(1.0)
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            
            self.hovering_edge = edge
            self.canvas.draw_idle()
    
    def _on_leave_axes(self, event):
        """Handle mouse leaving the axes area."""
        # Reset hover state when mouse leaves plot
        if self.hovering_edge and not self.active_edge:
            self.rect.set_edgecolor(self.edge_color_normal)
            self.rect.set_linewidth(1.0)
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.hovering_edge = None
            self.canvas.draw_idle()
    
    def _drag_edge(self, event):
        """Update rectangle based on edge or corner being dragged."""
        if not self.press_data or event.xdata is None or event.ydata is None:
            return
        
        dx = event.xdata - self.press_data['mouse_x']
        dy = event.ydata - self.press_data['mouse_y']
        
        orig_x = self.press_data['x']
        orig_y = self.press_data['y']
        orig_width = self.press_data['width']
        orig_height = self.press_data['height']
        
        new_x = orig_x
        new_y = orig_y
        new_width = orig_width
        new_height = orig_height
        
        # Minimum size constraint
        min_size = 20
        
        # Handle corners (drag both dimensions)
        if self.active_edge == 'top_left':
            # Drag top and left edges
            new_x = orig_x + dx
            new_y = orig_y + dy
            new_width = max(min_size, orig_width - dx)
            new_height = max(min_size, orig_height - dy)
            # Adjust x/y if we hit minimum size
            if orig_width - dx < min_size:
                new_x = orig_x + orig_width - min_size
            if orig_height - dy < min_size:
                new_y = orig_y + orig_height - min_size
        
        elif self.active_edge == 'top_right':
            # Drag top and right edges
            new_y = orig_y + dy
            new_width = max(min_size, orig_width + dx) 
            new_height = max(min_size, orig_height - dy)
            if orig_height - dy < min_size:
                new_y = orig_y + orig_height - min_size
        
        elif self.active_edge == 'bottom_left':
            # Drag bottom and left edges
            new_x = orig_x + dx
            new_width = max(min_size, orig_width - dx)
            new_height = max(min_size, orig_height + dy)
            if orig_width - dx < min_size:
                new_x = orig_x + orig_width - min_size
        
        elif self.active_edge == 'bottom_right':
            # Drag bottom and right edges
            new_width = max(min_size, orig_width + dx)
            new_height = max(min_size, orig_height + dy)
        
        # Handle edges (drag one dimension)
        elif self.active_edge == 'left':
            new_x = orig_x + dx
            new_width = max(min_size, orig_width - dx)
            if orig_width - dx < min_size:
                new_x = orig_x + orig_width - min_size
        
        elif self.active_edge == 'right':
            new_width = max(min_size, orig_width + dx)
        
        elif self.active_edge == 'top':
            new_y = orig_y + dy
            new_height = max(min_size, orig_height - dy)
            if orig_height - dy < min_size:
                new_y = orig_y + orig_height - min_size
        
        elif self.active_edge == 'bottom':
            new_height = max(min_size, orig_height + dy)
        
        # Constrain to axes bounds
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        new_x = max(xlim[0], min(new_x, xlim[1] - new_width))
        new_y = max(ylim[1], min(new_y, ylim[0] - new_height))
        
        # Update rectangle
        self.rect.set_xy((new_x, new_y))
        self.rect.set_width(new_width)
        self.rect.set_height(new_height)
        self._refresh_emphasis()

        # Redraw
        self.canvas.draw_idle()
    
    def update_rectangle(self, x, y, width, height):
        """Programmatically update rectangle."""
        self.rect.set_xy((x, y))
        self.rect.set_width(width)
        self.rect.set_height(height)
        self._refresh_emphasis()
        self.canvas.draw_idle()

    def _edge_coords(self, edge):
        """Return ([x0, x1], [y0, y1]) endpoints for an edge of the current rect.

        The axes y-axis is inverted for image coordinates, so the rect's y is the
        visual top edge and y+height the bottom edge.
        """
        x, y = self.rect.get_xy()
        w = self.rect.get_width()
        h = self.rect.get_height()
        if edge == "top":
            return [x, x + w], [y, y]
        if edge == "bottom":
            return [x, x + w], [y + h, y + h]
        if edge == "left":
            return [x, x], [y, y + h]
        if edge == "right":
            return [x + w, x + w], [y, y + h]
        return None

    def _remove_emphasis(self):
        """Remove any existing emphasis overlay lines."""
        for _edge, line in self.emphasis_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.emphasis_lines = []

    def _build_emphasis(self):
        """Create solid overlay line(s) on the emphasized edge(s)."""
        self._remove_emphasis()
        for edge in self.emphasis_edges:
            coords = self._edge_coords(edge)
            if coords is None:
                continue
            xs, ys = coords
            line = self.ax.plot(
                xs, ys,
                color=AppStyles.Colors.PLOT_INTERACTIVE_LINE_COLOR,
                linewidth=AppStyles.Dimensions.SCAN_DIRECTION_LINE_WIDTH,
                linestyle=AppStyles.Dimensions.SCAN_DIRECTION_LINE_STYLE,
                solid_capstyle="butt",
                zorder=self.rect.get_zorder() + 1,
            )[0]
            self.emphasis_lines.append((edge, line))

    def _refresh_emphasis(self):
        """Reposition the overlay line(s) to follow the current rect geometry."""
        for edge, line in self.emphasis_lines:
            coords = self._edge_coords(edge)
            if coords is None:
                continue
            xs, ys = coords
            line.set_data(xs, ys)

    def set_emphasis_edges(self, edges):
        """Set which edge(s) are emphasized and rebuild the overlay."""
        self.emphasis_edges = set(edges) if edges else set()
        self._build_emphasis()
        self.canvas.draw_idle()

    def readd_to_axes(self):
        """Re-add the patch and overlay line(s) after a non-clearing re-image."""
        self.ax.add_patch(self.rect)
        self._build_emphasis()

    def disconnect(self):
        """Disconnect all event handlers."""
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_leave)
        self._remove_emphasis()
        self.rect.remove()


class RTMPlotWidget(BasePlotWidget):
    """RTM plot widget."""
    
    # Signal emitted when crop changes: (x, y, width, height)
    crop_changed = Signal(int, int, int, int)

    def __init__(self, parent=None, min_height=None, default_size=512, target_display_size=512):
        """
        Initialize the RTM plot widget.
        
        :param min_height: Minimum height in pixels for the widget.
        :param default_size: Default image dimensions (default: 512).
        :param target_display_size: Target size for the longest image dimension on screen.
                                    Smaller images are upscaled with nearest-neighbor to
                                    approximately this size for display only.
        """
        super().__init__(parent, min_height=min_height)

        self.interactive_crop = None

        # Pattern scan direction (e.g. "BottomToTop"); drives the solid edge
        # highlight on the crop box. Empty string = no emphasis.
        self._scan_direction = ""
        # Pattern rotation in radians, CW-positive (AutoScript pattern.rotation);
        # rotates the on-screen scan direction before choosing the bold edge.
        self._scan_rotation = 0.0
        
        # Store current image dimensions in ORIGINAL pixel space
        # (used for worker communication and crop coordinate conversion)
        self.image_width = default_size
        self.image_height = default_size
        
        # Display upscaling (nearest-neighbor, display only)
        self._target_display_size = target_display_size
        self._display_scale = 1  # Integer scale factor (1 = no upscaling)
        
        # Relative crop rectangle (fractions 0.0-1.0 of image dimensions)
        # Preserved across image dimension changes for proportional scaling
        self._relative_crop = None  # (x_frac, y_frac, w_frac, h_frac)
        
        # Saved crop rectangle from previous session (original pixel space)
        # If set, used instead of 85% default on first image
        self._saved_crop_rect = None  # (x, y, width, height) or None
        
        # Cached imshow artist for set_data() fast path
        self._imshow_artist = None
        
        # Setup the plot area
        self._setup_plot_area()
        
        # Add a default crop rectangle
        self._add_default_crop_rectangle()
    
    def _setup_plot_area(self):
        """Setup plot area with display-scaled image dimensions."""
        # Clear axes
        self.ax.clear()
        self._style_axes(self.ax)
        
        # Set limits based on display dimensions (original × scale)
        display_w = self.image_width * self._display_scale
        display_h = self.image_height * self._display_scale
        self.ax.set_xlim(0, display_w)
        self.ax.set_ylim(display_h, 0)  # Inverted for image coordinates
        
        # Set title
        self.ax.set_title("Processed RTM Image", fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
        
        # Remove tick labels (just show background)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Styling
        self.ax.set_facecolor(AppStyles.Colors.PLOT_BG)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        
        # Invalidate cached imshow artist (axes were cleared)
        self._imshow_artist = None
        
        # Draw
        self.canvas.draw()
    
    def _upscale_for_display(self, image_array):
        """
        Upscale image with nearest-neighbor interpolation for display only.
        
        Computes a dynamic integer scale factor so the longest dimension
        approaches target_display_size. Images already at or above the
        target are not upscaled.
        
        :param image_array: Original image array (2D or 3D)
        :return: (upscaled_array, scale_factor)
        """
        height, width = image_array.shape[:2]
        scale = max(1, self._target_display_size // max(height, width))
        
        if scale <= 1:
            return image_array, 1
        
        # Nearest-neighbor upscale via array element repetition
        upscaled = np.repeat(np.repeat(image_array, scale, axis=0), scale, axis=1)
        return upscaled, scale
    
    def plot_image(self, image_array, reset_crop=True, title="Processed RTM Image"):
        """
        Plot image data and update plot dimensions.
        
        Images are dynamically upscaled with nearest-neighbor interpolation
        for display purposes only. The original pixel dimensions are preserved
        for all crop rectangle coordinates emitted to the worker.
        
        Uses a fast path (set_data) when image dimensions haven't changed,
        avoiding the expensive ax.clear() + imshow() cycle. This is the
        common case during monitoring.
        
        :param image_array: Image data (height, width) for grayscale or (height, width, channels) for color.
        :param reset_crop: If True, reset crop to default for new image size. If False, attempt to preserve existing crop.
        :param title: Plot title.
        """
        # Get original image dimensions
        if len(image_array.shape) == 2:
            height, width = image_array.shape
        elif len(image_array.shape) == 3:
            height, width = image_array.shape[:2]
        else:
            raise ValueError(f"Invalid image shape: {image_array.shape}")
        
        # Upscale for display
        display_image, scale = self._upscale_for_display(image_array)
        display_w = width * scale
        display_h = height * scale
        
        # Check if original dimensions changed
        dimensions_changed = (width != self.image_width or height != self.image_height)
        
        if dimensions_changed:
            logger.info(
                f"Image dimensions changed: {self.image_width}x{self.image_height} → "
                f"{width}x{height} (display: {display_w}x{display_h}, scale: {scale}x)"
            )
            self.image_width = width
            self.image_height = height
            self._display_scale = scale
            
            # Full redraw required - clear axes and rebuild
            self.ax.clear()
            self._style_axes(self.ax)
            self.ax.set_xlim(0, display_w)
            self.ax.set_ylim(display_h, 0)  # Inverted for image coordinates
            
            self._imshow_artist = self.ax.imshow(
                display_image, cmap='gray', aspect='equal',
                interpolation='nearest', extent=(0, display_w, display_h, 0)
            )
            
            self.ax.set_title(title, fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Scale crop proportionally to new image dimensions
            self._scale_crop_to_new_dimensions()
            self.canvas.draw()
        
        elif self._imshow_artist is not None:
            # Fast path — same dimensions, just swap the image data
            self._imshow_artist.set_data(display_image)
            self._imshow_artist.autoscale()  
            self.canvas.draw_idle()
        
        else:
            # First image at this size (no cached artist yet)
            self._display_scale = scale
            display_w = width * scale
            display_h = height * scale
            
            self.ax.set_xlim(0, display_w)
            self.ax.set_ylim(display_h, 0)
            
            self._imshow_artist = self.ax.imshow(
                display_image, cmap='gray', aspect='equal',
                interpolation='nearest', extent=(0, display_w, display_h, 0)
            )
            self.ax.set_title(title, fontsize=AppStyles.Dimensions.PLOT_TITLE_FONT_SIZE)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Re-add existing crop patch (and its emphasis overlay) or create default
            if self.interactive_crop:
                self.interactive_crop.readd_to_axes()
            else:
                self._add_default_crop_rectangle()
            self.canvas.draw()
    
    def set_saved_crop_rect(self, x: int, y: int, width: int, height: int):
        """
        Set a saved crop rectangle from a previous session.
        
        Will be used instead of the 85% default when the first image loads.
        Coordinates are in original pixel space.
        
        :param x: Left edge in pixels
        :param y: Top edge in pixels
        :param width: Width in pixels
        :param height: Height in pixels
        """
        self._saved_crop_rect = (x, y, width, height)

    def _add_default_crop_rectangle(self):
        """
        Add crop rectangle scaled to current image size.
        
        Uses saved crop from previous session if available (clamped to current
        image bounds). Otherwise creates an 85% centered default.
        """
        if self._saved_crop_rect is not None:
            sx, sy, sw, sh = self._saved_crop_rect
            # Clamp to current image dimensions
            crop_x = max(0, min(sx, self.image_width - 20))
            crop_y = max(0, min(sy, self.image_height - 20))
            crop_width = max(20, min(sw, self.image_width - crop_x))
            crop_height = max(20, min(sh, self.image_height - crop_y))
            # Clear so it's only used once (subsequent resets use relative_crop)
            self._saved_crop_rect = None
            logger.debug(
                f"Restoring saved crop rectangle: x={crop_x}, y={crop_y}, "
                f"width={crop_width}, height={crop_height}"
            )
        else:
            # Default crop: 85% of image size, centered
            crop_width = max(20, int(self.image_width * 0.85))
            crop_height = max(20, int(self.image_height * 0.85))
            crop_x = (self.image_width - crop_width) // 2
            crop_y = (self.image_height - crop_height) // 2
            logger.debug(
                f"Creating default crop rectangle: x={crop_x}, y={crop_y}, "
                f"width={crop_width}, height={crop_height} (display scale: {self._display_scale}x)"
            )
        
        # Store as relative fractions (scale-independent)
        if self.image_width > 0 and self.image_height > 0:
            self._relative_crop = (
                crop_x / self.image_width,
                crop_y / self.image_height,
                crop_width / self.image_width,
                crop_height / self.image_height
            )
        
        # Scale to display coordinates for drawing
        s = self._display_scale
        crop_params = {
            'x': crop_x * s,
            'y': crop_y * s,
            'width': crop_width * s,
            'height': crop_height * s
        }
        
        self.add_crop_rectangle(crop_params)
    
    def _scale_crop_to_new_dimensions(self):
        """
        Scale crop rectangle proportionally to new image dimensions.
        
        Uses stored relative crop fractions to compute new absolute pixel
        coordinates in original space. Draws the rectangle in display space
        and emits crop_changed in original space so the worker receives
        correct values.
        """
        if self._relative_crop is None:
            self._add_default_crop_rectangle()
            return
        
        x_frac, y_frac, w_frac, h_frac = self._relative_crop
        
        # Convert relative fractions to original pixel coordinates
        min_size = 20
        x = int(x_frac * self.image_width)
        y = int(y_frac * self.image_height)
        width = max(min_size, int(w_frac * self.image_width))
        height = max(min_size, int(h_frac * self.image_height))
        
        # Constrain to image bounds (original space)
        x = max(0, min(x, self.image_width - width))
        y = max(0, min(y, self.image_height - height))
        
        logger.info(
            f"Crop scaled to new dimensions: ({x}, {y}) {width}x{height} "
            f"(image: {self.image_width}x{self.image_height}, "
            f"display scale: {self._display_scale}x)"
        )
        
        # Scale to display coordinates for drawing
        s = self._display_scale
        crop_params = {
            'x': x * s,
            'y': y * s,
            'width': width * s,
            'height': height * s
        }
        self.add_crop_rectangle(crop_params)
        
        # Emit crop_changed in original pixel space for the worker
        self.crop_changed.emit(x, y, width, height)
    
    def add_crop_rectangle(self, rect_params, interactive=True):
        """
        Add crop rectangle to the plot.
        
        :param rect_params: Dicitonary with 'x', 'y', 'width', 'height' to define the crop rectangle.
        :param interactive: If True, creates an interactive rectangle that can be dragged. If False, creates a static rectangle.
        """
        # Remove old rectangle if exists
        if self.interactive_crop:
            self.interactive_crop.disconnect()
            self.interactive_crop = None
        
        if interactive:
            # Create interactive crop rectangle, re-applying any scan-direction
            # emphasis so it survives crop rebuilds (resize/scale/re-image).
            self.interactive_crop = InteractiveCropRectangle(
                self.ax,
                self.canvas,
                rect_params,
                on_change_callback=self._on_crop_changed,
                emphasis_edges=_scan_direction_to_edges(self._scan_direction, self._scan_rotation)
            )
        else:
            # Create static rectangle
            rect = Rectangle(
                (rect_params['x'], rect_params['y']),
                rect_params['width'],
                rect_params['height'],
                linewidth=AppStyles.Dimensions.PLOT_LINE_WIDTH,
                linestyle='dashed',
                edgecolor=AppStyles.Colors.PLOT_INTERACTIVE_LINE_COLOR,
                facecolor='none',
                antialiased=True
            )
            self.ax.add_patch(rect)
        
        self.canvas.draw_idle()
    
    def _on_crop_changed(self, x, y, width, height):
        """
        Callback when crop rectangle changes (coordinates in display space).
        
        Converts display-space coordinates back to original pixel space
        for storage and worker communication.
        """
        # Convert from display space to original pixel space
        s = self._display_scale
        orig_x = int(x / s)
        orig_y = int(y / s)
        orig_w = int(width / s)
        orig_h = int(height / s)
        
        logger.info(f"Crop changed: ({orig_x}, {orig_y}) {orig_w}x{orig_h}")
        
        # Store as relative fractions (scale-independent)
        if self.image_width > 0 and self.image_height > 0:
            self._relative_crop = (
                orig_x / self.image_width,
                orig_y / self.image_height,
                orig_w / self.image_width,
                orig_h / self.image_height
            )
        
        # Emit in original pixel space for the worker
        self.crop_changed.emit(orig_x, orig_y, orig_w, orig_h)
    
    def update_crop_rectangle(self, x, y, width, height):
        """
        Update crop rectangle programmatically (coordinates in original pixel space).
        
        Converts to display space before updating the interactive rectangle.
        """
        if self.interactive_crop:
            s = self._display_scale
            self.interactive_crop.update_rectangle(
                x * s, y * s, width * s, height * s
            )
    
    def get_crop_dimensions(self):
        """
        Get current crop dimensions in original pixel space.
        
        :return: (x, y, width, height) in original pixels, or None
        """
        if self.interactive_crop:
            x, y = self.interactive_crop.rect.get_xy()
            width = self.interactive_crop.rect.get_width()
            height = self.interactive_crop.rect.get_height()
            s = self._display_scale
            return int(x / s), int(y / s), int(width / s), int(height / s)
        return None
    
    def set_scan_direction(self, direction, rotation_rad=0.0):
        """
        Set the pattern scan direction (+ rotation) and highlight the matching crop-box edge.

        The on-screen scan direction = the local scan direction rotated by ``rotation_rad``
        (radians, clockwise-positive, matching AutoScript ``pattern.rotation``). The
        destination edge (e.g. the top edge for an un-rotated bottom-to-top scan, or the
        bottom edge once AutoTEM Cryo rotates it -180 deg) is drawn as a solid line; the
        other sides stay dashed. Unknown / radial / all-directions values clear the
        emphasis. Safe to call before or after the crop exists.

        :param direction: AutoScript scan-direction string (e.g. "BottomToTop"), or ""
        :param rotation_rad: pattern rotation in radians, clockwise-positive (default 0.0)
        """
        self._scan_direction = direction or ""
        self._scan_rotation = rotation_rad or 0.0
        edges = _scan_direction_to_edges(self._scan_direction, self._scan_rotation)
        logger.info(
            f"Scan direction set to '{self._scan_direction}' "
            f"(rotation {self._scan_rotation:.3f} rad) -> emphasize edges: {edges or 'none'}"
        )
        if self.interactive_crop:
            self.interactive_crop.set_emphasis_edges(edges)

    def get_image_dimensions(self):
        """Get current image dimensions."""
        return self.image_width, self.image_height
    
    def clear_plot(self):
        """Clear the plot."""
        if self.interactive_crop:
            self.interactive_crop.disconnect()
            self.interactive_crop = None
        
        self._setup_plot_area()
        self.canvas.draw_idle()