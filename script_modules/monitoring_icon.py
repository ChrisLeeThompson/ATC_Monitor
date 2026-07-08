"""
Module handles a graphic for indicating the pattern monitoring status.
"""
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal


class CatbugMonitoringIcon(QLabel):

    # Signal
    monitoring_status_changed = Signal(bool)  # True if monitoring is active, False otherwise

    def __init__(self, active_icon_path: str, inactive_icon_path: str, scale_factor: float = 1.6):
        super().__init__()
        self.active_icon_path = active_icon_path
        self.inactive_icon_path = inactive_icon_path
        self.scale_factor = scale_factor

        # Set initial state
        self._is_active = False

        # Setup label
        self._setup_label()

        # Set initial inactive state
        self.set_monitoring_active(False)
    
    def _setup_label(self):
        """Configure label properties."""
        # Load initial pixmap to get dimensions
        self.pix = QPixmap(self.inactive_icon_path)
        self.setPixmap(self.pix)
        # Set default pix map
        self.pix = QPixmap(self.inactive_icon_path)
        self.setPixmap(self.pix)
        # Scale the pix map
        self.pix.scaled(self.pix.width(), self.pix.height())
        # Set default label size
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.setFixedSize(int(self.pix.width() // self.scale_factor),
                          int(self.pix.height() // self.scale_factor))
        self.setMargin(4)

    @Slot(bool)
    def set_monitoring_active(self, active: bool):
        """
        Set the monitoring icon to active or inactive.
        """
        if self._is_active != active:
            self._is_active = active
            
            # Update icon
            if active:
                self.pix = QPixmap(self.active_icon_path)
            else:
                self.pix = QPixmap(self.inactive_icon_path)
            
            self.setPixmap(self.pix)
            
            # Emit signal to notify others
            self.monitoring_status_changed.emit(active)
    
    def is_active(self) -> bool:
        """
        Get current monitoring status.
        :return: True if monitoring is active, False otherwise.
        """
        return self._is_active
