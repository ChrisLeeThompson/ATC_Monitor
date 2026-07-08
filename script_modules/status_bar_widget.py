"""
Module manages the status bar component.
"""
from PySide6.QtWidgets import QStatusBar, QLabel
from PySide6.QtCore import Qt, Slot, QTimer
from script_modules.app_styles import AppStyles


class StatusBarWidget(QStatusBar):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create components
        self._create_components()
        # Add components to status bar
        self._add_components_to_status_bar()
        # Timer for auto-clearing timed messages
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(self._clear_timed_message)
        # Set style
        self.setStyleSheet(AppStyles.StatusBar.default())
    
    def _create_components(self):
        """Create components in the status bar."""
        # Create status label
        self.status_label = QLabel()
        # Set style
        self.status_label.setStyleSheet(AppStyles.Label.status_bar())
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _add_components_to_status_bar(self):
        """Add components to the status bar."""
        self.addWidget(self.status_label, 1)

    @Slot(str)
    def set_status_text(self, status_text: str):
        """
        Set the status label text (persistent until overwritten).
        
        :param status_text: Text to display in the status label.
        """
        # Cancel any pending timed clear since a new message takes priority
        self._clear_timer.stop()
        self.status_label.setText(status_text)
    
    @Slot(str, int)
    def set_timed_status_text(self, status_text: str, duration_s: int):
        """
        Set the status label text that auto-clears after a duration.
        
        :param status_text: Text to display in the status label.
        :param duration_s: Duration in seconds before the message clears.
        """
        self.status_label.setText(status_text)
        self._clear_timer.start(duration_s * 1000)
    
    def _clear_timed_message(self):
        """Clear the status label when the timer expires."""
        self.status_label.setText("")
    
    @Slot(bool)
    def set_status_label_visibility(self, visible: bool):
        """
        Set the visibility of the status label.
        :param visible: True to show, False to hide.
        """
        self.status_label.setVisible(visible)