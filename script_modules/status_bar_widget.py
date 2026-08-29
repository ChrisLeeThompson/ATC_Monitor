"""
Module manages the status bar component.
"""
from PySide6.QtWidgets import QStatusBar, QLabel
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFontMetrics
from script_modules.app_styles import AppStyles, StatusText


class StatusBarWidget(QStatusBar):
    """
    Two areas. The left area: one label, three message layers, display
    precedence:

        timed (while its timer runs) > warning > persistent

    The warning layer exists for sticky advisories (Auto CB outcomes,
    validation failures): before it, those lived in the persistent slot and
    lost to ordinary state churn -- any later state message overwrote them.
    The persistent slot keeps tracking underneath a standing warning, so
    clearing the warning reveals the newest state, never a stale one.

    The right area: a permanent connection indicator (the Hydra utilities
    status-bar pattern), fully independent of the left layers -- same
    font, right aligned, no timer. It always states the connection
    ("Not connected" / "Connecting to microscope..." / "Connected"),
    which frees the whole left area for messages to the operator.
    """

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
        # The persistent text a timed message temporarily covers. Without it a
        # timed message erased the persistent one for good: the mid-run "re-check
        # your completion thresholds" warning survived only until the next
        # 5-second worker status, which then cleared the bar to empty.
        self._persistent_text = ""
        # Sticky warning ("" = none). Outranks the persistent text until
        # explicitly cleared or replaced; a running timed message still covers
        # it, because timed messages are direct feedback to something the
        # operator just did.
        self._warning_text = ""
        # Set style
        self.setStyleSheet(AppStyles.StatusBar.default())

    def _create_components(self):
        """Create components in the status bar."""
        # Create status label
        self.status_label = QLabel()
        # Set style
        self.status_label.setStyleSheet(AppStyles.Label.status_bar())
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # Connection indicator: same styling as the message label, content
        # sized ("Connected" holds the slot narrow, so the left area keeps
        # nearly its whole width budget).
        self.connection_label = QLabel(StatusText.NOT_CONNECTED)
        self.connection_label.setStyleSheet(AppStyles.Label.status_bar())
        self.connection_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _add_components_to_status_bar(self):
        """Add components to the status bar."""
        self.addWidget(self.status_label, 1)
        # Permanent = QStatusBar's native right-side area: never covered by
        # showMessage-style traffic, survives everything on the left.
        self.addPermanentWidget(self.connection_label)

    def _set_label_text(self, text: str):
        """
        The single write path to the label, with the elision backstop.

        Authored messages target a ~90-character budget, but formatted
        details can still overflow at the app's window width -- overflow must
        degrade to an ellipsis, never clip mid-glyph. Elision engages only on
        a visible label: a never-shown widget (offscreen tests, construction
        time) reports a default width with no relation to the real bar, and
        eliding against it would corrupt text nothing is rendering. Kept
        defensive throughout -- a failed elision must fall back to the full
        text, and a positive-width label must never be handed an empty
        elision for a non-empty message.
        """
        if text and self.status_label.isVisible():
            width = self.status_label.width()
            if width > 0:
                try:
                    elided = QFontMetrics(self.status_label.font()).elidedText(
                        text, Qt.TextElideMode.ElideRight, width
                    )
                except Exception:
                    elided = ""
                if elided:
                    text = elided
        self.status_label.setText(text)

    def _refresh(self):
        """Repaint the label from the layered state (warning over persistent)."""
        self._set_label_text(self._warning_text or self._persistent_text)

    @Slot(str)
    def set_status_text(self, status_text: str):
        """
        Set the status label text (persistent until the next persistent message).

        A standing warning stays on top; the persistent slot keeps tracking
        underneath it so clearing the warning reveals this text.

        :param status_text: Text to display in the status label.
        """
        # Cancel any pending timed clear since a new message takes priority
        self._clear_timer.stop()
        self._persistent_text = status_text
        self._refresh()

    @Slot(str, int)
    def set_timed_status_text(self, status_text: str, duration_s: int):
        """
        Show a message for a duration, then restore the covered layer.

        :param status_text: Text to display in the status label.
        :param duration_s: Duration in seconds before the message expires.
        """
        self._set_label_text(status_text)
        self._clear_timer.start(duration_s * 1000)

    def _clear_timed_message(self):
        """Restore the covered layer (warning, else persistent) on expiry."""
        self._refresh()

    @Slot(str)
    def set_warning_text(self, warning_text: str):
        """
        Set (or with "" clear) the sticky warning shown over persistent text.

        :param warning_text: Warning to display; empty string clears it.
        """
        # A warning preempts an in-flight timed overlay: letting the stale
        # timer fire later would only repaint the same warning, while hiding
        # that the overlay was cut short.
        self._clear_timer.stop()
        self._warning_text = warning_text
        self._refresh()

    @Slot()
    def clear_warning(self):
        """Clear the sticky warning, revealing the latest persistent text."""
        self.set_warning_text("")

    @Slot(str)
    def set_connection_text(self, text: str):
        """
        Set the right-side connection indicator.

        Independent of the left label's layers: no timer covers it, no
        warning or persistent message touches it, and it never clears --
        it always states the current connection.

        :param text: The connection state to display.
        """
        if self.connection_label.text() != text:
            self.connection_label.setText(text)

    @Slot(bool)
    def set_status_label_visibility(self, visible: bool):
        """
        Set the visibility of the status label.
        :param visible: True to show, False to hide.
        """
        self.status_label.setVisible(visible)
