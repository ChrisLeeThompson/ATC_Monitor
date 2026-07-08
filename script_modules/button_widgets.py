"""
Module for button components.
"""
from PySide6.QtWidgets import QPushButton
from script_modules.app_styles import AppStyles


class StyledButton(QPushButton):

    def __init__(self, parent=None, button_text: str = ""):
        super().__init__(parent)
        self.setText(button_text)
        self.setStyleSheet(AppStyles.Button.default())


class StartButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Start")


class StopButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Stop")


class PauseResumeButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Pause")


class SettingsButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Settings")


class SaveButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Save")


class CancelButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Cancel")


class RestoreDefaultsButton(StyledButton):

    def __init__(self, parent=None):
        super().__init__(parent, "Restore Defaults")