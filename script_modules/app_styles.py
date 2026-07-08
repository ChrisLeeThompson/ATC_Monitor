"""
Central repository for application styles and tool tips.
"""
from pathlib import Path


class StyleColors:
    
    MAIN_BG = "#273945"
    GROUPBOX_BG = "#34454f" # 34454f, 2f3e47 (darker), 394856 (lighter)
    GROUPBOX_BG_DARK = "#2F3E47"
    INPUT_BG = "#1f2d36"
    INPUT_BORDER = "#0f1e28" # "#f0ad4e"
    INPUT_BORDER_AUTO = "#009900"
    BUTTON_BG = "#476273"
    BUTTON_HOVER = "#2ea2ec"
    BUTTON_PRESSED = "#1F6FA1" # 1c5985
    BUTTON_DISABLED = "#3d5462"
    CHECKBOX_INDICATOR_DISABLED = "#4C88AD"
    TEXT_PRIMARY = "#ffffff"
    TEXT_DISABLED = "#808f99"
    RESULT_MATCH = "#00ff85"    # 009900 (ASV green)
    TEXT_BLACK = "#000000"
    # Plot colors
    PLOT_BG = "#000000"
    PLOT_INTERACTIVE_LINE_COLOR = "#04f5ff"
    PLOT_LINE_COLOR = "#eb70a9" # plot line color, v2.5: #38003c, catbug: a1c7ea (blue), bright_blue: 9df1ed, catbug (glove): eb70a9, TEM blue: #00CEC8
    PLOT_SPINE_COLOR = "#ffffff"
    PLOT_SPECIMEN_CURRENT_COLOR = "#f5c842"

class StyleDimensions:

    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 920
    WINDOW_CONTENTS_MARGIN = 4
    BORDER_RADIUS_LARGE = "8.0px"
    BORDER_RADIUS_SMALL = "4.0px"
    MARGIN = "8px"
    PADDING = "4px"
    FONT_SIZE_NORMAL = "11pt"
    FONT_SIZE_LARGE = "14pt"
    SPINBOX_WIDTH = 100
    GROUPBOX_MARGIN = 4  # Margin for group boxes in the main window
    GRID_LAYOUT_CONTENTS_MARGIN = 8
    GRID_LAYOUT_HSPACING = 60
    GRID_LAYOUT_VSPACING = 10
    COLUMN_LAYOUT_MINIMUM_WIDTH = 460   # Minimum width for column layouts (pattern plot group boxes)
    CHECKBOX_SPACING = "16px"
    RESULT_LABEL_MARGIN = "2px"
    SETTINGS_DIALOG_WIDTH = 460
    COMBOBOX_WIDTH = 180
    # Plot dimensions
    PLOT_MINIMUM_HEIGHT = 200
    PLOT_LINE_WIDTH = 1.0
    PLOT_TITLE_FONT_SIZE = 11
    # Scan-direction crop edge indicator. The destination edge of the crop box
    # (e.g. the top edge for a bottom-to-top scan) is drawn solid to stand out
    # from the dashed sides. Adjust these to taste:
    #   - LINE_WIDTH: weight of the emphasized edge (1.0 = same as the dashed sides)
    #   - LINE_STYLE: "solid" (default) or "dashed" to match the rest of the box
    # Color reuses PLOT_INTERACTIVE_LINE_COLOR (the crop box color) above.
    SCAN_DIRECTION_LINE_WIDTH = 2.0
    SCAN_DIRECTION_LINE_STYLE = "solid"


class WindowText:

    WINDOW_TITLE = "ATC Monitor 3.3.0"

    WINDOW_INFO_LABEL_1 = ""
    
    WINDOW_INFO_LABEL_2 = "Click Start to begin monitoring."

    # Catbug status messages shown while a run is active (see information_label_2).
    WINDOW_INFO_MONITORING_ACTIVE = "Monitoring active..."
    WINDOW_INFO_MONITORING_IDLE = "Monitoring..."
    WINDOW_INFO_MONITORING_PAUSED = "Monitoring paused"

    SETTINGS_DIALOG_TITLE = "ATC Monitor Settings"


class ToolTips:

    # Control group box tool tips

    IMAGE_COUNT_RADIOBUTTON = "Select to set the number of RTM images to analyze in a batch."

    TIME_INTERVAL_RADIOBUTTON = "Select to set the number of seconds to acquire RTM images for analysis in a batch.\n" + \
                                "The number of images to acquire is automatically calculated \n" + \
                                "based on the rate of RTM images generated from the microscope."
    
    NUMBER_OF_IMAGES_LABEL = "Number of RTM images to process and analyze in a batch.\n" + \
                             "After analysis, another batch of images are processed.\n" + \
                             "Only applicable if 'Image Count' mode is selected."
    
    ANALYSIS_INTERVAL_LABEL = "Time interval (in seconds) to acquire RTM images for batch analysis.\n" + \
                              "The number of images to acquire is automatically calculated \n" + \
                              "based on the rate of RTM images generated from the microscope.\n" + \
                              "Only applicable if 'Time Interval' mode is selected."
    
    MEAN_PIXEL_SLOPE_THRESHOLD_LABEL = "The threshold value for the slope of the mean pixel values.\n" + \
                                       "Slope values below the threshold are considered passing criteria for stopping FIB patterning."
    
    MATCH_SCORE_THRESHOLD_LABEL = "The threshold value for the image match score.\n" + \
                                  "Match scores below the threshold are considered passing criteria for stopping FIB patterning.\n" + \
                                  "Match scores are derived from comparing the first image to the last image in a batch of images.\n" + \
                                  "Low match scores indicate better matches.\n" + \
                                  "During milling (a dynamic state), the match scores will increase."

    MAXIMUM_PIXELS_THRESHOLD_LABEL = "The threshold for the foreground metric; values at or below it are passing criteria for stopping FIB patterning.\n" + \
                                     "For the white-pixel methods this is the percentage of white pixels in the binary image.\n" + \
                                     "For the Top-Hat Foreground Energy method this is the continuous foreground-energy level (≈2.5-3.0, not a 0-100% count) -\n" + \
                                     "see the Binarization Method tooltip, and re-check this value whenever you change method."
    
    CONFIRMATION_ROUNDS_LABEL = "The number of consecutive rounds that must meet the passing criteria \n" + \
                                "before confirming the FIB patterning is complete."
    
    SHOW_GRAYSCALE_IMAGES_CHECKBOX = "Toggle to show grayscale RTM images in the image display area.\n" + \
                                     "When unchecked, the binary / foreground map is shown instead of grayscale RTM images."
    
    SAVE_DATA_CHECKBOX = "Toggle to save analysis data to the script root directory."

    # Pattern results group box tool tips

    MEAN_SLOPE_CHECKBOX = "Toggle to include mean pixel value slope as a criterion for confirming FIB patterning completion."

    MATCH_SCORE_CHECKBOX = "Toggle to include match score as a criterion for confirming FIB patterning completion."

    PERCENT_PIXELS_CHECKBOX = "Toggle to include percent pixels as a criterion for confirming FIB patterning completion."

    # Settings dialog tool tips

    ACQUISITION_DELAY_SECONDS_LABEL = "The number of seconds to delay before acquiring RTM images for analysis.\n" + \
                                      "The delay is intended to accommodate data processing (may not be necessary)."
    
    PERCENT_DIFFERENCE_THRESHOLD_LABEL = "The percentage drop in mean pixel value that will disengage the processing delay.\n" + \
                                         "If the mean pixel value drops by this percentage (and match scores have not increased\n" + \
                                         "above the specified threshold), then the results evaluation delay is disengaged and\n" + \
                                         "values will be evaluated for stopping FIB patterning."

    GAUSSIAN_BLUR_SIGMA_LABEL = "The sigma value for Gaussian blur applied to RTM images before analysis (background subtraction)."

    APPLY_DILATION_LABEL = "Toggle to apply a dilation morphological operation to the RTM images before analysis."

    THRESHOLD_NUM_CLASSES_LABEL = "The number of classes for multi-Otsu thresholding applied to RTM images before analysis (binarization)."

    BINARIZATION_METHOD_LABEL = "How the foreground is isolated for the percent-pixels completion metric.\n" + \
                                "The first three freeze one multi-Otsu class boundary captured once (no drift):\n\n" + \
                                "Brightest class: only the brightest pixels count - can miss faint, slightly-grey features.\n" + \
                                "Middle boundary: balanced capture of faint mid-grey features (default).\n" + \
                                "Low boundary: most inclusive; captures the most but includes more noise.\n\n" + \
                                "Top-Hat Foreground Energy: a morphological white top-hat (set the Top-Hat Radius)\n" + \
                                "that is agnostic to the background level - it isolates bright, sharply-textured\n" + \
                                "features even when the background is dark vs grey across patterns. It reports a\n" + \
                                "CONTINUOUS energy (~2-16), not a 0-100% count, so set Maximum Pixels to ~2.5-3.0.\n\n" + \
                                "Any change of method shifts the white-pixel % scale, so re-check the Maximum Pixels threshold."

    TOPHAT_RADIUS_LABEL = "Disk radius (px) for the Top-Hat Foreground Energy method = the largest foreground\n" + \
                          "feature scale to keep. Features smaller than this survive; the local/static background\n" + \
                          "(incl. a grid bar) is removed. Larger = captures bigger structures but more background;\n" + \
                          "smaller = only fine detail. Used by the Top-Hat metric and by Match On Foreground Map."

    MATCH_ON_FOREGROUND_LABEL = "Compute the match score on the normalized white top-hat foreground map (uses the\n" + \
                                "Top-Hat Radius) instead of the grayscale image. This makes the match score track\n" + \
                                "structural foreground change and ignore background / brightness drift. Works with\n" + \
                                "ANY binarization method (independent of the percent-pixels metric). The match-score\n" + \
                                "scale differs from grayscale, so RE-CHECK the Match Score threshold when enabling this."

    FOREGROUND_COMPLETION_MODE_LABEL = "How the foreground (percent-pixels / Top-Hat energy) completion criterion decides 'done'.\n\n" + \
                                       "Absolute (threshold): foreground value must fall below the Maximum Pixels threshold (original behavior).\n\n" + \
                                       "Relative drop + plateau: GRID-BAR IMMUNE. Done only when the foreground energy has both fallen to\n" + \
                                       "<= Energy Drop Fraction of its start-of-monitoring value AND plateaued (its slope ~ 0). A constant\n" + \
                                       "static feature (e.g. a grid bar) cancels out of both tests, so you don't have to raise an absolute\n" + \
                                       "threshold to see over it. Uses the Energy Drop Fraction and Energy Slope Threshold below."

    ENERGY_DROP_FRACTION_LABEL = "Relative drop + plateau mode only: the foreground energy must fall to at or below this fraction of\n" + \
                                 "its start-of-monitoring value before the foreground criterion can pass (e.g. 0.25 = dropped to 25%).\n" + \
                                 "Lower = stricter (more material must be removed)."

    ENERGY_SLOPE_THRESHOLD_LABEL = "Relative drop + plateau mode only: the |slope| of the foreground energy history below which it is\n" + \
                                   "considered to have plateaued (stopped dropping). Its scale tracks the energy scale, so TUNE THIS on-tool;\n" + \
                                   "too high completes early, too low can be blocked by noise (batch-averaging and confirmation rounds help)."

    SLOPE_METHOD_LABEL = "The method for calculating the slope of mean pixel values and match scores.\n\n" + \
                         "Gradient: Calculates the slope between the last n data points in the batch.\n" + \
                         "Linear Regression: Calculates the slope using linear regression between the last n data points in the batch."
    
    NUM_POINTS_FOR_SLOPE_LABEL = "The number of data points to use for calculating the slope of mean pixel values and match scores.\n" + \
                                 "Only applicable if 'Gradient' method is selected for slope calculation."
    
    LINEAR_REGRESSION_FIT_POINTS_LABEL = "The number of data points to use for calculating the slope of mean pixel values and match scores using linear regression.\n" + \
                                         "Only applicable if 'Linear Regression' method is selected for slope calculation."
    
    ASPECT_RATIO_THRESHOLD_LABEL = "The threshold for the aspect ratio of the patterns.\n" + \
                                   "If the aspect ratio of the patterns are below this threshold, then pattern monitoring will remain idle.\n" + \
                                   "This is designed to detect stress relief cut patterns from AutoTEM Cryo, which are often quite narrow."

    MIN_PATTERN_SPLITS_LABEL = "The minimum number of regions the RTM image is divided into for image analysis.\n" + \
                               "This applies to the crop rectangle area of the RTM image.\n" + \
                               "For example, if the value is 2, then the RTM image is divided into 4 sub-regions (two rows and two columns) for analysis.\n\n" + \
                               "The cropped area of the RTM image is divided into a grid of sub-regions. The mean pixel values and match scores are calculated for each sub-region.\n" + \
                               "The highest mean pixel value and match score among the sub-regions are used for the results evaluation algorithm.\n" + \
                               "Higher number of sub-regions can increase sensitivity to small changes in the patterns, but this may result in values not reaching their thresholds."

    TARGET_TILE_SIZE_LABEL = "The target tile size (in pixels) for the RTM sub-regions (i.e., 100x100 pixel tiles).\n" + \
                             "The actual number of sub-regions is calculated based on the size of the crop area, the target tile size, and the minimum number of sub-regions.\n" +\
                             "To reduce analysis sensitivity, increase the target tile size and reduce the minimum sub-region number."

    # Contrast/Brightness Calibration tool tips

    CB_AUTO_ON_START_LABEL = "Automatically balance detector contrast/brightness once when patterning starts,\n" + \
                             "then hold it static for the rest of the session (so the analysis criteria are not disrupted).\n" + \
                             "If not enabled, the user may need to manually optimize the contrast and brightness values of the live RTM data."

    CB_WHITE_LEVEL_LABEL = "FALLBACK detector full-scale / saturation ceiling (raw pixel value) used to detect clipping.\n" + \
                           "Calibration normally AUTO-DETECTS this from the imaging bit depth (2^bits - 1, e.g. 255 for\n" + \
                           "8-bit, 65535 for 16-bit); this value is only used when the bit depth cannot be read. The\n" + \
                           "first calibration logs the white level it actually used plus the observed pixel min/max."

    CB_TARGET_MEDIAN_FRACTION_LABEL = "Desired median image brightness as a fraction of full-scale (0-1).\n" + \
                                      "The controller nudges brightness toward this target when the image is not clipping."

    CB_TARGET_CONTRAST_SPAN_LABEL = "Desired contrast: the robust occupied span (p2-p98 of the pixels) as a\n" + \
                                    "fraction of full-scale (0-1). The controller adjusts detector contrast to\n" + \
                                    "fill this much of the range, then holds it static for the session.\n" + \
                                    "A single hot/dead pixel does not affect it (percentile-based, not min/max).\n" + \
                                    "Raising this fills more aggressively; the contrast servo will not push past\n" + \
                                    "the Max White/Black Clip limits, so use those to permit (or forbid) clipping."

    CB_MAX_WHITE_CLIP_FRACTION_LABEL = "Maximum acceptable fraction of pixels clipped at the ceiling (white).\n" + \
                                       "Set independently from the black limit (e.g. allow more white clipping than black).\n" + \
                                       "Acts as the contrast servo's clipping budget: raise it to deliberately allow\n" + \
                                       "clipping (which can aid binarization); lower it to keep highlights off the rail."

    CB_MAX_BLACK_CLIP_FRACTION_LABEL = "Maximum acceptable fraction of pixels clipped at the floor (black).\n" + \
                                       "Set independently from the white limit."

    CB_MIN_BOUND_LABEL = "Lower clamp for normalized detector contrast/brightness writes (0-1).\n" + \
                         "The controller will never set CB below this value."

    CB_MAX_BOUND_LABEL = "Upper clamp for normalized detector contrast/brightness writes (0-1).\n" + \
                         "The controller will never set CB above this value."

    CB_MAX_ITERATIONS_LABEL = "Maximum closed-loop adjustment steps before locking at the best setting reached.\n" + \
                              "Higher allows finer convergence but takes longer."

    CB_SETTLE_SECONDS_LABEL = "Time to wait after each contrast/brightness change before re-imaging.\n" + \
                              "Increase this if your RTM frames are slow to form after a change."

    CB_FRAMES_PER_MEASUREMENT_LABEL = "Number of RTM frames averaged per measurement to reduce noise.\n" + \
                                      "Independent of the analysis batch size, so calibration stays fast."


class MainWindowStyles:

    @staticmethod
    def window() -> str:
        return f"""
            QMainWindow {{
                background-color: {StyleColors.MAIN_BG}
            }}
        """


class DialogStyles:

    @staticmethod
    def modal() -> str:
        return f"""
            QDialog {{
                background-color: {StyleColors.MAIN_BG};
            }}
        """


class LabelStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                color: {StyleColors.TEXT_PRIMARY};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_LARGE};
            }}
        """
    
    @staticmethod
    def settings() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                color: {StyleColors.TEXT_PRIMARY};
                margin-left: {StyleDimensions.MARGIN};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_LARGE};
            }}
        """
    
    @staticmethod
    def result_match() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                font-weight: bold;
                color: "#38003c";
                background-color: {StyleColors.RESULT_MATCH};
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                margin-top: {StyleDimensions.RESULT_LABEL_MARGIN};
                margin-bottom: {StyleDimensions.RESULT_LABEL_MARGIN};
            }}
        """
    
    @staticmethod
    def result_label() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                color: {StyleColors.TEXT_PRIMARY};
                margin-top: {StyleDimensions.RESULT_LABEL_MARGIN};
                margin-bottom: {StyleDimensions.RESULT_LABEL_MARGIN};
            }}
        """
    
    @staticmethod
    def result_default() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                font-weight: bold;
                color: {StyleColors.TEXT_PRIMARY};
                margin-top: {StyleDimensions.RESULT_LABEL_MARGIN};
                margin-bottom: {StyleDimensions.RESULT_LABEL_MARGIN};
            }}
        """
    
    @staticmethod
    def status_bar() -> str:
        return f"""
            QLabel {{
                font-size: {StyleDimensions.FONT_SIZE_LARGE};
                color: {StyleColors.TEXT_PRIMARY};
                margin-left: {StyleDimensions.MARGIN};
                padding-left: {StyleDimensions.PADDING};
            }}
        """


class GroupBoxStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QGroupBox {{
                /* border: 1px solid {StyleColors.INPUT_BORDER}; */
                border-radius: {StyleDimensions.BORDER_RADIUS_LARGE};
                background-color: {StyleColors.GROUPBOX_BG};
                margin-top: {StyleDimensions.MARGIN};
                margin-left: {StyleDimensions.MARGIN};
                margin-right: {StyleDimensions.MARGIN};
                margin-bottom: {StyleDimensions.MARGIN};
                padding-top: {StyleDimensions.PADDING};
                padding-left: {StyleDimensions.PADDING};
                padding-right: {StyleDimensions.PADDING};
                padding-bottom: {StyleDimensions.PADDING};
            }}
        """

    @staticmethod
    def with_title() -> str:
        return f"""
            QGroupBox {{
                /* border: 1px solid {StyleColors.INPUT_BORDER}; */
                border-radius: {StyleDimensions.BORDER_RADIUS_LARGE};
                background-color: {StyleColors.GROUPBOX_BG};
                margin-top: {StyleDimensions.MARGIN};
                margin-left: {StyleDimensions.MARGIN};
                margin-right: {StyleDimensions.MARGIN};
                margin-bottom: {StyleDimensions.MARGIN};
                padding-top: 30px;
                padding-left: {StyleDimensions.PADDING};
                padding-right: {StyleDimensions.PADDING};
                padding-bottom: {StyleDimensions.PADDING};
                font-weight: bold;
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                color: {StyleColors.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: {StyleDimensions.MARGIN};
                margin-top: 4px;
                padding: 0 10px;
                background-color: {StyleColors.GROUPBOX_BG};
            }}
        """
    
    @staticmethod
    def plot() -> str:
        return f"""
            QGroupBox {{
                border: none;
                border-radius: {StyleDimensions.BORDER_RADIUS_LARGE};
                background-color: {StyleColors.GROUPBOX_BG};
                margin-top: {StyleDimensions.MARGIN};
                margin-left: {StyleDimensions.MARGIN};
                margin-right: {StyleDimensions.MARGIN};
                margin-bottom: {StyleDimensions.MARGIN};
                padding-top: {StyleDimensions.PADDING};
                padding-left: {StyleDimensions.PADDING};
                padding-right: {StyleDimensions.PADDING};
                padding-bottom: {StyleDimensions.PADDING};
            }}
        """
    
    @staticmethod
    def settings() -> str:
        return f"""
            QGroupBox {{
                border: 1px solid {StyleColors.INPUT_BORDER};
                border-radius: {StyleDimensions.BORDER_RADIUS_LARGE};
                background-color: {StyleColors.GROUPBOX_BG};
                margin-top: {StyleDimensions.MARGIN};
                margin-left: {StyleDimensions.MARGIN};
                margin-right: {StyleDimensions.MARGIN};
                margin-bottom: {StyleDimensions.MARGIN};
                padding-top: 30px;
                padding-left: {StyleDimensions.PADDING};
                padding-right: {StyleDimensions.PADDING};
                padding-bottom: {StyleDimensions.PADDING};
                font-weight: bold;
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                color: {StyleColors.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: {StyleDimensions.MARGIN};
                margin-top: 4px;
                padding: 0 10px;
                background-color: {StyleColors.GROUPBOX_BG};
            }}
        """


class ButtonStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QPushButton {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                margin: {StyleDimensions.MARGIN};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
            QPushButton:hover {{
                background-color: {StyleColors.BUTTON_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {StyleColors.BUTTON_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {StyleColors.BUTTON_DISABLED};
                color: {StyleColors.TEXT_DISABLED};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
        """


class DoubleSpinBoxStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QDoubleSpinBox {{
                background-color: {StyleColors.INPUT_BG};
                color: {StyleColors.TEXT_PRIMARY};
                border: 1px solid {StyleColors.INPUT_BORDER};
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
            QDoubleSpinBox:focus {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QDoubleSpinBox:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 0px;
                height: 0px;
            }}
        """
    
    @staticmethod
    def auto_adjusted() -> str:
        return f"""
            QDoubleSpinBox {{
                background-color: {StyleColors.INPUT_BG};
                color: {StyleColors.TEXT_PRIMARY};
                border: 1px solid {StyleColors.INPUT_BORDER_AUTO};
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
            QDoubleSpinBox:focus {{
                border: 1px solid {StyleColors.INPUT_BORDER_AUTO};
            }}
            QDoubleSpinBox:hover {{
                border: 1px solid {StyleColors.INPUT_BORDER_AUTO};
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 0px;
                height: 0px;
            }}
        """


class CheckBoxStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QCheckBox {{
                color: {StyleColors.TEXT_PRIMARY};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                spacing: {StyleDimensions.CHECKBOX_SPACING};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                border: 1px solid {StyleColors.INPUT_BORDER};
                background-color: {StyleColors.INPUT_BG};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox::indicator:checked {{
                background-color: {StyleColors.BUTTON_HOVER};
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox:disabled {{
                color: {StyleColors.TEXT_DISABLED};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {StyleColors.BUTTON_DISABLED};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
        """
    
    @staticmethod
    def result_checkbox() -> str:
        return f"""
            QCheckBox {{
                color: {StyleColors.TEXT_PRIMARY};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                spacing: {StyleDimensions.CHECKBOX_SPACING};
                margin-top: {StyleDimensions.RESULT_LABEL_MARGIN};
                margin-bottom: {StyleDimensions.RESULT_LABEL_MARGIN};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                border: 1px solid {StyleColors.INPUT_BORDER};
                background-color: {StyleColors.INPUT_BG};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox::indicator:checked {{
                background-color: {StyleColors.BUTTON_HOVER};
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox:disabled {{
                color: {StyleColors.TEXT_DISABLED};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {StyleColors.BUTTON_DISABLED};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
        """
    
    @staticmethod
    def settings_dialog() -> str:
        return f"""
            QCheckBox {{
                color: {StyleColors.TEXT_PRIMARY};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                border: 1px solid {StyleColors.INPUT_BORDER};
                background-color: {StyleColors.INPUT_BG};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox::indicator:checked {{
                background-color: {StyleColors.BUTTON_HOVER};
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QCheckBox:disabled {{
                color: {StyleColors.TEXT_DISABLED};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {StyleColors.BUTTON_DISABLED};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
        """


class RadioButtonStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QRadioButton {{
                color: {StyleColors.TEXT_PRIMARY};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
                spacing: {StyleDimensions.CHECKBOX_SPACING};
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 1px solid {StyleColors.INPUT_BORDER};
                background-color: {StyleColors.INPUT_BG};
            }}
            QRadioButton::indicator:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QRadioButton::indicator:checked {{
                background-color: {StyleColors.BUTTON_HOVER};
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QRadioButton:disabled {{
                color: {StyleColors.TEXT_DISABLED};
            }}
            QRadioButton::indicator:disabled {{
                background-color: {StyleColors.BUTTON_DISABLED};
            }}
            QToolTip {{
                background-color: {StyleColors.BUTTON_BG};
                color: {StyleColors.TEXT_PRIMARY};
                /*border: 1px solid #ffffff;*/
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
        """


class ComboBoxStyles:

    # Arrow icon path
    _arrow_path = str(Path(__file__).parent.parent / "script_assets" / "down_arrow_white.png").replace("\\", "/")

    @staticmethod
    def default() -> str:
        return f"""
            QComboBox {{
                background-color: {StyleColors.INPUT_BG};
                color: {StyleColors.TEXT_PRIMARY};
                border: 1px solid {StyleColors.INPUT_BORDER};
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
                padding: {StyleDimensions.PADDING};
                font-size: {StyleDimensions.FONT_SIZE_NORMAL};
            }}
            QComboBox:hover {{
                border: 1px solid {StyleColors.BUTTON_HOVER};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 40px;
            }}
            QComboBox::down-arrow {{
                image: url({ComboBoxStyles._arrow_path});
                width: 14px;
                height: 14px;
                border-left: 8px solid transparent;
                border-right: 8px solid transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {StyleColors.INPUT_BG};
                color: {StyleColors.TEXT_PRIMARY};
                border: 1px solid {StyleColors.INPUT_BORDER};
                /* border-radius: {StyleDimensions.BORDER_RADIUS_SMALL}; */
                outline: none;
                selection-background-color: transparent;
                selection-color: {StyleColors.TEXT_PRIMARY};
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 8px;
                border: 1px solid {StyleColors.INPUT_BORDER};
                outline: none;
            }}
            QComboBox QAbstractItemView::item:hover {{
                padding: 8px 8px;
                background-color: {StyleColors.BUTTON_BG};
                border: 1px solid {StyleColors.INPUT_BORDER};
                /* border-radius: {StyleDimensions.BORDER_RADIUS_SMALL}; */
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: transparent;
                border: 1px solid {StyleColors.INPUT_BORDER};
                /* border-radius: {StyleDimensions.BORDER_RADIUS_SMALL}; */
            }}
            QComboBox QAbstractItemView::item:selected:hover {{
                background-color: {StyleColors.BUTTON_BG};
                border: 1px solid {StyleColors.INPUT_BORDER};
                /* border-radius: {StyleDimensions.BORDER_RADIUS_SMALL}; */
            }}
        """


class StatusBarStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QStatusBar {{
                background-color: {StyleColors.MAIN_BG};
                border: none;
            }}
            QStatusBar::item {{
                background-color: transparent;
                border: none;
            }}
        """


class ScrollAreaStyles:

    @staticmethod
    def default() -> str:
        return f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {StyleColors.INPUT_BG};
                width: 12px;
                margin: 0px;
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
            }}
            QScrollBar::handle:vertical {{
                background: {StyleColors.BUTTON_BG};
                min-height: 24px;
                border-radius: {StyleDimensions.BORDER_RADIUS_SMALL};
            }}
            QScrollBar::handle:vertical:hover {{
                background: {StyleColors.BUTTON_HOVER};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """


class AppStyles:

    Colors = StyleColors
    Dimensions = StyleDimensions
    MainWindow = MainWindowStyles
    GroupBox = GroupBoxStyles
    Label = LabelStyles
    Button = ButtonStyles
    SpinBox = DoubleSpinBoxStyles
    AppText = WindowText
    AppToolTips = ToolTips
    CheckBox = CheckBoxStyles
    RadioButton = RadioButtonStyles
    ComboBox = ComboBoxStyles
    Dialog = DialogStyles
    StatusBar = StatusBarStyles
    ScrollArea = ScrollAreaStyles