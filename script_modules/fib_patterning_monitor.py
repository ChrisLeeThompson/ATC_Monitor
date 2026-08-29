"""
FIB Patterning Monitor Module

This module handles interaction with the FIB for monitoring
Real-Time Monitor (RTM) data during patterning.

Key responsibilities:
- Safe AutoScript import handling
- Microscope connection management
- Pattern validation (type, count, aspect ratio)
- Active device verification
- RTM data acquisition setup
"""

import importlib
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# ===========================
# Safe AutoScript Imports
# ===========================

try:
    from autoscript_sdb_microscope_client.sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.enumerations import (
        PatterningState,
        RtmMode,
        RtmCoordinateSystem,
        ScanningFilterType
    )
    from autoscript_sdb_microscope_client.dynamic_objects import (
        RectanglePattern,
        RegularCrossSectionPattern,
        CleaningCrossSectionPattern
    )
    from autoscript_toolkit.template_matchers import GetRtmPositionSettings
    # ORC client endpoint -- needed to neutralize its keep-alive (vitality-check)
    # thread, which is not thread-safe with normal calls (see
    # _suppress_autoscript_vitality_check below).
    from autoscript_core.orc.engines import ClientEndpoint
    AUTOSCRIPT_AVAILABLE = True
    logger.info("AutoScript successfully imported")
except ImportError:
    AUTOSCRIPT_AVAILABLE = False
    logger.warning("AutoScript not available", exc_info=True)

    # Define names to prevent NameError
    SdbMicroscopeClient = None
    RectanglePattern = None
    RegularCrossSectionPattern = None
    CleaningCrossSectionPattern = None
    GetRtmPositionSettings = None
    ClientEndpoint = None


# GetRtmDataSettings carries wait_for_next_data, the flag that makes get_data()
# block until genuinely new data exists ("the user does not want to process the
# same data twice"). It is the authoritative frame-advance signal: without it,
# Auto CB has to infer staleness from pixel statistics, which cannot tell a
# frozen buffer from a saturated scene whose frames legitimately do not change
# (the 2026-08-23 "RTM frames not advancing" false positives).
#
# Imported separately and tolerantly: the reference manual documents it under
# _structures_rtm, but the public re-export path varies by SDK build (the
# neighboring GetRtmPositionSettings ships from autoscript_toolkit), and a
# missing symbol must degrade to a bare get_data() rather than take the whole
# AutoScript import down with it.
GetRtmDataSettings = None
if AUTOSCRIPT_AVAILABLE:
    for _rtm_settings_module in ("autoscript_sdb_microscope_client.structures",
                                 "autoscript_sdb_microscope_client._structures_rtm",
                                 "autoscript_toolkit.template_matchers"):
        try:
            GetRtmDataSettings = getattr(
                importlib.import_module(_rtm_settings_module), "GetRtmDataSettings"
            )
            logger.info(f"GetRtmDataSettings imported from {_rtm_settings_module}")
            break
        except (ImportError, AttributeError):
            continue
    if GetRtmDataSettings is None:
        logger.warning(
            "GetRtmDataSettings not available in this AutoScript build - Auto CB "
            "falls back to statistical stale-frame detection"
        )


# acquire_rtm_data message for "the RTM has no new data" -- a legitimate answer
# from wait_for_next_data (the patterning job ended without collecting more),
# not an acquisition failure. Auto CB treats it as the frozen-buffer verdict.
RTM_NO_NEW_DATA = "No new RTM data"

# acquire_rtm_data message confirming the returned frame is one the caller has
# not seen before (wait_for_next_data was requested and honored). Auto CB skips
# its statistical stale-frame check entirely on this message -- the SDK has
# already answered the question the check exists to guess at.
RTM_FRESH_DATA = "RTM data acquired successfully (new frame)"


# ===========================
# Data Structures
# ===========================

@dataclass
class PatternInfo:
    """Information about a FIB pattern."""
    pattern_id: str
    pattern_type: str               # "RECTANGLE" (RCS and CCS are not supported)
    width: float                    # Pattern width in meters
    height: float                   # Pattern height in meters
    aspect_ratio: float             # width / height
    aspect_ratio_is_valid: bool     # Passes aspect ratio threshold
    scan_direction: str | None = None  # e.g. "BottomToTop"; None if unavailable
    rotation: float = 0.0           # pattern.rotation in radians, CW-positive; 0.0 if unavailable


@dataclass
class ValidationResult:
    """Result of pattern validation."""
    success: bool
    patterns: list[PatternInfo]
    message: str


# ===========================
# AutoScript Availability
# ===========================

def check_autoscript_available() -> tuple[bool, str]:
    """
    Check if AutoScript is available.
    
    :return: (is_available, message)
    """
    if AUTOSCRIPT_AVAILABLE:
        return True, "AutoScript available"
    else:
        return False, "AutoScript not available"


# ===========================
# Microscope Connection
# ===========================

# A value (ms) large enough that the vitality-check thread effectively never
# fires again during a session (~11.5 days).
_VITALITY_CHECK_INTERVAL_MS = 1_000_000_000
_vitality_check_suppressed = False


def _suppress_autoscript_vitality_check():
    """
    Neutralize the AutoScript client's keep-alive ("vitality check") thread.

    Works around an observed transport race. The ORC ClientEndpoint starts a
    daemon thread that sends a keep-alive on the transport socket every 250 ms.
    That socket (FrameSocket) is documented as not thread-safe -- "DO NOT
    send/receive simultaneously from different threads" -- so those keep-alive
    sends can overlap our call receives. In this application that overlap
    correlated with intermittent native faults, and suppressing the keep-alive
    resolved them. The overlap is likelier when calls are slow (e.g. FIB beam
    off), but is not limited to that case.

    We call AutoScript constantly, so a dropped connection is still detected and
    recovered on the next real call (perform_call); the proactive keep-alive is
    not needed for our usage. Applied in two best-effort, reversible layers and
    must run before microscope.connect():
      1) push VITALITY_CHECK_INTERVAL far out (documented constant), and
      2) replace the private thread-starter with a no-op so it never runs.
    """
    global _vitality_check_suppressed
    if ClientEndpoint is None:
        return
    try:
        ClientEndpoint.VITALITY_CHECK_INTERVAL = _VITALITY_CHECK_INTERVAL_MS
    except Exception:
        logger.warning("Could not raise AutoScript vitality-check interval", exc_info=True)
    try:
        # Name-mangled private method; no-op it so no keep-alive thread starts.
        setattr(ClientEndpoint, "_ClientEndpoint__start_vitality_check_thread",
                lambda self: None)
        if not _vitality_check_suppressed:
            logger.info(
                "AutoScript vitality-check (keep-alive) thread disabled to avoid "
                "overlapping sends and receives on the transport socket"
            )
            _vitality_check_suppressed = True
    except Exception:
        logger.warning("Could not disable AutoScript vitality-check thread", exc_info=True)


def _log_post_connect_state():
    """
    Log thread/suppression state right after connect, to verify the vitality
    suppression actually took (and to surface any unexpected background threads).
    """
    try:
        import threading
        names = [t.name for t in threading.enumerate()]
        patched = (
            ClientEndpoint is not None
            and getattr(ClientEndpoint._ClientEndpoint__start_vitality_check_thread,
                        "__name__", "") == "<lambda>"
        )
        interval = ClientEndpoint.VITALITY_CHECK_INTERVAL if ClientEndpoint is not None else None
        logger.info(
            f"Post-connect: {len(names)} live threads {names}; "
            f"vitality_suppressed={patched}, vitality_interval={interval}"
        )
    except Exception:
        logger.warning("Could not log post-connect state", exc_info=True)


def connect_to_microscope(host: str = "localhost") -> tuple[object | None, str]:
    """
    Connect to the microscope.

    :param host: Hostname or IP address (default: "localhost")
    :return: (microscope_object or None, message)
    """
    if not AUTOSCRIPT_AVAILABLE:
        return None, "AutoScript not available"

    try:
        logger.info(f"Attempting to connect to microscope at {host}...")
        microscope = SdbMicroscopeClient()
        # Must run before connect() -- the vitality-check thread is started during
        # connect() and reads the interval once at that point.
        _suppress_autoscript_vitality_check()
        microscope.connect(host)
        logger.info("Successfully connected to microscope")
        _log_post_connect_state()
        return microscope, "Connected to microscope"
    except Exception:
        logger.error("Failed to connect to microscope", exc_info=True)
        return None, "Connection failed"


def disconnect_from_microscope(microscope: object) -> str:
    """
    Disconnect from the microscope.
    
    :param microscope: Microscope object
    :return: Status message
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return "No microscope to disconnect"
    
    try:
        microscope.disconnect()
        logger.info("Disconnected from microscope")
        return "Disconnected from microscope"
    except Exception:
        logger.warning("Error during disconnect", exc_info=True)
        return "Disconnect error"


# ===========================
# Device Validation
# ===========================

def validate_active_device(microscope: object) -> tuple[bool | None, str]:
    """
    Validate that the FIB is the active imaging device.

    The FIB must be device 2 for RTM monitoring to work properly.

    Tri-state result so callers can distinguish a genuine communication
    failure from the benign "FIB is not the active device" state:
      - True  : FIB is the active imaging device
      - False : a different device is active (benign; operator must select FIB)
      - None  : the check could not be performed (microscope unavailable or a
                communication exception)

    :param microscope: Connected microscope object
    :return: (is_valid, message) where is_valid is True/False/None
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, "Microscope not available"

    try:
        active_device = microscope.imaging.get_active_device()
        logger.debug(f"Active imaging device: {active_device}")

        if active_device == 2:
            return True, "FIB is active imaging device"
        else:
            return False, (
                f"FIB is not the active imaging device (current: {active_device}). "
                "Please select the FIB quadrant in Microscope Control."
            )
    except Exception:
        logger.error("Error checking active device", exc_info=True)
        return None, "Device check failed"


# ===========================
# Pattern Validation
# ===========================

def get_pattern_info(pattern: object) -> PatternInfo | None:
    """
    Extract information from a pattern object.
    
    :param pattern: Pattern object from microscope
    :return: PatternInfo or None if pattern type not supported
    """
    if not AUTOSCRIPT_AVAILABLE:
        return None
    
    # Determine pattern type. Only rectangles are supported; Regular Cross
    # Section (RCS) and Cleaning Cross Section (CCS) are unsupported and are
    # rejected up front in validate_patterns.
    pattern_type = None
    if isinstance(pattern, RectanglePattern):
        pattern_type = "RECTANGLE"
    else:
        return None  # Unsupported pattern type
    
    # Extract dimensions
    try:
        width = pattern.width
        height = pattern.height
        aspect_ratio = width / height if height > 0 else 0.0

        # Scan direction (e.g. "BottomToTop"). getattr guards pattern types that
        # don't expose it; the value is used only to highlight a crop-box edge.
        raw_scan_direction = getattr(pattern, "scan_direction", None)
        scan_direction = str(raw_scan_direction) if raw_scan_direction is not None else None

        # Rotation (radians, clockwise-positive per AutoScript "pattern rotation angle").
        # AutoTEM Cryo rotates patterns by -180 deg, which flips the on-screen scan
        # direction; used to choose which crop-box edge to highlight.
        raw_rotation = getattr(pattern, "rotation", None)
        try:
            rotation = float(raw_rotation) if raw_rotation is not None else 0.0
        except (TypeError, ValueError):
            rotation = 0.0

        return PatternInfo(
            pattern_id=pattern.id,
            pattern_type=pattern_type,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            aspect_ratio_is_valid=True,  # Will be set by validate_patterns
            scan_direction=scan_direction,
            rotation=rotation
        )
    except Exception:
        logger.warning("Error extracting pattern info", exc_info=True)
        return None


def validate_patterns(
    microscope: object, 
    aspect_ratio_threshold: float = 0.3
) -> ValidationResult:
    """
    Validate patterns for monitoring.
    
    Checks:
    1. Pattern count (must be 1 or 2)
    2. Pattern types (must be RECTANGLE; RCS and CCS are not supported)
    3. Aspect ratios (must be >= threshold to filter stress relief cuts)
    
    :param microscope: Connected microscope object
    :param aspect_ratio_threshold: Minimum aspect ratio for valid patterns (default: 0.3)
    :return: ValidationResult with success status, pattern list, and message
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return ValidationResult(
            success=False,
            patterns=[],
            message="Microscope not available"
        )
    
    try:
        # Get all patterns from the patterning file
        all_patterns = microscope.patterning.get_patterns()
        logger.debug(f"Found {len(all_patterns)} pattern(s) in patterning file")
        
        # Filter and extract info from supported patterns
        pattern_infos = []
        unsupported_types = []
        
        for pattern in all_patterns:
            # Check for unsupported RCS patterns (monitoring of Regular Cross
            # Section is disabled - endpoint detection is tuned for rectangles)
            if isinstance(pattern, RegularCrossSectionPattern):
                unsupported_types.append("RegularCrossSectionPattern (RCS)")
                continue

            # Check for unsupported CCS patterns
            if isinstance(pattern, CleaningCrossSectionPattern):
                unsupported_types.append("CleaningCrossSectionPattern (CCS)")
                continue

            # Extract info from supported patterns
            info = get_pattern_info(pattern)
            if info:
                pattern_infos.append(info)
            else:
                unsupported_types.append(type(pattern).__name__)
        
        # Check for unsupported pattern types
        if unsupported_types:
            return ValidationResult(
                success=False,
                patterns=[],
                message=f"Unsupported pattern types detected: {', '.join(unsupported_types)}"
            )
        
        # Check pattern count
        num_patterns = len(pattern_infos)
        if num_patterns == 0:
            return ValidationResult(
                success=False,
                patterns=[],
                message="No supported patterns found in patterning file"
            )
        elif num_patterns > 2:
            return ValidationResult(
                success=False,
                patterns=pattern_infos,
                message=f"Too many patterns ({num_patterns}). Maximum supported: 2"
            )
        
        # Check aspect ratios
        invalid_patterns = []
        for info in pattern_infos:
            if info.aspect_ratio < aspect_ratio_threshold:
                info.aspect_ratio_is_valid = False
                width_um = info.width * 1e6
                height_um = info.height * 1e6
                invalid_patterns.append(
                    f"{info.pattern_id} (AR: {info.aspect_ratio:.3f})"
                )
                logger.debug(
                    f"Pattern {info.pattern_id} aspect ratio "
                    f"{info.aspect_ratio:.3f} is below threshold "
                    f"{aspect_ratio_threshold:.2f} "
                    f"({width_um:.2f} µm × {height_um:.2f} µm)"
                )
        
        # If any patterns have invalid aspect ratios, return failure
        if invalid_patterns:
            return ValidationResult(
                success=False,
                patterns=pattern_infos,
                # Every number in a message that reaches the status bar must
                # carry a format spec: the threshold arrives as a raw float
                # from the spinbox/QSettings and renders as e.g.
                # "0.44999999999999996" otherwise (field report 2026-08-21).
                message=f"Pattern(s) with aspect ratio < "
                        f"{aspect_ratio_threshold:.2f}: "
                        f"{', '.join(invalid_patterns)}"
            )
        
        # All validation passed
        # Create summary message for return
        # Micrometres with fixed decimals, not raw metres: raw floats render
        # as "1.0163876069124408e-05 m" in any surface that shows this message.
        pattern_summary = ", ".join([
            f"{info.pattern_type} ({info.width * 1e6:.2f} µm × "
            f"{info.height * 1e6:.2f} µm, AR={info.aspect_ratio:.3f})"
            for info in pattern_infos
        ])
        
        return ValidationResult(
            success=True,
            patterns=pattern_infos,
            message=f"Validated {num_patterns} pattern(s): {pattern_summary}"
        )
        
    except Exception:
        logger.error("Error during pattern validation", exc_info=True)
        return ValidationResult(
            success=False,
            patterns=[],
            message="Pattern validation error"
        )


# ===========================
# RTM Setup
# ===========================

def setup_rtm_monitoring(microscope: object, mode: str = "HIGH_RESOLUTION") -> tuple[bool, str]:
    """
    Setup RTM monitoring with specified resolution mode.
    
    :param microscope: Connected microscope object
    :param mode: RTM mode - "HIGH_RESOLUTION" or "LOW_RESOLUTION" (default: HIGH_RESOLUTION)
    :return: (success, message)
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return False, "Microscope not available"
    
    try:
        # Set RTM mode
        if mode == "HIGH_RESOLUTION":
            microscope.patterning.real_time_monitor.mode = RtmMode.HIGH_RESOLUTION
            logger.info("RTM mode set to HIGH_RESOLUTION")
        elif mode == "LOW_RESOLUTION":
            microscope.patterning.real_time_monitor.mode = RtmMode.LOW_RESOLUTION
            logger.info("RTM mode set to LOW_RESOLUTION")
        else:
            return False, f"Invalid RTM mode: {mode}"
        
        # Restart RTM acquisition
        microscope.patterning.real_time_monitor.restart()
        logger.info("RTM acquisition restarted")
        
        return True, f"RTM monitoring setup complete ({mode})"
        
    except Exception:
        logger.error("Error setting up RTM monitoring", exc_info=True)
        return False, "RTM setup failed"


def acquire_rtm_data(
    microscope: object, wait_for_next_data: bool = False
) -> tuple[object | None, object | None, str]:
    """
    Acquire RTM data and positions from the microscope.

    :param microscope: Connected microscope object
    :param wait_for_next_data: when True, ask the RTM for the next data set
        instead of whatever is currently buffered, so the caller can never
        process the same frame twice. Auto CB uses this as its frame-advance
        signal. The monitoring loop leaves it False: it must never block on a
        pattern that has stopped emitting. Requires GetRtmDataSettings -- the
        call degrades to a plain get_data() when the SDK does not expose it,
        which the caller detects from the returned message.
    :return: (rtm_data, rtm_positions, message). rtm_data is None with the
        message RTM_NO_NEW_DATA when wait_for_next_data found no new frame.
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, None, "Microscope not available"

    try:
        # Get RTM data. With wait_for_next_data the SDK returns None when the
        # patterning job ended without collecting more -- "no new frame", which
        # is reported distinctly so the caller does not read it as a failure.
        if wait_for_next_data and GetRtmDataSettings is not None:
            rtm_data = microscope.patterning.real_time_monitor.get_data(
                GetRtmDataSettings(None, True)
            )
            if rtm_data is None:
                return None, None, RTM_NO_NEW_DATA
        else:
            rtm_data = microscope.patterning.real_time_monitor.get_data()

        # Create position settings for IMAGE_PIXELS coordinate system
        position_settings = GetRtmPositionSettings(None, RtmCoordinateSystem.IMAGE_PIXELS)
        
        # Get RTM positions using the position settings
        rtm_positions = microscope.patterning.real_time_monitor.get_positions(position_settings)

        if wait_for_next_data and GetRtmDataSettings is not None:
            return rtm_data, rtm_positions, RTM_FRESH_DATA
        return rtm_data, rtm_positions, "RTM data acquired successfully"

    except Exception:
        logger.error("Error acquiring RTM data", exc_info=True)
        return None, None, "RTM data acquisition failed"


# ===========================
# Patterning State Check
# ===========================

def check_patterning_state(microscope: object) -> tuple[bool | None, str]:
    """
    Check if patterning is currently running.

    Tri-state result so a transient communication failure is not mistaken for
    "patterning stopped" (which would wipe the monitoring session):
      - True  : patterning is active
      - False : patterning is genuinely not active
      - None  : the state could not be read (microscope unavailable or a
                communication exception)

    :param microscope: Connected microscope object
    :return: (is_running, message) where is_running is True/False/None
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, "Microscope not available"

    try:
        state = microscope.patterning.state
        is_running = (state == PatterningState.RUNNING)

        if is_running:
            return True, "Patterning is active"
        else:
            return False, f"Patterning is not active (state: {state})"

    except Exception:
        logger.error("Error checking patterning state", exc_info=True)
        return None, "State check failed"


def check_fib_beam_on(microscope: object) -> tuple[bool | None, str]:
    """
    Check whether the FIB (ion) beam is turned on.

    The ion beam auto-turns-off after a period of inactivity (~60 min idle).
    Several AutoScript calls (e.g. service.system.name) fault natively when the
    beam is off, so callers should check this first and wait while it is off.

    Tri-state result so a transient communication failure is not mistaken for
    "beam off":
      - True  : the ion beam is on
      - False : the ion beam is off
      - None  : the state could not be read (microscope unavailable or error)

    :param microscope: Connected microscope object
    :return: (is_on, message) where is_on is True/False/None
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, "Microscope not available"

    try:
        is_on = bool(microscope.beams.ion_beam.is_on)
        return is_on, ("Ion beam is on" if is_on else "Ion beam is off")
    except Exception:
        logger.error("Error checking ion beam state", exc_info=True)
        return None, "Beam state check failed"


def stop_patterning(microscope: object) -> tuple[bool, str]:
    """
    Stop the current patterning operation.
    
    :param microscope: Connected microscope object
    :return: (success, message)
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return False, "Microscope not available"
    
    try:
        microscope.patterning.stop()
        logger.info("Patterning stopped")
        return True, "Patterning stopped"
    except Exception:
        logger.error("Error stopping patterning", exc_info=True)
        return False, "Stop patterning failed"


def restart_rtm(microscope: object) -> tuple[bool, str]:
    """
    Restart the real-time monitor acquisition.
    
    This should be called:
    - After initial RTM setup
    - After stopping patterning
    - When patterning transitions from stopped to running
    
    :param microscope: Connected microscope object
    :return: (success, message)
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return False, "Microscope not available"
    
    try:
        microscope.patterning.real_time_monitor.restart()
        logger.info("RTM acquisition restarted")
        return True, "RTM restarted"
    except Exception:
        logger.error("Error restarting RTM", exc_info=True)
        return False, "RTM restart failed"


# ===========================
# Detector Contrast/Brightness
# ===========================

def read_detector_cb(microscope: object) -> tuple[float | None, float | None]:
    """
    Read the current detector contrast and brightness.

    :param microscope: Connected microscope object
    :return: (contrast, brightness) as normalized floats in [0, 1], or
             (None, None) if unavailable or on error.
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, None

    try:
        contrast = float(microscope.detector.contrast.value)
        brightness = float(microscope.detector.brightness.value)
        return contrast, brightness
    except Exception:
        logger.warning("Failed to read detector contrast/brightness", exc_info=True)
        return None, None


def _clamp_to_detector_limits(prop: object, value: float, lo: float, hi: float,
                              cached: tuple[float, float] | None = None) -> float:
    """
    Clamp value to [lo, hi] and, when readable, to the property's own limits.

    AutoScript CB properties may expose a ``.limits`` (min, max); we never write
    outside it. ``cached`` supplies limits read earlier (the calibration loop
    reads them once instead of paying two SDK round-trips per write). If limits
    can't be read we fall back to the configured bounds.
    """
    low, high = lo, hi
    if cached is not None:
        low = max(low, cached[0])
        high = min(high, cached[1])
    else:
        try:
            limits = prop.limits
            low = max(low, float(limits.min))
            high = min(high, float(limits.max))
        except Exception:
            pass
    if high < low:
        high = low
    return min(high, max(low, value))


def read_detector_cb_limits(
    microscope: object,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Read the detector contrast and brightness property limits once, for caching.

    :return: ((c_min, c_max) | None, (b_min, b_max) | None) -- None per channel
             when the limits are unreadable (callers fall back to live reads).
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None, None

    def _limits(prop):
        try:
            limits = prop.limits
            return float(limits.min), float(limits.max)
        except Exception:
            return None

    return (_limits(microscope.detector.contrast),
            _limits(microscope.detector.brightness))


def set_detector_cb(
    microscope: object,
    contrast: float | None = None,
    brightness: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    limits: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None,
) -> tuple[bool, float | None, float | None, str]:
    """
    Write detector contrast and/or brightness, clamped to ``bounds`` and (when
    available) the detector's own property limits, then read the values back.

    Either channel may be None to leave it unchanged.

    :param microscope: Connected microscope object
    :param contrast: Desired normalized contrast in [0, 1], or None to skip
    :param brightness: Desired normalized brightness in [0, 1], or None to skip
    :param bounds: (min, max) clamp applied before the detector's own limits
    :param limits: optional cached (contrast_limits, brightness_limits) from
        read_detector_cb_limits, avoiding two SDK reads per write
    :return: (success, contrast_after, brightness_after, message)
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return False, None, None, "Microscope not available"

    lo, hi = bounds
    c_limits, b_limits = limits if limits is not None else (None, None)
    try:
        if contrast is not None:
            c = _clamp_to_detector_limits(
                microscope.detector.contrast, float(contrast), lo, hi,
                cached=c_limits
            )
            microscope.detector.contrast.value = c
        if brightness is not None:
            b = _clamp_to_detector_limits(
                microscope.detector.brightness, float(brightness), lo, hi,
                cached=b_limits
            )
            microscope.detector.brightness.value = b
    except Exception:
        logger.error("Failed to set detector contrast/brightness", exc_info=True)
        c_after, b_after = read_detector_cb(microscope)
        return False, c_after, b_after, "CB write failed"

    # Read back to confirm the applied values.
    c_after, b_after = read_detector_cb(microscope)
    return True, c_after, b_after, "CB updated"


def read_scanning_bit_depth(microscope: object) -> int | None:
    """
    Read the bit depth of the active imaging beam's live acquisition.

    ``RtmMode.LOW_RESOLUTION`` rides the imaging pipeline, so RTM pixel full-scale is
    ``2**bit_depth - 1``. The auto-CB routine uses this to size its white level from
    the actual detector depth instead of assuming 8-bit (the RTM *resolution* mode is
    orthogonal to bit depth). Tries the active imaging device (electron beam = 1, ion
    beam = 2 per ImagingDevice), defaulting to the ion beam for the FIB.

    :param microscope: Connected microscope object
    :return: bit depth (e.g. 8 or 16), or None if unavailable / on error.
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None

    try:
        try:
            device = microscope.imaging.get_active_device()
        except Exception:
            device = None

        if device == 1:  # ImagingDevice.ELECTRON_BEAM
            beam = microscope.beams.electron_beam
        else:            # ImagingDevice.ION_BEAM (2) or unknown -> FIB default
            beam = microscope.beams.ion_beam

        # bit_depth is a value-object (.value) on some AutoScript builds and a
        # plain int on others (observed on Hydra Bio) -- accept both.
        raw = beam.scanning.bit_depth
        bits = int(getattr(raw, "value", raw))
        return bits if bits > 0 else None
    except Exception:
        logger.warning("Failed to read scanning bit depth", exc_info=True)
        return None


# ===========================
# System Identification
# ===========================

def get_system_name(microscope: object) -> str:
    """
    Read the system name from the microscope.
    
    Used to determine hardware capabilities (e.g., specimen current
    availability). Arctis systems do not support specimen current.
    
    :param microscope: Connected microscope object
    :return: System name string, or empty string if unavailable
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return ""
    
    try:
        name = microscope.service.system.name
        logger.info(f"System name: {name}")
        return name
    except Exception:
        logger.warning("Failed to read system name", exc_info=True)
        return ""


# ===========================
# Specimen Current
# ===========================

def get_specimen_current(microscope: object) -> float | None:
    """
    Read the specimen current from the microscope stage.
    
    :param microscope: Connected microscope object
    :return: Specimen current in amps (float), or None if unavailable
    """
    if not AUTOSCRIPT_AVAILABLE or microscope is None:
        return None
    
    try:
        return microscope.state.specimen_current.value
    except Exception:
        logger.warning("Failed to read specimen current", exc_info=True)
        return None