"""
Worker Parameter Manager Module

Manages live parameter updates during workflow execution with thread-safe
queuing and state-aware application.

Key responsibilities:
- Queue parameter updates (thread-safe)
- Look up each parameter's declared state-reset requirement in PARAMETER_IMPACTS
- Refuse any parameter not declared there
- Apply updates at safe batch boundaries
- Generate user-friendly status messages
- Validate values against the dialog's own spin-box limits

The registry is the module's central rule: a parameter's impact and its bounds
are declared, never inferred from its name. Names carry no meaning here, so a
rename cannot silently reclassify a parameter, and an unregistered name is
dropped rather than applied with whatever handling happens to match it.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto

from script_modules.results_evaluation import (
    StallConfig, evaluate_foreground_stall
)
from script_modules.atc_monitor_parameters import SpinBoxConstraints


logger = logging.getLogger(__name__)

# The dialog's own spin-box limits, reused so the live-update path can never
# accept a value the UI would have clamped. Defaults-only instance: these are
# class-level constants, not operator state.
_SPIN_BOX_CONSTRAINTS = SpinBoxConstraints()


# Display names for machine-generated parameter names in user-facing status
# messages. Fallback for unmapped names: title-cased snake_case.
PARAM_DISPLAY_NAMES = {
    "number_of_images": "Number of Images",
    "analysis_interval_seconds": "Analysis Interval",
    "confirmation_rounds": "Confirmation Rounds",
    "mean_slope_enabled": "Mean Slope",
    "match_score_enabled": "Match Score",
    "percent_pixels_enabled": "Percent Pixels",
    # Per-pattern thresholds, in the terminology the results panel uses. (The
    # percent-pixels control is relabeled per binarization method, so its
    # threshold keeps the binarization-neutral name the spin box carries.)
    **{f"pattern_{idx}_{key}": f"Pattern {idx} {label}"
       for idx in (1, 2) for key, label in (
           ("mean_slope_threshold", "Mean Slope Threshold"),
           ("match_score_threshold", "Match Score Threshold"),
           ("max_pixels_threshold", "Maximum Pixels Threshold"))},
    # Auto CB. Without these the .title() fallback puts machine identifiers on
    # the status bar -- "Cb Max White Clip Fraction updated to 0.01" -- which is
    # the exact string the registry comment below cites as the bug it fixes.
    # Names match the Settings dialog's own row labels so the operator can find
    # the control the message is talking about.
    "auto_cb_on_start": "Auto CB: Auto-Calibrate on Start",
    "cb_recalibrate_every_session": "Auto CB: Recalibrate Every Session",
    "cb_white_level": "Auto CB: Detector White Level",
    "cb_lower_margin": "Auto CB: Lower Margin",
    "cb_upper_margin": "Auto CB: Upper Margin",
    "cb_max_white_clip_fraction": "Auto CB: Max White Clip",
    "cb_max_black_clip_fraction": "Auto CB: Max Black Clip",
    "cb_min_bound": "Auto CB: C/B Lower Bound",
    "cb_max_bound": "Auto CB: C/B Upper Bound",
    "cb_max_iterations": "Auto CB: Max Iterations",
    "cb_settle_seconds": "Auto CB: Settle Time",
    "cb_frames_per_measurement": "Auto CB: Verify Frames",
}


def display_name(param_name: str) -> str:
    """
    Operator-facing name for a parameter, for status and validation messages.

    :param param_name: machine parameter name
    :return: its display name, else a title-cased fallback
    """
    return PARAM_DISPLAY_NAMES.get(
        param_name, param_name.replace('_', ' ').title())


def _fmt_value(value) -> str:
    """
    Render a parameter value for a status-bar message.

    Spinbox float noise must never reach the bar: three 0.05 steps from 0.30
    produce 0.44999999999999996, and interpolating it raw is the exact field
    defect of 2026-08-21. ".6g" renders it as 0.45 while leaving ints, bools,
    and non-numeric values verbatim.
    """
    return format(value, ".6g") if isinstance(value, float) else str(value)


# ===========================
# Enumerations
# ===========================

class ParameterImpact(Enum):
    """
    Classification of how parameter changes affect monitoring state.
    
    Which class a parameter belongs to is declared in PARAMETER_IMPACTS. The
    example names below are the registered keys, so they can be looked up there
    directly -- an unregistered name is refused, so the un-prefixed forms of the
    per-pattern parameters would not be accepted.

    THRESHOLD: Changes evaluation criteria
        - Resets confirmation counter
        - Keeps reference template, accumulated data
        - Examples: pattern_1_mean_slope_threshold, pattern_2_max_pixels_threshold,
                    mean_slope_enabled and the other criteria flags

    TEMPLATE: Invalidates reference template
        - Resets reference template and confirmation counter
        - Keeps accumulated data
        - Examples: pattern_1_crop_rect, pattern_2_crop_rect

    MONITORING: Updates monitoring configuration
        - Just updates the value
        - No state reset (confirmation_count clamped if needed)
        - Examples: confirmation_rounds, analysis_interval_seconds

    DEFERRED: Applied to next patterning session
        - Updates stored value
        - Takes effect when patterning restarts
        - The largest class: number_of_images, plus auto_cb_on_start and the eleven
          cb_* calibration settings. Those are deferred because the one-shot CB
          calibration is their only reader and it runs once per patterning
          session -- which is what makes them safe to edit while monitoring runs.
    """
    THRESHOLD = auto()
    TEMPLATE = auto()
    MONITORING = auto()
    DEFERRED = auto()


# ===========================
# Data Structures
# ===========================

@dataclass
class ParameterUpdate:
    """
    Represents a queued parameter update.
    
    Updates are queued and applied at safe points (start of batch processing)
    to avoid race conditions and ensure consistent state.
    """
    param_name: str
    value: object
    pattern_idx: int = -1  # -1 means all patterns, 0/1 for specific pattern
    impact: ParameterImpact | None = None
    
    def __post_init__(self):
        """Automatically determine impact if not specified."""
        if self.impact is None:
            self.impact = classify_parameter_impact(self.param_name)


# ===========================
# Parameter Classification
# ===========================

# Every live-updatable parameter declares what it is here: its impact on
# monitoring state, and which SpinBoxConstraints entry bounds it. Nothing is
# inferred from the name, in either direction.
#
# Impact used to be a substring cascade ('threshold' in name -> THRESHOLD,
# 'enabled' -> THRESHOLD, ...). Three ways that misroutes:
#
#   * A cb_* field matches no rule and falls through to a MONITORING default.
#     MONITORING means "in force now" and says so -- "Cb Max White Clip Fraction
#     updated to 0.01" -- but a one-shot calibration parameter cannot take effect
#     until the next patterning session, so the operator would be told the
#     opposite of what happened.
#   * threshold_num_classes, percent_difference_threshold and
#     aspect_ratio_threshold all contain "threshold", so all three route to
#     _apply_threshold_update, which resets every pattern's confirmation counter
#     -- for parameters with no bearing on the completion criteria at all. Wired
#     to a spin box emitting per step, confirmation could never reach its target
#     and the monitor would never auto-stop the mill.
#   * A rename silently reclassifies a parameter without touching its code. Any
#     boolean renamed to *_enabled becomes THRESHOLD; a crop name containing
#     "threshold" classifies as THRESHOLD instead of TEMPLATE and quietly stops
#     invalidating the reference template.
#
# None of those reached a released build -- the Settings dialog was locked for
# the whole run and no emitter fed the cascade an unclassified name -- but all
# three are one new emitter away, which is what this table forecloses.
#
# The `bounds` column exists because the two naming families do not agree and
# deliberately are not being aligned: the constraints spell these
# maximum_pixels_threshold / mean_pixel_slope_threshold, matching the UI labels,
# the QSettings keys and the run_metadata schema, while the worker names them
# per-pattern max_pixels_threshold / mean_slope_threshold. Renaming either side
# crosses a boundary that costs more than the tidiness is worth -- the UI names
# are persisted QSettings keys, so changing them would silently reset an
# operator's tuned thresholds to defaults on next launch, and they are the
# ui_parameters keys in every saved run. Declaring the pairing here is the
# alternative to deriving it by string surgery, which is what silently skipped
# the range check on the three controls the operator touches most.
#
# An unregistered name is refused rather than defaulted: applying an
# unclassified parameter with whatever state handling happens to match its name
# is the failure this table exists to prevent.
_PATTERN_THRESHOLDS = (
    # queued suffix           SpinBoxConstraints prefix
    ("mean_slope_threshold",  "mean_pixel_slope_threshold"),
    ("match_score_threshold", "match_score_threshold"),
    ("max_pixels_threshold",  "maximum_pixels_threshold"),
)

#: Registered parameter -> the SpinBoxConstraints prefix that bounds it. Absent
#: means the parameter's own name is the prefix.
#:
#: Every per-pattern threshold is listed, including the one whose suffix already
#: matches its constraint: the pattern_N_ prefix means the parameter name is
#: never the constraint name, so omitting the "same" ones would drop them back
#: onto the un-prefixed fallback and skip their range check -- the very bug this
#: table exists to close, reintroduced by an optimization.
PARAMETER_BOUNDS = {
    f"pattern_{idx}_{suffix}": constraint
    for idx in (1, 2) for suffix, constraint in _PATTERN_THRESHOLDS
}

PARAMETER_IMPACTS = {
    # Per-pattern completion thresholds: the criteria changed, so any partial
    # confirmation streak was judged under the old rule.
    **{f"pattern_{idx}_{suffix}": ParameterImpact.THRESHOLD
       for idx in (1, 2) for suffix, _constraint in _PATTERN_THRESHOLDS},

    # Criteria on/off checkboxes (global, applied to every pattern).
    "mean_slope_enabled": ParameterImpact.THRESHOLD,
    "match_score_enabled": ParameterImpact.THRESHOLD,
    "percent_pixels_enabled": ParameterImpact.THRESHOLD,

    # Crop rectangles: the reference template no longer describes the region.
    "pattern_1_crop_rect": ParameterImpact.TEMPLATE,
    "pattern_2_crop_rect": ParameterImpact.TEMPLATE,

    # Monitoring configuration: effective now, no state reset.
    "confirmation_rounds": ParameterImpact.MONITORING,
    "analysis_interval_seconds": ParameterImpact.MONITORING,

    # Batch size: applies to the next patterning session.
    "number_of_images": ParameterImpact.DEFERRED,

    # Auto CB. Read once per patterning session, when CBEdgeConfig is built at
    # the top of the one-shot calibration, and read by nothing else -- so they
    # are safe to change while monitoring runs, but cannot take effect until that
    # next calibration. DEFERRED is what says so to the operator.
    #
    # auto_cb_on_start is safe here only because the arming order was fixed: the
    # opt-in is now read at the trigger, after this queue drains, so enabling it
    # mid-run genuinely takes effect on the next patterning session. Read at the
    # stopped->running transition (as it was), a mid-run enable armed from the
    # stale value and silently did nothing.
    "auto_cb_on_start": ParameterImpact.DEFERRED,
    # Read at the same trigger as auto_cb_on_start, so a mid-run change
    # applies at the next patterning session's calibration decision.
    "cb_recalibrate_every_session": ParameterImpact.DEFERRED,
    "cb_white_level": ParameterImpact.DEFERRED,
    "cb_lower_margin": ParameterImpact.DEFERRED,
    "cb_upper_margin": ParameterImpact.DEFERRED,
    "cb_max_white_clip_fraction": ParameterImpact.DEFERRED,
    "cb_max_black_clip_fraction": ParameterImpact.DEFERRED,
    "cb_min_bound": ParameterImpact.DEFERRED,
    "cb_max_bound": ParameterImpact.DEFERRED,
    "cb_max_iterations": ParameterImpact.DEFERRED,
    "cb_settle_seconds": ParameterImpact.DEFERRED,
    "cb_frames_per_measurement": ParameterImpact.DEFERRED,
}


#: Registered parameters carrying a boolean, for validation. The criteria flags
#: are covered by their *_enabled suffix; this names the ones that are not.
_BOOLEAN_PARAMETERS = frozenset(
    {"auto_cb_on_start", "cb_recalibrate_every_session"})


def classify_parameter_impact(param_name: str) -> ParameterImpact | None:
    """
    Look up a parameter's declared impact.

    :param param_name: Name of the parameter
    :return: its ParameterImpact, or None when the name is not registered
        (the caller must refuse the update -- see PARAMETER_IMPACTS)
    """
    return PARAMETER_IMPACTS.get(param_name)


# ===========================
# Parameter Update Manager
# ===========================

class WorkerParameterManager:
    """
    Manages live parameter updates during workflow execution.
    
    Provides thread-safe parameter updates by queueing changes and applying
    them at safe points in the workflow (batch boundaries).
    """
    
    def __init__(self):
        """Initialize the parameter manager."""
        self._pending_updates = []  # List of ParameterUpdate objects
        self._update_count = 0
        # Guards _pending_updates. queue_update() is now called on the GUI thread
        # (DirectConnection) while apply_pending_updates() runs on the worker
        # thread, so the list must be lock-protected.
        self._lock = threading.Lock()

    def queue_update(self, param_name: str, value: object,
                     pattern_idx: int = -1) -> bool:
        """
        Queue a parameter update to be applied at the next safe point.

        If an update for the same parameter already exists in the queue, it is
        replaced with the new value (de-duplication). This ensures only the
        latest value is applied, avoiding wasted work from intermediate changes.

        :param param_name: Name of the parameter to update
        :param value: New value for the parameter
        :param pattern_idx: Which pattern to update (-1 for all, 0/1 for specific)
        :return: True when queued; False when the name is not registered in
            PARAMETER_IMPACTS and the update was refused
        """
        impact = classify_parameter_impact(param_name)
        if impact is None:
            # Refused, not defaulted: applying an unclassified parameter with
            # whatever state handling happens to match its name is the bug this
            # registry exists to prevent.
            logger.warning(
                f"Parameter update refused: '{param_name}' is not declared in "
                f"PARAMETER_IMPACTS, so its state handling is unknown"
            )
            return False

        update = ParameterUpdate(
            param_name=param_name,
            value=value,
            pattern_idx=pattern_idx,
            impact=impact
        )

        with self._lock:
            # Check if an update for this parameter already exists (de-dup)
            existing_index = None
            for i, existing_update in enumerate(self._pending_updates):
                if (existing_update.param_name == param_name and
                        existing_update.pattern_idx == pattern_idx):
                    existing_index = i
                    break

            if existing_index is not None:
                # Replace existing update with new value
                old_value = self._pending_updates[existing_index].value
                self._pending_updates[existing_index] = update
                logger.info(
                    f"Parameter update replaced: {param_name} = {old_value} -> {value} "
                    f"(pattern_idx={pattern_idx}, impact={update.impact.name})"
                )
            else:
                # Add new update to queue
                self._pending_updates.append(update)
                self._update_count += 1
                logger.info(
                    f"Parameter update queued #{self._update_count}: "
                    f"{param_name} = {value} (pattern_idx={pattern_idx}, impact={update.impact.name})"
                )
        return True

    def has_pending_updates(self) -> bool:
        """Check if there are pending updates to apply."""
        with self._lock:
            return len(self._pending_updates) > 0
    
    def apply_pending_updates(
        self,
        pattern_states: list,
        parameters: object
    ) -> list[str]:
        """
        Apply all queued parameter updates and return status messages.
        
        Updates are applied in the order they were queued. State is reset
        appropriately based on the parameter impact type.
        
        :param pattern_states: List of PatternState objects
        :param parameters: WorkerParameters object
        :return: List of status messages to display to user
        """
        # Atomically take the pending updates and clear the queue, so the GUI
        # thread can keep queue_update()-ing while we apply this snapshot.
        with self._lock:
            if not self._pending_updates:
                return []
            pending = self._pending_updates
            self._pending_updates = []

        status_messages = []

        # Apply each update (outside the lock -- may touch pattern_states/params)
        for update in pending:
            msg = self._apply_single_update(
                update,
                pattern_states,
                parameters
            )

            if msg:
                status_messages.append(msg)

        logger.info(f"Applied {len(pending)} parameter update(s)")

        return status_messages
    
    def _apply_single_update(
        self,
        update: ParameterUpdate,
        pattern_states: list,
        parameters: object
    ) -> str:
        """
        Apply a single parameter update with appropriate state management.
        
        :param update: ParameterUpdate to apply
        :param pattern_states: List of PatternState objects
        :param parameters: WorkerParameters object
        :return: Status message describing the change
        """
        # Determine which patterns to affect
        if update.pattern_idx == -1:
            affected_patterns = pattern_states
            pattern_label = "all patterns"
        else:
            if update.pattern_idx >= len(pattern_states):
                # Pattern not active yet - update parameters object only
                # so the value is ready when the pattern initializes
                setattr(parameters, update.param_name, update.value)
                logger.debug(
                    f"Stored {update.param_name}={update.value} "
                    f"(Pattern {update.pattern_idx + 1} not yet active)"
                )
                return (f"Stored {display_name(update.param_name)} = "
                        f"{_fmt_value(update.value)} for future use")
            affected_patterns = [pattern_states[update.pattern_idx]]
            pattern_label = f"Pattern {update.pattern_idx + 1}"
        
        # Apply based on impact type
        if update.impact == ParameterImpact.THRESHOLD:
            return self._apply_threshold_update(
                update, affected_patterns, parameters, pattern_label
            )
        
        elif update.impact == ParameterImpact.TEMPLATE:
            return self._apply_template_update(
                update, affected_patterns, parameters, pattern_label
            )
        
        elif update.impact == ParameterImpact.MONITORING:
            return self._apply_monitoring_update(
                update, affected_patterns, parameters, pattern_label
            )
        
        elif update.impact == ParameterImpact.DEFERRED:
            return self._apply_deferred_update(
                update, parameters, pattern_label
            )
    
    def _apply_threshold_update(
        self,
        update: ParameterUpdate,
        affected_patterns: list,
        parameters: object,
        pattern_label: str
    ) -> str:
        """
        Apply threshold parameter update or criteria enabled flag change.
        
        Resets: Confirmation counter
        Keeps: Reference template, accumulated data
        """
        param_lower = update.param_name.lower()
        
        # Handle criteria enabled flags (global on parameters, not per-pattern)
        if 'enabled' in param_lower:
            setattr(parameters, update.param_name, update.value)
            # Reset confirmation counters since evaluation criteria changed
            for ps in affected_patterns:
                if ps.evaluation_state:
                    ps.evaluation_state.reset_confirmation()
            # Registry-backed name, not a string munge: the registered
            # display names for the criteria flags already omit "enabled"
            # ("Mean Slope"), and the registry is this module's rule for
            # everything else about a parameter.
            flag_display = display_name(update.param_name)
            state_str = "enabled" if update.value else "disabled"
            return f"{flag_display} criterion {state_str} - confirmation reset"
        
        # Bound before the loop: affected_patterns is legitimately empty when an
        # all-patterns update is queued before the states exist (a batch boundary
        # is declared when there are no pattern states at all). Assigning this
        # only inside the loop made the return below raise UnboundLocalError,
        # which propagates out of _process_rtm_data and kills the run.
        threshold_name = "Threshold"

        # Handle threshold updates on PatternState
        for ps in affected_patterns:
            if 'mean_slope' in param_lower:
                ps.mean_slope_threshold = update.value
                threshold_name = "Mean slope"
            elif 'match_score' in param_lower:
                ps.match_score_threshold = update.value
                threshold_name = "Match score"
            elif 'max_pixels' in param_lower or 'maximum_pixels' in param_lower:
                ps.max_pixels_threshold = update.value
                threshold_name = "Maximum pixels"
            else:
                threshold_name = "Threshold"
            
            # Reset confirmation counter (criteria changed)
            if ps.evaluation_state:
                ps.evaluation_state.reset_confirmation()
        
        # Also update in parameters object
        setattr(parameters, update.param_name, update.value)
        
        return (f"{threshold_name} threshold updated to "
                f"{_fmt_value(update.value)} ({pattern_label}) - "
                f"confirmation reset")
    
    def _apply_template_update(
        self,
        update: ParameterUpdate,
        affected_patterns: list,
        parameters: object,
        pattern_label: str
    ) -> str:
        """
        Apply crop rectangle update.
        
        Resets: Reference template, confirmation counter
        Keeps: Accumulated data
        """
        for ps in affected_patterns:
            # No-op change: an identical rect must not reset the reference,
            # confirmation, or the stall latch's history.
            if update.value is not None and ps.crop_rect is not None \
                    and tuple(update.value) == tuple(ps.crop_rect):
                continue

            # Update crop rectangle
            ps.crop_rect = update.value
            # Record when it took effect: updates apply at a batch boundary, so the
            # new rect first governs batch (batch_count + 1).
            ps.crop_rect_history.append({
                "batch": ps.batch_count + 1,
                "rect": list(update.value) if update.value else None,
            })

            # Reset reference template (will be recaptured)
            ps.reference_template = None

            # Restart the stall latch's view of the foreground history: the new
            # crop is a new region/scale, so the running peak and flat window
            # must not span the change. If the trace had already dropped from
            # its peak, carry that state across the reset (see
            # apply_stall_carry) -- otherwise a crop nudge on an already-floored
            # gridbar trace would permanently disarm the latch in its flagship
            # scenario, with re-latching impossible because the post-crop slice
            # has no peak to have dropped from.
            pre_crop_slice = ps.white_pixel_percentages[ps.stall_history_start:]
            if pre_crop_slice:
                pre_crop_stall = evaluate_foreground_stall(
                    pre_crop_slice,
                    StallConfig(
                        stall_window=getattr(parameters, "stall_window", 16),
                        stall_drop_fraction=getattr(
                            parameters, "stall_drop_fraction", 0.35),
                        stall_rel_tolerance=getattr(
                            parameters, "stall_rel_tolerance", 0.10),
                        stall_abs_tolerance=getattr(
                            parameters, "stall_abs_tolerance", 0.15),
                    ),
                )
                if pre_crop_stall.dropped:
                    ps.stall_dropped_carry = True
                    logger.warning(
                        "Stall latch history reset by crop change while the "
                        "foreground trace was already dropped - carrying the "
                        "dropped state so the latch can re-arm after the change"
                    )
            ps.stall_history_start = len(ps.white_pixel_percentages)

            # Reset confirmation counter
            if ps.evaluation_state:
                ps.evaluation_state.reset_confirmation()
        
        # Update in parameters
        setattr(parameters, update.param_name, update.value)
        
        return f"Crop rectangle updated ({pattern_label}) - template and confirmation reset"
    
    def _apply_monitoring_update(
        self,
        update: ParameterUpdate,
        affected_patterns: list,
        parameters: object,
        pattern_label: str
    ) -> str:
        """
        Apply monitoring configuration update.
        
        Resets: Nothing (just updates value)
        Special: Clamps confirmation_count if needed
        """
        # Update parameter
        setattr(parameters, update.param_name, update.value)
        
        # Special handling for confirmation_rounds
        if 'confirmation' in update.param_name.lower():
            # Clamp existing confirmation counters if they exceed new max
            for ps in affected_patterns:
                if ps.evaluation_state and ps.evaluation_state.confirmation_count > update.value:
                    ps.evaluation_state.confirmation_count = update.value
                    logger.info(
                        f"Pattern {ps.pattern_index + 1}: confirmation count "
                        f"clamped to {update.value}"
                    )
            
            return f"Confirmation rounds updated to {update.value}"
        
        # Other monitoring parameters (shouldn't happen with current UI)
        return (f"{display_name(update.param_name)} updated to "
                f"{_fmt_value(update.value)}")
    
    def _apply_deferred_update(
        self,
        update: ParameterUpdate,
        parameters: object,
        pattern_label: str
    ) -> str:
        """
        Apply deferred update - takes effect next patterning session.
        
        Resets: Nothing (stored for next session)
        """
        # Update parameter but don't affect current session
        setattr(parameters, update.param_name, update.value)
        
        return (f"{display_name(update.param_name)} updated to "
                f"{_fmt_value(update.value)} "
                f"(applies to next patterning session)")
    
    def get_pending_update_summary(self) -> str:
        """
        Get a summary of pending updates.
        
        :return: Human-readable summary string
        """
        with self._lock:
            if not self._pending_updates:
                return "No pending updates"

            summary_lines = [f"{len(self._pending_updates)} pending update(s):"]

            for i, update in enumerate(self._pending_updates, 1):
                pattern_info = "all" if update.pattern_idx == -1 else f"P{update.pattern_idx + 1}"
                summary_lines.append(
                    f"  {i}. {update.param_name} = {update.value} "
                    f"({pattern_info}, {update.impact.name})"
                )

        return "\n".join(summary_lines)

    def clear_pending_updates(self):
        """Clear all pending updates without applying them."""
        with self._lock:
            num_cleared = len(self._pending_updates)
            self._pending_updates.clear()

        if num_cleared > 0:
            logger.warning(f"Cleared {num_cleared} pending update(s) without applying")


# ===========================
# Utility Functions
# ===========================

def spin_box_range(param_name: str) -> tuple[float, float] | None:
    """
    The operator-facing range for a parameter, taken from the same
    SpinBoxConstraints the dialog's spin boxes are built from.

    Reusing them keeps one source of truth: a limit widened in the dialog is
    widened here automatically, and the live-update path can never accept a value
    the UI would have refused.

    The constraint is looked up through PARAMETER_BOUNDS rather than derived from
    the parameter's name. Deriving it is what silently skipped the range check on
    the three per-pattern thresholds -- the controls the operator touches most --
    because the queue names them pattern_1_max_pixels_threshold while the
    constraints spell them maximum_pixels_threshold_*.

    :param param_name: Parameter name
    :return: (minimum, maximum), or None for parameters with no declared range
        (crop rects and booleans have none)
    """
    prefix = PARAMETER_BOUNDS.get(param_name, param_name)
    low = getattr(_SPIN_BOX_CONSTRAINTS, f"{prefix}_min", None)
    high = getattr(_SPIN_BOX_CONSTRAINTS, f"{prefix}_max", None)
    if low is None or high is None:
        return None
    return float(low), float(high)


def validate_parameter_update(param_name: str, value: object) -> tuple[bool, str]:
    """
    Validate a parameter update before queuing.

    Checks:
    - The name is registered in PARAMETER_IMPACTS (unknown names are refused)
    - The value is inside the operator-facing spin-box range, where one exists
    - Thresholds are positive numbers
    - Confirmation rounds is positive integer
    - Crop rectangles are valid (x, y, width, height)
    - Number of images is positive integer
    - Enabled flags are boolean

    :param param_name: Parameter name
    :param value: New value
    :return: (is_valid, error_message)
    """
    # An unregistered name has no declared state handling, so there is no safe
    # way to apply it. Refuse here, where the caller already surfaces the reason.
    if param_name not in PARAMETER_IMPACTS:
        return False, f"{display_name(param_name)} is not an adjustable setting"

    # Range check against the dialog's own constraints, for every parameter that
    # declares one. This is what stops an out-of-range CB setting reaching the
    # controller through the live path (the dialog clamps; a direct queue_update
    # did not).
    bounds = spin_box_range(param_name)
    if bounds is not None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"{display_name(param_name)} must be a number, got {value!r}"
        low, high = bounds
        if not low <= float(value) <= high:
            return False, (f"{display_name(param_name)} must be between "
                           f"{low:g} and {high:g}, got {value}")

    # Everything below is legacy name-based validation, reachable only for a
    # registered parameter that declares no spin-box range (today: booleans and
    # crop rectangles). Give a new registered parameter a PARAMETER_BOUNDS
    # entry rather than a clause here -- a name containing "threshold" or
    # "confirmation" would re-enter this cascade and be validated by substring.
    #
    # Boolean flags. Named explicitly rather than sniffed from the name: the
    # substring test below misses auto_cb_on_start, and a rename would silently
    # move a parameter in or out of this check.
    if param_name in _BOOLEAN_PARAMETERS or 'enabled' in param_name.lower():
        if not isinstance(value, bool):
            return False, f"{display_name(param_name)} must be on or off, got {value!r}"
        return True, ""
    
    # Thresholds should be positive numbers
    if 'threshold' in param_name.lower():
        if not isinstance(value, (int, float)) or value < 0:
            return False, f"Threshold must be a non-negative number, got {value}"
        return True, ""
    
    # Confirmation rounds should be positive integer
    if 'confirmation' in param_name.lower():
        if not isinstance(value, int) or value < 1:
            return False, f"Confirmation rounds must be a positive integer, got {value}"
        return True, ""
    
    # Number of images should be positive integer
    if 'number_of_images' in param_name.lower():
        if not isinstance(value, int) or value < 1:
            return False, f"Number of images must be a positive integer, got {value}"
        return True, ""
    
    # Analysis interval should be positive integer
    if 'analysis_interval' in param_name.lower():
        if not isinstance(value, int) or value < 1:
            return False, f"Analysis interval must be a positive integer, got {value}"
        return True, ""
    
    # Crop rectangles should be tuples of 4 integers
    if 'crop' in param_name.lower():
        if not isinstance(value, tuple) or len(value) != 4:
            return False, f"Crop rectangle must be (x, y, width, height), got {value}"
        if not all(isinstance(v, int) for v in value):
            return False, f"Crop rectangle values must be integers, got {value}"
        if value[2] <= 0 or value[3] <= 0:
            return False, f"Crop dimensions must be positive, got width={value[2]}, height={value[3]}"
        return True, ""
    
    return True, ""