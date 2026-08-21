"""
Worker Parameter Manager Module

Manages live parameter updates during workflow execution with thread-safe
queuing and state-aware application.

Key responsibilities:
- Queue parameter updates (thread-safe)
- Determine state reset requirements based on parameter type
- Apply updates at safe batch boundaries
- Generate user-friendly status messages
- Manage parameter validation
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto

from script_modules.results_evaluation import (
    StallConfig, evaluate_foreground_stall
)


logger = logging.getLogger(__name__)


# Display names for machine-generated parameter names in user-facing status
# messages. Fallback for unmapped names: title-cased snake_case.
PARAM_DISPLAY_NAMES = {
    "number_of_images": "Number of Images",
    "analysis_interval_seconds": "Analysis Interval",
    "mean_slope_enabled": "Mean Slope",
    "match_score_enabled": "Match Score",
    "percent_pixels_enabled": "Percent Pixels",
}


# ===========================
# Enumerations
# ===========================

class ParameterImpact(Enum):
    """
    Classification of how parameter changes affect monitoring state.
    
    THRESHOLD: Changes evaluation criteria
        - Resets confirmation counter
        - Keeps reference template, accumulated data
        - Examples: mean_slope_threshold, match_score_threshold, max_pixels_threshold,
                    criteria enabled flags
    
    TEMPLATE: Invalidates reference template
        - Resets reference template and confirmation counter
        - Keeps accumulated data
        - Examples: crop_rect
    
    MONITORING: Updates monitoring configuration
        - Just updates the value
        - No state reset (confirmation_count clamped if needed)
        - Examples: confirmation_rounds
    
    DEFERRED: Applied to next patterning session
        - Updates stored value
        - Takes effect when patterning restarts
        - Examples: number_of_images
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

def classify_parameter_impact(param_name: str) -> ParameterImpact:
    """
    Classify a parameter by its impact on monitoring state.
    
    Only handles parameters exposed in the UI:
    - Thresholds (mean_slope, match_score, max_pixels)
    - Criteria enabled flags (from checkboxes)
    - Crop rectangles
    - Confirmation rounds
    - Number of images
    
    :param param_name: Name of the parameter
    :return: ParameterImpact classification
    """
    # Threshold parameters - reset confirmation only
    if 'threshold' in param_name.lower():
        return ParameterImpact.THRESHOLD
    
    # Criteria enabled flags - reset confirmation (evaluation criteria changed)
    if 'enabled' in param_name.lower():
        return ParameterImpact.THRESHOLD
    
    # Crop rectangle - reset template and confirmation
    if 'crop' in param_name.lower():
        return ParameterImpact.TEMPLATE
    
    # Confirmation rounds - just update value
    if 'confirmation' in param_name.lower():
        return ParameterImpact.MONITORING
    
    # Analysis interval - recalculates window size using calibrated rate
    if 'analysis_interval' in param_name.lower():
        return ParameterImpact.MONITORING
    
    # Number of images - defer to next session
    if 'number_of_images' in param_name.lower():
        return ParameterImpact.DEFERRED
    
    # Default to monitoring (shouldn't happen with UI parameters)
    logger.warning(f"Unknown parameter type: {param_name}, defaulting to MONITORING")
    return ParameterImpact.MONITORING


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

    def queue_update(self, param_name: str, value: object, pattern_idx: int = -1):
        """
        Queue a parameter update to be applied at the next safe point.
        
        If an update for the same parameter already exists in the queue, it is
        replaced with the new value (de-duplication). This ensures only the
        latest value is applied, avoiding wasted work from intermediate changes.
        
        :param param_name: Name of the parameter to update
        :param value: New value for the parameter
        :param pattern_idx: Which pattern to update (-1 for all, 0/1 for specific)
        """
        # Create new update
        update = ParameterUpdate(
            param_name=param_name,
            value=value,
            pattern_idx=pattern_idx
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
                return f"Stored {update.param_name.replace('_', ' ')} = {update.value} for future use"
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
            flag_display = update.param_name.replace('_', ' ').replace('enabled', '').strip().title()
            state_str = "enabled" if update.value else "disabled"
            return f"{flag_display} criterion {state_str} - confirmation reset"
        
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
        
        return f"{threshold_name} threshold updated to {update.value} ({pattern_label}) - confirmation reset"
    
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
            # must not span the change. If the trace had ALREADY dropped from
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
        param_display = PARAM_DISPLAY_NAMES.get(update.param_name, update.param_name.replace('_', ' ').title())
        return f"{param_display} updated to {update.value}"
    
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
        
        param_display = PARAM_DISPLAY_NAMES.get(update.param_name, update.param_name.replace('_', ' ').title())
        return f"{param_display} updated to {update.value} (applies to next patterning session)"
    
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

def validate_parameter_update(param_name: str, value: object) -> tuple[bool, str]:
    """
    Validate a parameter update before queuing.
    
    Checks:
    - Thresholds are positive numbers
    - Confirmation rounds is positive integer
    - Crop rectangles are valid (x, y, width, height)
    - Number of images is positive integer
    - Enabled flags are boolean
    
    :param param_name: Parameter name
    :param value: New value
    :return: (is_valid, error_message)
    """
    # Enabled flags should be boolean
    if 'enabled' in param_name.lower():
        if not isinstance(value, bool):
            return False, f"Enabled flag must be boolean, got {value}"
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