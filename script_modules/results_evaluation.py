"""
Results Evaluation Module

This module handles evaluation of RTM image processing results to determine
when FIB milling patterns have reached completion.

The monitoring workflow has three phases:
1. DELAY - Initial phase while the pattern starts milling (delay_active=True)
2. MONITORING - Active evaluation against thresholds (delay_active=False)
3. COMPLETE - Pattern has met all criteria for required confirmation rounds

During DELAY phase, no criteria checking occurs. Once delay is deactivated,
the module evaluates up to four criteria each round (only those enabled via
checkboxes in the Pattern Results group box):
- Mean pixel slope: Rate of brightness change (should be near zero when milling completes)
- Match score: Pattern similarity to reference (should be low/stable when complete)
- White pixels: Percentage of bright pixels in binary image (should be low when complete)

All enabled criteria must be met simultaneously for consecutive rounds
(confirmation_rounds) before marking the pattern as complete. Disabled criteria
automatically pass. If no criteria are enabled, rounds pass to completion.
"""

from dataclasses import dataclass
import logging

from script_modules.atc_monitor_parameters import ForegroundCompletionMode

# Foreground energy below this is treated as "no usable start baseline" so the
# RELATIVE_PLATEAU relative-drop test fails closed (never falsely completes).
_ENERGY_EPS = 1e-6


@dataclass
class EvaluationState:
    """
    Tracks the evaluation state for a single pattern.
    
    This state persists across evaluation rounds and gets updated each time
    evaluate_pattern_completion() is called.
    """
    confirmation_count: int = 0          # Number of consecutive rounds meeting criteria
    is_complete: bool = False            # Whether pattern has reached completion
    
    # Current measurements (for logging/display purposes)
    current_slope: float | None = None
    current_match_score: float | None = None
    current_white_pixels: float | None = None

    # Authoritative per-criterion pass/fail from the most recent monitoring round.
    # These honor the enabled-checkboxes and the foreground completion mode, so the
    # GUI can display them directly instead of recomputing (which would let the
    # indicators disagree with the confirmation counter). None until first evaluated.
    slope_ok: bool | None = None
    match_ok: bool | None = None
    pixels_ok: bool | None = None

    # Flags
    all_criteria_met: bool = False       # Whether current round meets all criteria
    
    def reset_confirmation(self):
        """Reset confirmation counter (called when criteria not met)."""
        self.confirmation_count = 0
        self.all_criteria_met = False
    
    def increment_confirmation(self):
        """Increment confirmation counter (called when criteria met)."""
        self.confirmation_count += 1
        self.all_criteria_met = True
    
    def mark_complete(self):
        """Mark pattern as complete."""
        self.is_complete = True


@dataclass
class EvaluationCriteria:
    """
    Threshold values and enabled flags for pattern completion evaluation.
    
    These come from UI parameters and define what "complete" means.
    Enabled flags are controlled by checkboxes in the Pattern Results group box.
    """
    # Thresholds
    mean_pixel_slope_threshold: float
    match_score_threshold: float
    maximum_pixels_threshold: float
    confirmation_rounds: int
    
    # Enabled flags (from checkboxes)
    mean_slope_enabled: bool = True
    match_score_enabled: bool = True
    percent_pixels_enabled: bool = True

    # Foreground completion mode + RELATIVE_PLATEAU thresholds. ABSOLUTE keeps the
    # original "energy <= maximum_pixels_threshold" behavior; RELATIVE_PLATEAU is
    # grid-bar-immune (relative drop + slope plateau on the energy series).
    foreground_completion_mode: ForegroundCompletionMode = ForegroundCompletionMode.ABSOLUTE
    energy_drop_fraction: float = 0.25
    energy_slope_threshold: float = 0.1


@dataclass
class CriteriaResult:
    """
    Per-criterion pass/fail breakdown for one evaluation round.

    ``all_met`` is the AND of the three booleans and is what drives the
    confirmation counter; the individual booleans are surfaced to the GUI so the
    green/red indicators reflect exactly the same decision (including enabled-flag
    auto-pass and the foreground completion mode).
    """
    slope_ok: bool
    match_ok: bool
    pixels_ok: bool

    @property
    def all_met(self) -> bool:
        return self.slope_ok and self.match_ok and self.pixels_ok


def evaluate_pattern_completion(
    state: EvaluationState,
    criteria: EvaluationCriteria,
    mean_pixel_slope: float,
    match_score: float,
    white_pixels_percentage: float,
    delay_active: bool,
    energy_slope: float = 0.0,
    start_energy: float | None = None
) -> EvaluationState:
    """
    Evaluate whether a pattern has reached completion based on current measurements.
    
    This is the main evaluation function called each processing round. It:
    1. Skips evaluation if delay is active (returns state unchanged)
    2. Checks if all enabled criteria are met
    3. Updates confirmation counter (increment if met, reset if not)
    4. Marks pattern complete if confirmation rounds reached
    
    :param state: Current evaluation state for this pattern
    :param criteria: Threshold criteria from UI parameters
    :param mean_pixel_slope: Current slope of mean pixel values (absolute value)
    :param match_score: Current pattern match score
    :param white_pixels_percentage: Current white pixel percentage
    :param delay_active: Whether delay processing is currently active
    :return: Updated evaluation state
    """
    # Update current measurements for logging/display
    state.current_slope = mean_pixel_slope
    state.current_match_score = match_score
    state.current_white_pixels = white_pixels_percentage
    
    # Skip evaluation during delay phase
    if delay_active:
        logging.debug("Delay active - skipping criteria evaluation")
        return state
    
    # Skip evaluation if already complete
    if state.is_complete:
        logging.debug("Pattern already complete - skipping evaluation")
        return state
    
    # Check all enabled criteria (returns the per-criterion breakdown)
    result = check_completion_criteria(
        mean_pixel_slope=mean_pixel_slope,
        match_score=match_score,
        white_pixels_percentage=white_pixels_percentage,
        criteria=criteria,
        energy_slope=energy_slope,
        start_energy=start_energy
    )

    # Surface the authoritative per-criterion status so the GUI displays exactly
    # what the counter is based on.
    state.slope_ok = result.slope_ok
    state.match_ok = result.match_ok
    state.pixels_ok = result.pixels_ok

    # Update confirmation counter based on criteria
    if result.all_met:
        state.increment_confirmation()
        logging.info(
            f"Criteria met - confirmation round {state.confirmation_count}/{criteria.confirmation_rounds}"
        )
        
        # Check if we've reached required confirmation rounds
        if state.confirmation_count >= criteria.confirmation_rounds:
            state.mark_complete()
            logging.info("Pattern complete")
    else:
        # Criteria not met - reset counter
        if state.confirmation_count > 0:
            logging.info("Criteria not met - resetting confirmation counter")
        state.reset_confirmation()
    
    return state


def check_completion_criteria(
    mean_pixel_slope: float,
    match_score: float,
    white_pixels_percentage: float,
    criteria: EvaluationCriteria,
    energy_slope: float = 0.0,
    start_energy: float | None = None
) -> CriteriaResult:
    """
    Check the enabled completion criteria and return the per-criterion breakdown.

    Only enabled criteria are evaluated. Disabled criteria automatically pass.
    If no criteria are enabled, every criterion passes (rounds pass to completion).

    :param mean_pixel_slope: Current slope (absolute value)
    :param match_score: Current match score
    :param white_pixels_percentage: Current white pixel / foreground-energy value
    :param criteria: Threshold values and enabled flags
    :param energy_slope: |slope| of the foreground-energy history (RELATIVE_PLATEAU only)
    :param start_energy: foreground energy at start of monitoring (RELATIVE_PLATEAU only)
    :return: CriteriaResult with slope_ok/match_ok/pixels_ok (and .all_met)
    """
    # Each criterion: enabled → check threshold, disabled → auto-pass (True)
    slope_ok = (abs(mean_pixel_slope) <= criteria.mean_pixel_slope_threshold
                if criteria.mean_slope_enabled else True)

    match_ok = (match_score <= criteria.match_score_threshold
                if criteria.match_score_enabled else True)

    # Foreground criterion: ABSOLUTE level, or grid-bar-immune RELATIVE_PLATEAU.
    if not criteria.percent_pixels_enabled:
        pixels_ok = True
        pixels_detail = f"Foreground: {white_pixels_percentage:.2f} [DISABLED*]"
    elif criteria.foreground_completion_mode == ForegroundCompletionMode.RELATIVE_PLATEAU:
        # Done only when energy has (a) dropped to <= a fraction of its start AND
        # (b) plateaued. A constant grid-bar offset cancels from both tests. The
        # 'dropped' AND-clause also blocks flat-and-high completion and the early
        # case where the slope is 0.0 only for lack of points.
        dropped = (start_energy is not None and start_energy > _ENERGY_EPS
                   and white_pixels_percentage <= start_energy * criteria.energy_drop_fraction)
        plateaued = energy_slope <= criteria.energy_slope_threshold
        pixels_ok = dropped and plateaued
        target = (start_energy * criteria.energy_drop_fraction
                  if start_energy is not None else float('nan'))
        pixels_detail = (
            f"Foreground[REL]: {white_pixels_percentage:.2f} <= {target:.2f} "
            f"(drop {'OK' if dropped else 'FAIL'}), "
            f"slope {energy_slope:.4f} <= {criteria.energy_slope_threshold} "
            f"(plateau {'OK' if plateaued else 'FAIL'}) "
            f"[{'OK' if pixels_ok else 'FAIL'}]"
        )
    else:  # ABSOLUTE
        pixels_ok = white_pixels_percentage <= criteria.maximum_pixels_threshold
        pixels_detail = (
            f"Foreground[ABS]: {white_pixels_percentage:.2f} <= "
            f"{criteria.maximum_pixels_threshold} [{'OK' if pixels_ok else 'FAIL'}]"
        )

    # Log individual criterion status for debugging
    logging.debug(
        f"Criteria check - "
        f"Slope: {abs(mean_pixel_slope):.4f} <= {criteria.mean_pixel_slope_threshold} "
        f"[{'OK' if slope_ok else 'FAIL'}{'*' if not criteria.mean_slope_enabled else ''}], "
        f"Match: {match_score:.4f} <= {criteria.match_score_threshold} "
        f"[{'OK' if match_ok else 'FAIL'}{'*' if not criteria.match_score_enabled else ''}], "
        f"{pixels_detail}"
    )

    return CriteriaResult(slope_ok=slope_ok, match_ok=match_ok, pixels_ok=pixels_ok)


def check_all_patterns_complete(pattern_states: list[EvaluationState]) -> bool:
    """
    Check if all patterns have reached completion.
    
    This is used to determine when to stop patterning. Patterning should only
    stop when ALL active patterns are complete.
    
    For single pattern monitoring: returns True when that pattern is complete
    For dual pattern monitoring: returns True only when BOTH patterns are complete
    
    :param pattern_states: List of EvaluationState objects for all active patterns
    :return: True if all patterns are complete, False otherwise
    """
    if not pattern_states:
        return False
    
    return all(state.is_complete for state in pattern_states)