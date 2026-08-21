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
- White pixels: Percentage of bright pixels / foreground energy (should be low when
  complete; in ABSOLUTE_PLUS_STALL mode the criterion also passes when the trace
  has provably floored -- the grid-bar stall latch, see evaluate_foreground_stall)

All enabled criteria must be met simultaneously for consecutive rounds
(confirmation_rounds) before marking the pattern as complete. Disabled criteria
automatically pass. If no criteria are enabled, rounds pass to completion.
"""

from dataclasses import dataclass, replace
import logging
from statistics import median

from script_modules.atc_monitor_parameters import ForegroundCompletionMode

logger = logging.getLogger(__name__)

# A running peak below this is treated as "the trace never carried energy", so
# the stall latch's drop test fails closed (never falsely completes on a trace
# that was ~0 from the start -- plain ABSOLUTE already handles those).
_ENERGY_EPS = 1e-6


# ===========================
# Foreground stall latch (ABSOLUTE_PLUS_STALL)
# ===========================

@dataclass
class StallConfig:
    """
    Configuration for the foreground stall latch.

    Defaults were validated offline against the full 2026-08-17 field campaign
    (26 completed channels byte-identical to plain ABSOLUTE; the two hung runs
    latch within 3 cycles of the operator's manual stop; no latch during
    mid-run humps or innocent stalls).
    """
    stall_window: int = 16            # consecutive rounds the smoothed trace must stay flat
    stall_drop_fraction: float = 0.35  # smoothed value must be <= this fraction of the running peak
    stall_rel_tolerance: float = 0.10  # relative flatness tolerance (fraction of current value)
    stall_abs_tolerance: float = 0.15  # absolute flatness tolerance (foreground units)
    smoothing_points: int = 3          # trailing-median smoothing width (current value / flat window)
    baseline_smoothing_points: int = 5  # trailing-median width for the running PEAK: wider than
                                        # smoothing_points so a <=2-batch transient (e.g. a charging
                                        # flash) cannot inflate the peak and fake the drop condition
                                        # (adversarial review 2026-08-18: a 2-batch >=2.86x flash
                                        # slipped through median-3 and latched a steady trace)


@dataclass
class StallResult:
    """
    Outcome of one stall evaluation over the foreground history.

    ``latched`` is the AND of ``dropped`` and ``flat``; the components and the
    intermediate values are surfaced for logging, metrics persistence, and the
    GUI arming-level overlay.
    """
    smoothed: float | None = None     # trailing median of the last smoothing_points values
    baseline: float | None = None     # running peak of the smoothed series
    window_span: float | None = None  # max - min over the last stall_window smoothed values
    dropped: bool = False
    flat: bool = False
    latched: bool = False


def evaluate_foreground_stall(
    energy_history: list[float],
    config: StallConfig,
) -> StallResult:
    """
    Evaluate the grid-bar stall latch over a pattern's foreground history.

    The latch detects "the trace has floored": it requires the median-smoothed
    trace to have BOTH dropped to <= stall_drop_fraction of its running peak
    (so a mid-run plateau near the peak -- active milling pausing -- can never
    latch) AND stayed flat across the last stall_window rounds (windowed span
    within max(stall_rel_tolerance * value, stall_abs_tolerance) -- so a
    slowly-decaying tail keeps milling until it truly levels out). Smoothing is
    a trailing median so single-cycle spikes neither break a genuine flat
    window nor fake one.

    Pure function (no Qt, no state): feed it the full history each round.

    :param energy_history: Per-round foreground values (white-pixel %% or
        top-hat energy), monitoring phase, oldest first
    :param config: Stall latch tuning
    :return: StallResult; ``latched`` is False until the history is at least
        stall_window long, so the latch cannot fire before
        ~stall_window + confirmation_rounds rounds of monitoring
    """
    n = len(energy_history)
    if n == 0:
        return StallResult()

    k = max(1, int(config.smoothing_points))
    smoothed_series = [
        float(median(energy_history[max(0, i + 1 - k): i + 1])) for i in range(n)
    ]
    smoothed = smoothed_series[-1]

    # The running peak uses a WIDER trailing median than the flat window: a
    # transient shorter than half of baseline_smoothing_points (e.g. a 2-batch
    # charging flash) must not inflate the peak, or the drop test would read
    # the trace's ordinary level as "dropped" and the latch could fire on a
    # scene that never milled at all (false-complete direction).
    kb = max(1, int(config.baseline_smoothing_points))
    baseline = max(
        float(median(energy_history[max(0, i + 1 - kb): i + 1])) for i in range(n)
    )

    window = max(1, int(config.stall_window))
    if n < window:
        return StallResult(smoothed=smoothed, baseline=baseline)

    recent = smoothed_series[-window:]
    window_span = max(recent) - min(recent)

    dropped = (baseline > _ENERGY_EPS
               and smoothed <= config.stall_drop_fraction * baseline)
    tolerance = max(config.stall_rel_tolerance * smoothed,
                    config.stall_abs_tolerance)
    flat = window_span <= tolerance

    return StallResult(
        smoothed=smoothed,
        baseline=baseline,
        window_span=window_span,
        dropped=dropped,
        flat=flat,
        latched=dropped and flat,
    )


def stall_carry_still_valid(post_crop_history: list[float],
                            config: StallConfig) -> bool:
    """
    Whether a ``dropped`` state carried across a crop change is still credible.

    A crop change resets the latch's history (a new crop is a new energy
    scale), which would otherwise permanently disarm the latch on an
    already-floored trace -- the exact gridbar hang it exists for. The carried
    state stays valid only while the post-crop trace stays near its own floor;
    a significant rise (>= 1.5x the post-crop smoothed minimum, with an
    absolute guard for near-zero floors) means new milling activity and the
    carry must be discarded (fail toward never-complete, not false-complete).
    """
    if not post_crop_history:
        return True
    k = max(1, int(config.smoothing_points))
    smoothed = [
        float(median(post_crop_history[max(0, i + 1 - k): i + 1]))
        for i in range(len(post_crop_history))
    ]
    lo, hi = min(smoothed), max(smoothed)
    return hi <= max(1.5 * lo, lo + config.stall_abs_tolerance)


def apply_stall_carry(
    stall: StallResult,
    post_crop_history: list[float],
    config: StallConfig,
    carry_active: bool,
) -> tuple[StallResult, bool]:
    """
    Fold a dropped-state carried across a crop change into this round's stall
    evaluation.

    Returns ``(possibly-updated result, carry flag to persist)``. The carry
    only assists the ``dropped`` condition -- flatness must still be earned
    over a full post-crop window, and the carry is discarded the moment the
    post-crop trace shows renewed activity (see stall_carry_still_valid) or
    once the post-crop history establishes ``dropped`` on its own.
    """
    if not carry_active:
        return stall, False
    if stall.dropped:
        return stall, False  # re-armed naturally; carry no longer needed
    if not stall_carry_still_valid(post_crop_history, config):
        return stall, False
    if stall.window_span is not None and stall.flat:
        return replace(stall, dropped=True, latched=True), True
    return stall, True


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
    # Which path satisfied the foreground criterion this round: "ABS" (below the
    # absolute threshold) or "STALL" (grid-bar stall latch). None when the
    # criterion failed, is disabled, or has not been evaluated. Surfaced so the
    # GUI/metadata can flag stall-latched completions prominently.
    pixels_via: str | None = None

    # Flags
    all_criteria_met: bool = False       # Whether current round meets all criteria
    # True when ANY round of the current confirmation streak passed the
    # foreground criterion via the stall latch. pixels_via alone reflects only
    # the latest round; a streak can mix ABS and STALL rounds (raw value
    # straddling the threshold while the smoothed latch holds), and a
    # completion where the latch was load-bearing on any round must be flagged
    # for operator review.
    stall_used_in_streak: bool = False

    def reset_confirmation(self):
        """Reset confirmation counter (called when criteria not met)."""
        self.confirmation_count = 0
        self.all_criteria_met = False
        self.stall_used_in_streak = False
    
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

    # Foreground completion mode. ABSOLUTE keeps the original "energy <=
    # maximum_pixels_threshold" behavior; ABSOLUTE_PLUS_STALL additionally
    # passes when the stall latch (computed by the caller via
    # evaluate_foreground_stall and passed into check_completion_criteria)
    # reports the trace has floored.
    foreground_completion_mode: ForegroundCompletionMode = ForegroundCompletionMode.ABSOLUTE_PLUS_STALL


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
    # "ABS" / "STALL" when pixels_ok passed via that path; None otherwise.
    pixels_via: str | None = None

    @property
    def all_met(self) -> bool:
        return self.slope_ok and self.match_ok and self.pixels_ok


def evaluate_pattern_completion(
    state: EvaluationState,
    criteria: EvaluationCriteria,
    mean_pixel_slope: float,
    match_score: float | None,
    white_pixels_percentage: float,
    delay_active: bool,
    stall: StallResult | None = None,
    log_tag: str = ""
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
    :param match_score: Current pattern match score, or None when no score could
        be computed this round (fails the match criterion closed)
    :param white_pixels_percentage: Current white pixel percentage
    :param delay_active: Whether delay processing is currently active
    :param stall: Stall-latch evaluation for this round (from
        evaluate_foreground_stall over the foreground history), or None when
        unavailable -- the stall path then simply cannot pass
        (ABSOLUTE_PLUS_STALL degrades to plain ABSOLUTE; fails closed)
    :param log_tag: Optional prefix (e.g. "Pattern 1: ") for log attribution --
        without it, dual-pattern runs produce indistinguishable
        "Criteria met" lines (2026-08-17 triage pain)
    :return: Updated evaluation state
    """
    # Update current measurements for logging/display
    state.current_slope = mean_pixel_slope
    state.current_match_score = match_score
    state.current_white_pixels = white_pixels_percentage
    
    # Skip evaluation during delay phase
    if delay_active:
        logger.debug(f"{log_tag}Delay active - skipping criteria evaluation")
        return state

    # Skip evaluation if already complete
    if state.is_complete:
        logger.debug(f"{log_tag}Pattern already complete - skipping evaluation")
        return state

    # Check all enabled criteria (returns the per-criterion breakdown)
    result = check_completion_criteria(
        mean_pixel_slope=mean_pixel_slope,
        match_score=match_score,
        white_pixels_percentage=white_pixels_percentage,
        criteria=criteria,
        stall=stall,
        log_tag=log_tag
    )

    # Surface the authoritative per-criterion status so the GUI displays exactly
    # what the counter is based on.
    state.slope_ok = result.slope_ok
    state.match_ok = result.match_ok
    state.pixels_ok = result.pixels_ok
    state.pixels_via = result.pixels_via

    # Update confirmation counter based on criteria
    if result.all_met:
        if result.pixels_via == "STALL":
            state.stall_used_in_streak = True
        state.increment_confirmation()
        logger.info(
            f"{log_tag}Criteria met - confirmation round "
            f"{state.confirmation_count}/{criteria.confirmation_rounds}"
        )

        # Check if we've reached required confirmation rounds
        if state.confirmation_count >= criteria.confirmation_rounds:
            state.mark_complete()
            logger.info(f"{log_tag}Pattern complete")
    else:
        # Criteria not met - reset counter
        if state.confirmation_count > 0:
            logger.info(f"{log_tag}Criteria not met - resetting confirmation counter")
        state.reset_confirmation()
    
    return state


def check_completion_criteria(
    mean_pixel_slope: float,
    match_score: float | None,
    white_pixels_percentage: float,
    criteria: EvaluationCriteria,
    stall: StallResult | None = None,
    log_tag: str = ""
) -> CriteriaResult:
    """
    Check the enabled completion criteria and return the per-criterion breakdown.

    Only enabled criteria are evaluated. Disabled criteria automatically pass.
    If no criteria are enabled, every criterion passes (rounds pass to completion).

    :param mean_pixel_slope: Current slope (absolute value)
    :param match_score: Current match score, or None when unavailable this round.
        None FAILS the (enabled) match criterion -- an unavailable score must
        never read as a match, or a broken input could stop milling early.
    :param white_pixels_percentage: Current white pixel / foreground-energy value
    :param criteria: Threshold values and enabled flags
    :param stall: Stall-latch evaluation for this round (ABSOLUTE_PLUS_STALL
        only), or None when unavailable -- the stall path then cannot pass
    :param log_tag: Optional prefix (e.g. "Pattern 1: ") for log attribution
    :return: CriteriaResult with slope_ok/match_ok/pixels_ok/pixels_via (and .all_met)
    """
    # Each criterion: enabled → check threshold, disabled → auto-pass (True)
    slope_ok = (abs(mean_pixel_slope) <= criteria.mean_pixel_slope_threshold
                if criteria.mean_slope_enabled else True)

    match_ok = ((match_score is not None
                 and match_score <= criteria.match_score_threshold)
                if criteria.match_score_enabled else True)

    # Foreground criterion: absolute level, optionally OR'd with the grid-bar
    # stall latch (ABSOLUTE_PLUS_STALL). The stall path exists because exposed
    # static material (e.g. a grid bar) floors the trace at a sample-dependent
    # non-zero level that the absolute threshold cannot be tuned to sit above
    # in advance (2026-08-17 Runs 11/12 hung exactly this way).
    pixels_via = None
    if not criteria.percent_pixels_enabled:
        pixels_ok = True
        pixels_detail = f"Foreground: {white_pixels_percentage:.2f} [DISABLED*]"
    else:
        abs_ok = white_pixels_percentage <= criteria.maximum_pixels_threshold
        stall_allowed = (criteria.foreground_completion_mode
                         == ForegroundCompletionMode.ABSOLUTE_PLUS_STALL)
        stall_ok = bool(stall_allowed and stall is not None and stall.latched)
        pixels_ok = abs_ok or stall_ok
        if abs_ok:
            pixels_via = "ABS"
        elif stall_ok:
            pixels_via = "STALL"

        abs_detail = (
            f"{white_pixels_percentage:.2f} <= "
            f"{criteria.maximum_pixels_threshold} [{'OK' if abs_ok else 'FAIL'}]"
        )
        if stall_allowed and stall is not None:
            if stall.window_span is None:
                stall_detail = "stall: window not yet filled [not latched]"
            else:
                stall_detail = (
                    f"stall: value {stall.smoothed:.2f} vs peak "
                    f"{stall.baseline:.2f} (drop {'OK' if stall.dropped else 'FAIL'}), "
                    f"span {stall.window_span:.3f} (flat {'OK' if stall.flat else 'FAIL'}) "
                    f"[{'LATCHED' if stall.latched else 'not latched'}]"
                )
            pixels_detail = f"Foreground[ABS+STALL]: {abs_detail} | {stall_detail}"
        elif stall_allowed:
            pixels_detail = f"Foreground[ABS+STALL]: {abs_detail} | stall: n/a"
        else:
            pixels_detail = f"Foreground[ABS]: {abs_detail}"

    # Log individual criterion status for debugging
    match_str = "n/a" if match_score is None else f"{match_score:.4f}"
    logger.debug(
        f"{log_tag}Criteria check - "
        f"Slope: {abs(mean_pixel_slope):.4f} <= {criteria.mean_pixel_slope_threshold} "
        f"[{'OK' if slope_ok else 'FAIL'}{'*' if not criteria.mean_slope_enabled else ''}], "
        f"Match: {match_str} <= {criteria.match_score_threshold} "
        f"[{'OK' if match_ok else 'FAIL'}{'*' if not criteria.match_score_enabled else ''}], "
        f"{pixels_detail}"
    )

    return CriteriaResult(slope_ok=slope_ok, match_ok=match_ok,
                          pixels_ok=pixels_ok, pixels_via=pixels_via)


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