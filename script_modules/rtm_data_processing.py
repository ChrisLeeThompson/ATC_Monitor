"""
Real-Time Monitor (RTM) processing.
This module provides functions to convert RTM data into displayable images.
"""

import logging
import math
from collections import namedtuple

import numpy


logger = logging.getLogger(__name__)


def get_images_from_rtm_data(rtm_data, rtm_positions, pattern_metadata=None):
    """
    Convert RTM data to image arrays, keyed by pattern_id.

    Returns a dict mapping each pattern's ``pattern_id`` to its 2D numpy array
    (cropped to the valid region), or to None if no RTM data was available for
    that pattern. Keying by pattern_id -- rather than returning a list aligned to
    the order of rtm_positions -- lets the caller route each image to the correct
    pattern even when the microscope returns the RTM patterns in a different order
    between frames (otherwise the two patterns' images get swapped; see
    workflow_worker._process_rtm_data).

    :param rtm_data: List of RTM data sets containing pattern_id and intensity values
    :param rtm_positions: List of RTM position patterns containing pattern_id and positions
    :param pattern_metadata: Pre-computed pattern bounds and index arrays to avoid recalculation. Keys are pattern_ids.
    """

    result = {}

    for pattern in rtm_positions:
        try:
            data_set = next(x for x in rtm_data if x.pattern_id == pattern.pattern_id)
        except StopIteration:
            logger.debug(
                f"No RTM data for pattern {pattern.pattern_id} - "
                f"inserting None placeholder"
            )
            result[pattern.pattern_id] = None
            continue
        
        # Use cached metadata if available (includes pre-computed index arrays)
        if pattern_metadata and pattern.pattern_id in pattern_metadata:
            meta = pattern_metadata[pattern.pattern_id]
            x_min, x_max = meta['x_min'], meta['x_max']
            y_min, y_max = meta['y_min'], meta['y_max']
            y_indices = meta['y_indices']
            x_indices = meta['x_indices']
        else:
            x_coords = [p[0] for p in pattern.positions]
            y_coords = [p[1] for p in pattern.positions]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            y_indices = [pattern.positions[i][1] - y_min for i in range(1, len(pattern.positions))]
            x_indices = [pattern.positions[i][0] - x_min for i in range(1, len(pattern.positions))]
        
        # Create array
        arr = numpy.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        
        # Vectorized fill using cached index arrays
        arr[y_indices, x_indices] = data_set.values[1:]
        
        # Crop
        valid_rows = numpy.any(arr > -1, axis=1)
        valid_cols = numpy.any(arr > -1, axis=0)
        arr = arr[numpy.ix_(valid_rows, valid_cols)]

        result[pattern.pattern_id] = arr

    return result


def precompute_pattern_metadata(rtm_positions):
    """
    Pre-compute pattern bounds and index offset arrays for performance optimization.
    
    Caches the offset index arrays that get_images_from_rtm_data uses for
    vectorized array filling, eliminating per-call list comprehensions.
    
    :param rtm_positions: List of RTM position patterns
    :return: Dictionary mapping pattern_id to bounds and index metadata
    """
    
    pattern_metadata = {}
    
    for pattern in rtm_positions:
        x_coords = [p[0] for p in pattern.positions]
        y_coords = [p[1] for p in pattern.positions]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Pre-compute offset index arrays (skip position 0 per existing logic)
        y_indices = [pattern.positions[i][1] - y_min for i in range(1, len(pattern.positions))]
        x_indices = [pattern.positions[i][0] - x_min for i in range(1, len(pattern.positions))]
        
        pattern_metadata[pattern.pattern_id] = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'y_indices': y_indices,
            'x_indices': x_indices
        }

    return pattern_metadata


# ===========================
# Auto contrast/brightness: balance metric + control law (pure functions)
# ===========================
#
# These are deliberately hardware-free so they can be unit-tested over synthetic
# histograms. The worker (workflow_worker.WorkflowWorker._calibrate_detector_cb)
# does the I/O (grab RTM frame -> measure -> write detector CB -> settle) and
# uses these to decide the next contrast/brightness values.

# Per-measurement balance of an image's valid pixels.
#   white_clip / black_clip : fraction of valid pixels pinned at the ceiling / floor
#   median_fraction         : median(valid) / white_level  (0..1)
#   n_valid                 : number of valid (>= 0) pixels measured
#   vmin / vmax             : observed min / max of valid pixels (raw counts)
#   span_fraction           : robust occupied span (p_hi - p_lo) / white_level (0..1),
#                             using symmetric percentiles so a single hot/dead pixel
#                             does not skew it (unlike raw vmax - vmin). This is the
#                             feedback signal for the contrast servo.
CBBalance = namedtuple(
    "CBBalance",
    ["white_clip", "black_clip", "median_fraction", "n_valid", "vmin", "vmax",
     "span_fraction"],
)

# Tuning for the closed-loop controller. Bounds/targets come from user settings;
# the gains/step/tol are control parameters.
#   brightness_gain      : proportional gain mapping the median error (fraction) to a
#                          brightness adjustment -- shrinks the step near target so the
#                          loop converges instead of oscillating by a fixed step.
#   clip_gain            : proportional gain mapping clip-fraction excess to brightness.
#   brightness_step      : per-iteration CAP on |brightness change| (anti-windup).
#   target_contrast_span : desired robust occupied span (p_hi - p_lo)/white_level; the
#                          contrast servo's setpoint (mirrors target_median_fraction).
#   contrast_gain        : proportional gain mapping the span error to a contrast change.
#   contrast_tol         : convergence band on the span error.
#   contrast_step        : per-iteration CAP on |contrast change| (anti-windup).
#   span_floor           : below this span (flat/degenerate field) the contrast servo
#                          holds -- there is no real contrast to chase, so don't slam
#                          the gain to its cap amplifying noise.
CBControlConfig = namedtuple(
    "CBControlConfig",
    [
        "white_level",
        "target_median_fraction",
        "max_white_clip",
        "max_black_clip",
        "min_bound",
        "max_bound",
        "brightness_step",
        "brightness_gain",
        "clip_gain",
        "contrast_step",
        "median_tol",
        "target_contrast_span",
        "contrast_gain",
        "contrast_tol",
        "span_floor",
    ],
)


# Saturation-aware interior. The median/span that the brightness/contrast servos
# regulate are measured over the NON-rail interior (0 < x < white_level), so a wall
# of pixels pinned at a rail can't drag those statistics (which would otherwise pull
# the median across target and cause a false move). Clip fractions are still measured
# over ALL valid pixels (the operator's clip budget). Fall back to all valid pixels
# when the interior is too thin to be a stable estimate.
_CB_MIN_INTERIOR_FRACTION = 0.02   # interior must be >= 2% of valid pixels ...
_CB_MIN_INTERIOR_COUNT = 50        # ... and >= this many, else use all valid pixels

# Declip / cost / acceptance tunables (internal; promotable to CBControlConfig if
# operators ever need them).
_CB_TAINTED_SPAN_PENALTY = 0.15  # ADDED to the span-error cost term when a clipped
                                 # rail pins a percentile and fakes an on-target span
                                 # (additive since v3.3.8; was a floor)
_CB_HARD_CLIP_FRACTION = 0.30    # clip fraction at/above which a rail DOMINATES the
                                 # histogram: the span reading is meaningless and the
                                 # cure is prescribed by which rail it is
_CB_STALL_EPS = 0.02             # all four balance metrics moving less than this
                                 # after a knob change counts as a stall (0.02, not
                                 # 0.01: 1-frame field noise is ~0.01-0.08, so the
                                 # old value never fired on-tool; the saturated
                                 # regime where stalls matter is rail-pinned and
                                 # nearly noiseless, so 0.02 still fires there)
_CB_STEP_GROWTH = 2.0            # walk/declip step escalation multiplier
_CB_SPAN_RECOVERY_TRIES = 3      # bounded contrast-up attempts on a clean but
                                 # degenerate (span <= floor) field

# Clip hysteresis: the field 2026-08-14 limit cycle was declip acting at exactly the
# acceptance threshold while measurement noise straddled it (bclip 0.015-0.05 bounced
# a declip/recenter tug-of-war for 10+ measurements). ACCEPT tolerates a small margin
# over the user's clip limit (also ~where percentile pinning would begin to bias the
# span reading); ACT fires only well above it. Readings inside the deadband neither
# block acceptance nor trigger clip moves.
_CB_CLIP_ACCEPT_MARGIN = 0.01
_CB_CLIP_ACT_MARGIN = 0.04

# Termination / verification.
_CB_PATIENCE = 4                 # consecutive measurements with neither a best-cost
                                 # improvement nor bracket progress -> lock best
_CB_MIN_COST_IMPROVEMENT = 0.02  # improvement smaller than this is noise, not progress
_CB_MAX_VERIFY_RETRIES = 2       # pooled-verify attempts before letting patience end it

# Slope learning trust: a measured response smaller than these is inside the
# 1-frame noise floor, so its SIGN is a coin flip -- _measured_slope would floor
# the magnitude but keep the noise-determined sign, and a wrong-sign slope sends
# every subsequent secant the wrong way (review finding, 2026-08-15). Responses
# below the floor neither teach a slope nor contradict one.
_CB_MIN_LEARN_MEDIAN = 0.08      # min |d median| to trust a brightness-slope sign
_CB_MIN_LEARN_LOGSPAN = 0.10     # min |d log10 span| to trust a contrast-slope sign

# Contrast bracket (the searched window on the dB knob).
_CB_BRACKET_MIN_WIDTH = 0.01     # bracket narrower than this while still clipped means
                                 # no usable window exists at this brightness
_CB_BRACKET_B_DRIFT = 0.05       # a bracket endpoint is only valid while brightness is
                                 # within this of where the endpoint was measured

# Brightness secant jump cap: the knob is linear so a deadbeat jump is legitimate,
# but cap it at a few base steps as anti-windup against a noise-corrupted slope.
_CB_BRIGHTNESS_JUMP_CAP = 0.15

# Contrast secant jump cap. The contrast slope is learned OPPORTUNISTICALLY from
# clean same-brightness walk pairs (never from a dedicated probe, which saturates
# on narrow-window tools). On a gentle plant the learned slope produces long,
# legitimate jumps (a gspan~20dB tool needs most of the knob); on a harsh plant
# the steep learned slope yields tiny steps naturally. The cap bounds the damage
# of a floor-limited bogus slope to one recoverable overshoot (the bracket and
# bisection absorb it).
_CB_CONTRAST_JUMP_CAP = 0.3

# cb_cost clip weights: white clipping saturates exactly the bright texture the
# downstream top-hat energy measures; black pixels are background the top-hat removes
# anyway. Weight white-clip excess above black accordingly.
_CB_COST_WHITE_CLIP_WEIGHT = 2.5
_CB_COST_BLACK_CLIP_WEIGHT = 1.5


def _balance_stalled(prev, cur, eps=_CB_STALL_EPS):
    """True when a knob change produced no measurable movement in any balance metric."""
    if prev is None or cur is None:
        return False
    return (abs(cur.white_clip - prev.white_clip) < eps
            and abs(cur.black_clip - prev.black_clip) < eps
            and abs(cur.median_fraction - prev.median_fraction) < eps
            and abs(cur.span_fraction - prev.span_fraction) < eps)


def compute_cb_balance(image, white_level, contrast_percentile=2.0):
    """
    Measure clipping/brightness balance of an image's valid pixels.

    Accepts any array shape (a single pattern image or a pooled 1-D array of
    pixels from several frames/patterns). Pixels < 0 are the RTM "missing"
    sentinel and are excluded.

    :param image: numpy array (or array-like) of raw detector counts
    :param white_level: detector full-scale / saturation ceiling (raw counts)
    :param contrast_percentile: P for the robust occupied span -- the span runs
        from the P-th to the (100 - P)-th percentile, so up to ~P% of outliers at
        each rail (a hot/dead pixel, a saturated speck) do not inflate it. P=2.0
        uses p2..p98.
    :return: a CBBalance, or None when there are no valid pixels / invalid level
    """
    arr = numpy.asarray(image)
    valid = arr[arr >= 0].astype(float)
    n_valid = int(valid.size)
    if n_valid == 0 or white_level <= 0:
        return None

    vmin = float(valid.min())
    vmax = float(valid.max())
    # Clip fractions: over ALL valid pixels (the operator's clip budget).
    white_clip = float(numpy.count_nonzero(valid >= white_level) / n_valid)
    black_clip = float(numpy.count_nonzero(valid <= 0.0) / n_valid)

    # Median / span: over the non-rail interior to remove rail-pile-up bias. Fall
    # back to all valid pixels when the interior is too thin (near-total clipping).
    interior = valid[(valid > 0.0) & (valid < white_level)]
    min_interior = max(_CB_MIN_INTERIOR_COUNT, _CB_MIN_INTERIOR_FRACTION * n_valid)
    reg = interior if interior.size >= min_interior else valid

    median_fraction = float(numpy.median(reg) / white_level)
    # Robust occupied span: percentile spread, not raw vmax - vmin, so a single
    # outlier pixel cannot widen it. Clamp the percentile into a sane range.
    p = min(49.0, max(0.0, float(contrast_percentile)))
    p_lo, p_hi = numpy.percentile(reg, [p, 100.0 - p])
    span_fraction = float((p_hi - p_lo) / white_level)
    return CBBalance(white_clip, black_clip, median_fraction, n_valid, vmin, vmax,
                     span_fraction)


def next_cb_step(balance, contrast, brightness, cfg):
    """
    Decide the next detector contrast/brightness from the current balance.

    TWO damped-proportional servos run as one co-converging loop, GATED on clipping:

      Brightness (detector offset) -> drives the median to target_median_fraction;
      biased by any clip-limit excess so clipping is relieved first. Change is
      proportional to the error, capped at brightness_step, so the step shrinks near
      target and the loop converges instead of limit-cycling by a fixed step.

      Contrast (detector gain) -> drives the robust occupied span (span_fraction) to
      target_contrast_span, proportional and capped at contrast_step. It is only run
      when NO rail is clipping, because a clipped rail pins a percentile and makes the
      span reading a lie (it would race the brightness clip-relief into an oscillation).

    Precedence each iteration:
      P1  both rails clipped  -> brightness can't help; compress range (contrast - step).
      P2  one rail  clipped   -> relieve with brightness; FREEZE contrast.
      P3  no clipping         -> run BOTH servos (here span<->gain and median<->offset
                                 are nearly independent, so they decouple and converge).

    span_floor holds the contrast servo on a flat/degenerate field (span ~ 0), where
    there is no real contrast to chase and the servo would otherwise slam the gain to
    its cap amplifying noise.

    :param balance: CBBalance from compute_cb_balance
    :param contrast: current normalized contrast in [0, 1]
    :param brightness: current normalized brightness in [0, 1]
    :param cfg: CBControlConfig
    :return: (new_contrast, new_brightness, converged)
    """
    white_over = balance.white_clip > cfg.max_white_clip
    black_over = balance.black_clip > cfg.max_black_clip
    median_err = cfg.target_median_fraction - balance.median_fraction  # +ve => too dark
    median_off = abs(median_err) > cfg.median_tol

    # The span reading is only trustworthy when nothing is pinned at a rail.
    span_trustworthy = not white_over and not black_over
    span_err = cfg.target_contrast_span - balance.span_fraction  # +ve => contrast too low
    span_off = span_trustworthy and abs(span_err) > cfg.contrast_tol

    # Balanced: within both clip limits, the median band, and (when trustworthy) the
    # span band.
    if not white_over and not black_over and not median_off and not span_off:
        return contrast, brightness, True

    new_c = contrast
    new_b = brightness

    if white_over and black_over:
        # P1 -- histogram pinned at BOTH rails: brightness can't fix it -> compress.
        new_c = contrast - cfg.contrast_step
    elif white_over or black_over:
        # P2 -- one rail clipped: relieve with brightness, freeze contrast (the span
        # reading is a pinned-percentile artifact right now).
        b_error = cfg.brightness_gain * median_err
        if white_over:
            b_error -= cfg.clip_gain * (balance.white_clip - cfg.max_white_clip)
        if black_over:
            b_error += cfg.clip_gain * (balance.black_clip - cfg.max_black_clip)
        step = max(-cfg.brightness_step, min(cfg.brightness_step, b_error))
        new_b = brightness + step
    else:
        # P3 -- no clipping: run both decoupled servos.
        b_step = cfg.brightness_gain * median_err
        b_step = max(-cfg.brightness_step, min(cfg.brightness_step, b_step))
        new_b = brightness + b_step
        # Contrast servo, with the degenerate-field guard.
        if span_off and balance.span_fraction > cfg.span_floor:
            c_step = cfg.contrast_gain * span_err
            c_step = max(-cfg.contrast_step, min(cfg.contrast_step, c_step))
            new_c = contrast + c_step

    new_c = min(cfg.max_bound, max(cfg.min_bound, new_c))
    new_b = min(cfg.max_bound, max(cfg.min_bound, new_b))
    return new_c, new_b, False


def cb_clip_acceptable(balance, cfg):
    """
    True when both clip fractions are within the user limit plus the ACCEPT margin.

    This is the acceptance-side clip test (used by cb_converged and the span-trust
    decision in cb_cost). The margin exists because a 1-frame clip reading has noise
    of the same order as the limit itself; the pooled verify measurement re-checks
    any accepted point at lower noise.
    """
    return (balance.white_clip <= cfg.max_white_clip + _CB_CLIP_ACCEPT_MARGIN
            and balance.black_clip <= cfg.max_black_clip + _CB_CLIP_ACCEPT_MARGIN)


def cb_clip_actionable(balance, cfg):
    """
    True when either clip fraction is far enough over the user limit that a clip-
    relief move should fire. Readings between ACCEPT and ACT are a deadband: not
    accepted as converged-clean, but not worth chasing either (the 2026-08-14 field
    limit cycle lived entirely inside that band).
    """
    return (balance.white_clip > cfg.max_white_clip + _CB_CLIP_ACT_MARGIN
            or balance.black_clip > cfg.max_black_clip + _CB_CLIP_ACT_MARGIN)


def cb_cost(balance, cfg):
    """
    Scalar "badness" of a measured balance, for best-so-far selection (lower = better).

    A static one-shot calibration must lock the lowest-cost CB it actually measured,
    not whatever the final iteration wrote. Terms:
      - median error vs target,
      - span error vs target; when a clipped rail taints the reading (a pinned
        percentile can fake an on-target span) a pessimistic penalty is ADDED to
        the term (additive since v3.3.8; the old floor made every tainted span
        term identical, collapsing clipped candidates' ordering to median error
        alone) -- dropping it entirely made clipped baselines artificially
        cheap and let them win best-so-far over clean, nearly-converged points,
      - clipping BEYOND the user's limit; white excess is weighted above black
        because saturation destroys exactly the bright texture the downstream
        top-hat energy measures, while black pixels are background it removes.

    :param balance: CBBalance from compute_cb_balance
    :param cfg: CBControlConfig
    :return: non-negative float; 0.0 means on every target with no excess clipping
    """
    m_err = abs(cfg.target_median_fraction - balance.median_fraction)
    white_excess = max(0.0, balance.white_clip - cfg.max_white_clip)
    black_excess = max(0.0, balance.black_clip - cfg.max_black_clip)
    span_trust = cb_clip_acceptable(balance, cfg)
    raw_s_err = abs(cfg.target_contrast_span - balance.span_fraction)
    # Tainted span: ADD the pessimistic penalty instead of flooring at it, so
    # two clipped candidates still order by their real span error (the floor
    # made every tainted span term identical, collapsing their ordering to
    # median error alone -- part of the 2026-08-20 dark-lock preference).
    s_err = raw_s_err if span_trust else raw_s_err + _CB_TAINTED_SPAN_PENALTY
    return (m_err + s_err + _CB_COST_WHITE_CLIP_WEIGHT * white_excess
            + _CB_COST_BLACK_CLIP_WEIGHT * black_excess)


# ===========================
# Bracket -> bisect calibration (side-classification control)
# ===========================
#
# The contrast knob is logarithmic -- the SDK defines it as "contrast voltage
# converted to decibels then normalized", and the measured pixel span scales with
# the *linear* gain, so span-vs-knob is exponential. On some tools (2026-08-14
# field campaign, Hydra) the entire usable window between an all-black and an
# all-white image is ~0.03-0.05 knob units -- NARROWER than any safe fixed probe
# perturbation, so slope-probing the contrast knob saturates the image and
# measures nothing. The orchestrator below therefore never estimates a contrast
# slope: it classifies each measurement by SIDE (gain too high / too low / in
# window), brackets the window geometrically, and bisects in knob space -- over a
# window that narrow the dB/linear distinction is irrelevant, and side
# classification of a hard-clipped frame is nearly immune to measurement noise.
# Brightness is a ~linear volt offset; its median slope IS measured (from the
# run's own moves) and used for a capped linear secant.
#
#   contrast_delta   : SEED step of the contrast bracket walk (grows x2 per
#                      consecutive same-direction move, so any window width is
#                      reached in logarithmic time; per-tool learned values can
#                      be passed here to start pre-adapted).
#   brightness_delta : first brightness move when no median slope is known yet
#                      (the move doubles as the probe).
#   slope_floor_brightness : minimum trusted |d(median)/d(brightness)|; guards a
#                      quantized difference from exploding the secant step.
#   slope_floor_contrast : retained for config compatibility (no contrast slope
#                      is estimated anymore).
#   min_knob_delta   : a brightness change must be at least this large for a
#                      measured slope to be trusted.
#   initial_k_brightness / initial_k_contrast : optional slope SEEDS learned from
#                      a previous run on the same tool (both slopes are plant
#                      properties, independent of the scene: d(median)/d(b) = the
#                      offset gain, d(log10 span)/d(c) = the dB/knob mapping).
#                      The run's own measurements overwrite them as soon as a
#                      trustworthy pair exists, so a stale seed costs at most one
#                      capped, bracket-clamped jump.
CBProbeConfig = namedtuple(
    "CBProbeConfig",
    ["contrast_delta", "brightness_delta", "slope_floor_contrast",
     "slope_floor_brightness", "min_knob_delta",
     "initial_k_brightness", "initial_k_contrast"],
    defaults=[None, None],
)

# Outcome of one calibration run. `contrast`/`brightness` is the CB to lock (the
# lowest-cost point measured, or the accepted point); `status` is a short tag.
# `plant` reports what the run LEARNED about this tool's response (measured
# brightness slope, the bracketed contrast window) so the worker can persist it
# per-microscope and seed the next run -- the controller never assumes a plant,
# it discovers one, and this is how the discovery carries across runs/tools.
CBResult = namedtuple(
    "CBResult",
    ["contrast", "brightness", "converged", "cost", "n_measurements", "status",
     "plant", "best_clean"],
    # best_clean: (contrast, brightness) of the lowest-cost measurement whose
    # clips were NOT actionable, or None if no such point was measured. The
    # worker's commit check reverts to it instead of blindly to the start.
    defaults=[None, None],
)


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))


def cb_converged(balance, cfg):
    """
    True when a balance is ACCEPTABLE: clip within the user limits plus the accept
    margin, and median/span inside their bands around the targets.

    The bands are cfg.median_tol / cfg.contrast_tol, which the worker now ships
    WIDE (0.20 / 0.25): the downstream consumers (rescaling filter, top-hat energy,
    scale-invariant matcher) need usable dynamic range, not an exact histogram, and
    the old 1-sigma point gates were statistically unreachable under real
    measurement noise -- 8 of 9 field episodes ended budget-exhausted chasing them.
    """
    if not cb_clip_acceptable(balance, cfg):
        return False
    median_off = abs(cfg.target_median_fraction - balance.median_fraction) > cfg.median_tol
    span_off = abs(cfg.target_contrast_span - balance.span_fraction) > cfg.contrast_tol
    return not median_off and not span_off


def secant_step(stat, target, knob, slope, step_cap, lo, hi, log_space=False):
    """
    One deadbeat/secant move of a single knob to drive ``stat`` toward ``target``.

    ``slope`` is the locally-measured sensitivity d(stat)/d(knob), or -- when
    ``log_space`` -- d(log10 stat)/d(knob), which is the right coordinate for the dB
    contrast knob (linear gain, and therefore span, is exponential in the knob). The
    step is capped at ``step_cap`` (anti-windup) and clamped to [lo, hi]. Returns
    ``knob`` unchanged when the move is undefined (zero slope; non-positive stat/target
    in log space).
    """
    if slope == 0.0:
        return knob
    if log_space:
        if stat <= 0.0 or target <= 0.0:
            return knob
        err = math.log10(target) - math.log10(stat)
    else:
        err = target - stat
    step = _clamp(err / slope, -step_cap, step_cap)
    return _clamp(knob + step, lo, hi)


def _measured_slope(stat0, stat1, knob0, knob1, log_space, floor, min_knob_delta):
    """
    Estimate d(stat)/d(knob) (or d(log10 stat)/d(knob)) from two measured points, or
    None when untrustworthy. Returns None if the knobs are too close to divide safely
    (an 8-bit-quantized difference would be noise-dominated). Otherwise floors the
    magnitude at ``floor`` (keeping sign) so a near-zero measured slope can't explode a
    deadbeat step.
    """
    dk = knob1 - knob0
    if abs(dk) < min_knob_delta:
        return None
    if log_space:
        if stat0 <= 0.0 or stat1 <= 0.0:
            return None
        dstat = math.log10(stat1) - math.log10(stat0)
    else:
        dstat = stat1 - stat0
    slope = dstat / dk
    if slope == 0.0:
        return None
    return math.copysign(max(abs(slope), floor), slope)


def _signed_probe_delta(knob, delta, lo, hi):
    """Pick a probe perturbation that stays inside [lo, hi] (probe inward near a bound)."""
    if knob + delta <= hi:
        return delta
    if knob - delta >= lo:
        return -delta
    up, down = hi - knob, knob - lo
    return up if up >= down else -down


def run_cb_calibration(measure_at, contrast, brightness, cfg, probe, budget,
                       note=None, verify_at=None):
    """
    One-shot detector contrast/brightness calibration: classify -> cure/bracket ->
    bisect -> brightness secant -> band-accept (pooled verify), locking the
    lowest-cost CB actually measured when no point is accepted.

    Strategy (see the section comment above CBProbeConfig for the physics):

      1. accept immediately when the current balance is in band -- confirmed with
         one pooled ``verify_at`` measurement when provided;
      2. cure rail-dominated states by the field-hardened rule table
         (offset-driven white -> brightness down; gain-driven white -> contrast
         down; black -> brightness up), stepping geometrically so bit-identical
         frames cannot stall the walk;
      3. record every measurement as contrast-bracket evidence ("gain too high" /
         "gain too low") and BISECT the knob-space bracket once both sides exist
         at the current brightness -- the bracket endpoints double as visited-
         state memory, so ping-ponging between the same two points is
         structurally impossible;
      4. recenter the median with a measured-slope linear secant on brightness
         (the slope is learned from the run's own moves; the first move doubles
         as the probe);
      5. terminate on acceptance, on patience (no cost improvement and no bracket
         progress for _CB_PATIENCE measurements -> lock best-so-far), or on
         budget exhaustion.

    Pure orchestration -- all hardware I/O is behind ``measure_at``/``verify_at``
    so this runs in tests against a synthetic detector. ``measure_at(c, b)`` must
    apply the CB, settle, and return ``(CBBalance | None, c_applied, b_applied)``;
    the applied values may differ from the request if the detector clamps them,
    and the orchestrator tracks the applied values as the true knob position.
    ``verify_at`` has the same contract but should pool more frames (lower
    noise); when provided, every acceptance is re-confirmed through it and the
    verify measurement counts against the budget.

    :param measure_at: callable(c, b) -> (CBBalance | None, c_applied, b_applied)
    :param contrast: starting normalized contrast in [0, 1]
    :param brightness: starting normalized brightness in [0, 1]
    :param cfg: CBControlConfig (targets / bands / clip limits / bounds)
    :param probe: CBProbeConfig (walk seed + brightness probe/slope trust)
    :param budget: max number of measurements (== cb_max_iterations), verify incl.
    :param note: optional callable(str) for progress logging
    :param verify_at: optional pooled-frames measurement callable (same contract)
    :return: CBResult (``.plant`` reports the learned tool response)
    """
    lo, hi = cfg.min_bound, cfg.max_bound
    budget = max(1, int(budget))
    span_lo_band = cfg.target_contrast_span - cfg.contrast_tol
    span_hi_band = cfg.target_contrast_span + cfg.contrast_tol
    seed_c = max(1e-3, probe.contrast_delta)

    def _finite_or_none(value):
        # A persisted seed can be hand-edited or corrupted into NaN/Infinity
        # (json accepts them); a non-finite slope makes every secant degenerate.
        return value if (value is not None and math.isfinite(value)) else None

    state = {
        "c": contrast, "b": brightness, "n": 0, "bal": None,
        "best_cost": float("inf"), "best_c": contrast, "best_b": brightness,
        "best_actionable": False,       # clips at the raw cost-argmin were actionable
        "best_wclip": None, "best_bclip": None,
        # Lowest-cost point whose clips were NOT actionable. Locking a clipped
        # cost-argmin at patience only feeds the commit check a point it will
        # revert (2026-08-20: an infeasible target-median/black-clip pair made
        # the argmin a bclip=0.14 point every session).
        "best_clean_cost": float("inf"), "best_clean_c": None, "best_clean_b": None,
        "k_b": _finite_or_none(probe.initial_k_brightness),
        "k_c": _finite_or_none(probe.initial_k_contrast),
        "k_b_learned": False,           # True once THIS run measured the slope
        "k_c_learned": False,
        "k_b_contradicted": False,      # a seed was invalidated by a measured
                                        # opposite-sign response (poison marker)
        "c_lo": None, "c_lo_b": None,   # highest gain-too-low contrast (+ its b)
        "c_hi": None, "c_hi_b": None,   # lowest gain-too-high contrast (+ its b)
        "b_rail_free": None,            # last brightness whose frame was not rail-dominated
        "esc_c": seed_c, "esc_b": cfg.brightness_step,
        "last_c_dir": 0, "last_b_dir": 0,
        "since_improve": 0, "verify_fails": 0, "span_tries": 0,
        "best_excess": float("inf"),    # least total clip excess seen (progress)
        "prev": None,                   # (bal, c, b) of the latest measurement
        "before": None,                 # (bal, c, b) of the one before it
    }

    def _plant():
        # Report only what THIS run measured: echoing a seed back would let a
        # wrong persisted slope survive forever (it gets re-persisted verbatim
        # and re-seeded on every subsequent run on that tool).
        window = None
        if state["c_lo"] is not None and state["c_hi"] is not None:
            window = (state["c_lo"], state["c_hi"])
        return {"k_brightness": state["k_b"] if state["k_b_learned"] else None,
                "k_contrast": state["k_c"] if state["k_c_learned"] else None,
                "k_brightness_contradicted": state["k_b_contradicted"],
                "c_window": window}

    def _bracket_update(bal, c, b):
        """Classify a measurement as gain-too-high / gain-too-low evidence and
        tighten the contrast bracket. Returns True when an endpoint moved."""
        w_act = bal.white_clip > cfg.max_white_clip + _CB_CLIP_ACT_MARGIN
        b_act = bal.black_clip > cfg.max_black_clip + _CB_CLIP_ACT_MARGIN
        hard_w = bal.white_clip >= _CB_HARD_CLIP_FRACTION
        hard_b = bal.black_clip >= _CB_HARD_CLIP_FRACTION
        span = bal.span_fraction
        # A hard-white frame whose readable interior span already FITS the target
        # is offset-driven -- the gain is not the culprit, so it must not become
        # gain-too-high evidence (it would poison the bracket).
        offset_white = cfg.span_floor < span <= cfg.target_contrast_span
        too_high = ((hard_w and not offset_white and not hard_b)
                    or (w_act and not b_act and span > cfg.target_contrast_span)
                    or (not w_act and not b_act and span > span_hi_band))
        # Symmetrically, hard black with a READABLE span is offset territory; only
        # a collapsed span marks the gain itself as too low. Clean below-band span
        # (including a collapsed one) is direct too-low evidence.
        too_low = ((hard_b and span <= cfg.span_floor)
                   or (not w_act and not b_act and span < span_lo_band))
        changed = False
        if too_high and (state["c_hi"] is None or c < state["c_hi"]):
            state["c_hi"], state["c_hi_b"] = c, b
            changed = True
        if too_low and (state["c_lo"] is None or c > state["c_lo"]):
            state["c_lo"], state["c_lo_b"] = c, b
            changed = True
        return changed

    def _bracket():
        """The contrast bracket, valid only near the brightness it was measured
        at (a brightness move shifts which gains clip). Crossed endpoints (noise
        or drift artifacts) clear the bracket rather than mislead the bisect."""
        c_lo, c_hi = state["c_lo"], state["c_hi"]
        if c_lo is None or c_hi is None:
            return None
        if (abs(state["b"] - state["c_lo_b"]) > _CB_BRACKET_B_DRIFT
                or abs(state["b"] - state["c_hi_b"]) > _CB_BRACKET_B_DRIFT):
            return None
        if c_hi <= c_lo:
            state["c_lo"] = state["c_hi"] = None
            state["c_lo_b"] = state["c_hi_b"] = None
            return None
        return c_lo, c_hi

    def take(c_req, b_req, tag, measure=None):
        measure = measure_at if measure is None else measure
        bal, c_app, b_app = measure(c_req, b_req)
        state["n"] += 1
        prev = state["prev"]
        state["before"] = prev          # the measurement preceding the current one
        state["prev"] = (bal, c_app, b_app)
        state["c"], state["b"], state["bal"] = c_app, b_app, bal
        if bal is None:
            return None
        cost = cb_cost(bal, cfg)
        progressed = cost < state["best_cost"] - _CB_MIN_COST_IMPROVEMENT
        if cost < state["best_cost"]:
            state["best_cost"] = cost
            state["best_c"], state["best_b"] = c_app, b_app
            state["best_actionable"] = cb_clip_actionable(bal, cfg)
            state["best_wclip"], state["best_bclip"] = bal.white_clip, bal.black_clip
        if (not cb_clip_actionable(bal, cfg)
                and cost < state["best_clean_cost"]):
            state["best_clean_cost"] = cost
            state["best_clean_c"], state["best_clean_b"] = c_app, b_app
        # Rail-clearing IS progress even while the cost hovers (deep-clip cures
        # trade rails for several moves before the cost term can fall).
        excess = (max(0.0, bal.white_clip - cfg.max_white_clip)
                  + max(0.0, bal.black_clip - cfg.max_black_clip))
        if excess < state["best_excess"] - 0.05:
            progressed = True
        state["best_excess"] = min(state["best_excess"], excess)
        stalled = (prev is not None and prev[0] is not None
                   and _balance_stalled(prev[0], bal))
        # Bracket evidence is recorded even from a bit-identical frame (a lower
        # contrast that is STILL pinned white genuinely tightens c_hi), but a
        # stalled frame must not count as PROGRESS: the review's offset-overload
        # replay showed a contrast flail through identical pinned frames
        # resetting patience on every step and structurally disabling the stop.
        if _bracket_update(bal, c_app, b_app) and not stalled:
            progressed = True   # structural progress even when cost got worse
        state["since_improve"] = 0 if progressed else state["since_improve"] + 1
        if progressed:
            # A materially better state means the cure is close: stop sprinting.
            # (The brightness ladder otherwise kept doubling past a near-good
            # landing -- ep1 replay overshot b 0.432 -> 0.832 into full white.)
            state["esc_b"] = cfg.brightness_step
        if max(bal.white_clip, bal.black_clip) < _CB_HARD_CLIP_FRACTION:
            # Remember a brightness at which the image was NOT rail-dominated:
            # the offset-bisection cure homes on it when a later frame is
            # fully pinned by the offset.
            state["b_rail_free"] = b_app
        if prev is not None and prev[0] is not None:
            p_bal, p_c, p_b = prev
            d_median = bal.median_fraction - p_bal.median_fraction
            d_b = b_app - p_b
            if abs(c_app - p_c) < 1e-9 and abs(d_b) >= probe.min_knob_delta:
                # Pure-brightness move. A response above the sign-trust floor
                # that CONTRADICTS the current slope invalidates it (a stale or
                # noise-flipped seed would otherwise send every jump the wrong
                # way forever -- no pair in that failure loop is ever clean
                # enough to re-learn through the gate below).
                if (state["k_b"] is not None
                        and abs(d_median) >= _CB_MIN_LEARN_MEDIAN
                        and (d_median > 0) != (state["k_b"] * d_b > 0)):
                    state["k_b"] = None
                    state["k_b_learned"] = False
                    state["k_b_contradicted"] = True
                # Learn only between states where no rail DOMINATES (rail
                # migration fakes wrong-sign slopes) AND the response clears
                # the sign-trust floor (below it the sign is noise).
                rail_free = (max(p_bal.white_clip, p_bal.black_clip)
                             < _CB_HARD_CLIP_FRACTION
                             and max(bal.white_clip, bal.black_clip)
                             < _CB_HARD_CLIP_FRACTION)
                if rail_free and abs(d_median) >= _CB_MIN_LEARN_MEDIAN:
                    k = _measured_slope(p_bal.median_fraction, bal.median_fraction,
                                        p_b, b_app, log_space=False,
                                        floor=probe.slope_floor_brightness,
                                        min_knob_delta=probe.min_knob_delta)
                    if k is not None:
                        state["k_b"] = k
                        state["k_b_learned"] = True
            # Learn the contrast slope (d log10 span / dc) from a clean
            # same-brightness pair -- the walk's own measurements, never a
            # dedicated probe. Clean-only: a pinned percentile fakes the span.
            if (abs(b_app - p_b) < 1e-9
                    and not cb_clip_actionable(p_bal, cfg)
                    and not cb_clip_actionable(bal, cfg)
                    and p_bal.span_fraction > cfg.span_floor
                    and bal.span_fraction > cfg.span_floor
                    and abs(math.log10(bal.span_fraction)
                            - math.log10(p_bal.span_fraction))
                    >= _CB_MIN_LEARN_LOGSPAN):
                k = _measured_slope(p_bal.span_fraction, bal.span_fraction,
                                    p_c, c_app, log_space=True,
                                    floor=probe.slope_floor_contrast,
                                    min_knob_delta=probe.min_knob_delta)
                if k is not None:
                    state["k_c"] = k
                    state["k_c_learned"] = True
            # A brightness move that changed nothing (deep-black terrain) grows
            # the ladder step; the contrast walk grows itself geometrically.
            if stalled and abs(d_b) > 1e-9:
                state["esc_b"] = min(state["esc_b"] * _CB_STEP_GROWTH, hi - lo)
        if note is not None:
            note(f"Auto CB {tag}: contrast={c_app:.3f}, brightness={b_app:.3f}, "
                 f"median={bal.median_fraction:.3f}, span={bal.span_fraction:.3f}, "
                 f"wclip={bal.white_clip:.3f}, bclip={bal.black_clip:.3f}, "
                 f"cost={cost:.3f}")
        return bal

    def _walk_c(direction):
        """Geometric contrast walk: seed step, x2 per consecutive same-direction
        move (any window width is reached in log time); direction change resets."""
        if direction != state["last_c_dir"]:
            state["esc_c"] = seed_c
        state["last_c_dir"] = direction
        step = state["esc_c"]
        state["esc_c"] = min(state["esc_c"] * _CB_STEP_GROWTH, hi - lo)
        return _clamp(state["c"] + direction * step, lo, hi)

    def _move_b(direction):
        """Brightness ladder move: like the contrast walk, the step grows x2 per
        consecutive same-direction move (a deep-black cure must not crawl across
        the knob at the base step); it resets on direction change and on any
        material progress (in take). An intervening brightness cure also restarts
        the contrast walk's growth run -- the gain context has changed."""
        if direction != state["last_b_dir"]:
            state["esc_b"] = cfg.brightness_step
        state["last_b_dir"] = direction
        state["last_c_dir"] = 0
        state["esc_c"] = seed_c
        step = state["esc_b"]
        state["esc_b"] = min(state["esc_b"] * _CB_STEP_GROWTH, hi - lo)
        return _clamp(state["b"] + direction * step, lo, hi)

    def _bisect_move():
        """Midpoint of a valid, wide-enough bracket, or None."""
        br = _bracket()
        if br is None or br[1] - br[0] <= _CB_BRACKET_MIN_WIDTH:
            return None
        mid = (br[0] + br[1]) / 2.0
        if abs(mid - state["c"]) < 1e-6:
            return None
        state["esc_c"] = seed_c
        state["last_c_dir"] = 0
        return mid

    def _decide():
        """Choose the next (c, b, tag) move, or None when no productive move
        exists (caller locks best-so-far)."""
        bal = state["bal"]
        span = bal.span_fraction
        w_act = bal.white_clip > cfg.max_white_clip + _CB_CLIP_ACT_MARGIN
        b_act = bal.black_clip > cfg.max_black_clip + _CB_CLIP_ACT_MARGIN
        hard_w = bal.white_clip >= _CB_HARD_CLIP_FRACTION
        hard_b = bal.black_clip >= _CB_HARD_CLIP_FRACTION

        if w_act or b_act:
            # A fully pinned rail (hard clip, collapsed span) with a known
            # rail-free brightness nearby is an OFFSET problem -- when the
            # offset alone exceeds full scale, no gain can fix it. BISECT the
            # brightness toward the last rail-free value; this iterates (each
            # still-pinned midpoint homes in; any unpinned frame refreshes
            # b_rail_free), unlike the one-shot halving the review refuted.
            if ((hard_w or hard_b) and span <= cfg.span_floor
                    and state["b_rail_free"] is not None
                    and abs(state["b_rail_free"] - state["b"])
                    > probe.min_knob_delta):
                new_b = (state["b"] + state["b_rail_free"]) / 2.0
                state["last_b_dir"] = 1 if new_b > state["b"] else -1
                state["esc_b"] = cfg.brightness_step
                return state["c"], _clamp(new_b, lo, hi), "declip"
            mid = _bisect_move()
            if mid is not None:
                return mid, state["b"], "bisect"
            br = _bracket()
            if br is not None:
                # Bracket collapsed (or its midpoint is already measured) while
                # still clipped: no usable window exists at this brightness ->
                # cure by offset, relieving whichever rail is worse off, and
                # restart window discovery at the new offset.
                state["c_lo"] = state["c_hi"] = None
                state["c_lo_b"] = state["c_hi_b"] = None
                white_excess = bal.white_clip - cfg.max_white_clip
                black_excess = bal.black_clip - cfg.max_black_clip
                direction = -1 if white_excess > black_excess else 1
                return state["c"], _move_b(direction), "declip"
            if w_act and b_act:
                # BOTH rails actionable (hard or soft): brightness only trades
                # one rail for the other -- the occupied span does not fit at
                # this gain, so compress. (The 2026-08-14 ep1 replay showed the
                # both-HARD-only version of this rule laddering brightness for
                # 5 wasted measurements while the rails traded.)
                if state["c"] > lo + 1e-6:
                    return _walk_c(-1), state["b"], "declip"
                return None
            if hard_w:
                offset_white = cfg.span_floor < span <= cfg.target_contrast_span
                if offset_white and state["b"] > lo + 1e-6:
                    return state["c"], _move_b(-1), "declip"
                if state["c"] > lo + 1e-6:
                    return _walk_c(-1), state["b"], "declip"
                if state["b"] > lo + 1e-6:
                    return state["c"], _move_b(-1), "declip"
                return None
            if hard_b:
                if state["b"] < hi - 1e-6:
                    return state["c"], _move_b(1), "declip"
                if state["c"] < hi - 1e-6:
                    # Brightness railed high yet still hard black: lift the gain.
                    return _walk_c(1), state["b"], "declip"
                return None
            # Soft actionable clip: white with a wide span is gain-driven, all
            # else is a centring problem.
            if w_act:
                if span > cfg.target_contrast_span and state["c"] > lo + 1e-6:
                    return _walk_c(-1), state["b"], "declip"
                if state["b"] > lo + 1e-6:
                    return state["c"], _move_b(-1), "declip"
                return None
            if state["b"] < hi - 1e-6:
                return state["c"], _move_b(1), "declip"
            return None

        # Clean (below ACT). Degenerate span first: recover gain or fail open.
        if span <= cfg.span_floor:
            if (state["span_tries"] < _CB_SPAN_RECOVERY_TRIES
                    and state["c"] < hi - 1e-6):
                state["span_tries"] += 1
                return _walk_c(1), state["b"], "span_recover"
            return None

        # Span out of band: bisect when bracketed, secant-jump when a slope has
        # been learned (gentle plants need long travel), else walk toward the band.
        if span < span_lo_band or span > span_hi_band:
            mid = _bisect_move()
            if mid is not None:
                return mid, state["b"], "bisect"
            if state["k_c"] is not None:
                new_c = secant_step(span, cfg.target_contrast_span, state["c"],
                                    state["k_c"], _CB_CONTRAST_JUMP_CAP, lo, hi,
                                    log_space=True)
                br = _bracket()
                if br is not None:
                    new_c = _clamp(new_c, br[0] + 1e-3, br[1] - 1e-3)
                if abs(new_c - state["c"]) > 1e-6:
                    state["esc_c"] = seed_c
                    state["last_c_dir"] = 0
                    return new_c, state["b"], "jump_c"
            direction = 1 if span < span_lo_band else -1
            at_rail = (state["c"] >= hi - 1e-6) if direction > 0 else (state["c"] <= lo + 1e-6)
            if not at_rail:
                return _walk_c(direction), state["b"], "walk_c"
            # Contrast railed; fall through to the median stage.

        # Median out of band: measured-slope secant, first move doubles as probe.
        m_err = cfg.target_median_fraction - bal.median_fraction
        if abs(m_err) > cfg.median_tol:
            if state["k_b"] is not None:
                new_b = secant_step(bal.median_fraction, cfg.target_median_fraction,
                                    state["b"], state["k_b"],
                                    _CB_BRIGHTNESS_JUMP_CAP, lo, hi,
                                    log_space=False)
                if abs(new_b - state["b"]) > 1e-6:
                    state["last_b_dir"] = 1 if new_b > state["b"] else -1
                    return state["c"], new_b, "jump_b"
                # A degenerate secant (zero/railed step, e.g. a corrupted seed)
                # must not dead-end the brightness leg: fall through to probe.
            db = _signed_probe_delta(
                state["b"],
                probe.brightness_delta if m_err > 0 else -probe.brightness_delta,
                lo, hi)
            if abs(db) < 1e-9:
                return None
            state["last_b_dir"] = 1 if db > 0 else -1
            return state["c"], _clamp(state["b"] + db, lo, hi), "probe_b"

        # In band except a deadband clip reading (between ACCEPT and ACT): one
        # gentle offset trim toward relief; hysteresis keeps this from cycling.
        if not cb_clip_acceptable(bal, cfg):
            white_side = (bal.white_clip
                          > cfg.max_white_clip + _CB_CLIP_ACCEPT_MARGIN)
            direction = -1 if white_side else 1
            trim = min(probe.brightness_delta, cfg.brightness_step)
            new_b = _clamp(state["b"] + direction * trim, lo, hi)
            if abs(new_b - state["b"]) > 1e-6:
                state["last_b_dir"] = direction
                return state["c"], new_b, "trim_clip"
        return None

    def _best_clean():
        if state["best_clean_c"] is None:
            return None
        return (state["best_clean_c"], state["best_clean_b"])

    def best_result(status):
        c, b, cost = state["best_c"], state["best_b"], state["best_cost"]
        if state["best_actionable"] and state["best_clean_c"] is not None:
            # The lowest-cost point violates the caller's own clip limits while
            # a clip-clean point WAS measured: the targets are unreachable
            # clip-clean on this scene (e.g. the target median needs more black
            # clip than max_black_clip allows). Never lock a point the commit
            # check can only revert -- lock the best clean one and say why.
            c, b, cost = (state["best_clean_c"], state["best_clean_b"],
                          state["best_clean_cost"])
            # Mark EVERY substituted status (patience / budget-exhausted /
            # measure-failed), so run_metadata can identify infeasible-config
            # sessions on any termination path, not just patience.
            status = f"{status}-infeasible"
            if note is not None:
                note(
                    f"Auto CB infeasible: the lowest-cost point is actionably "
                    f"clipped (wclip={state['best_wclip']:.3f}, "
                    f"bclip={state['best_bclip']:.3f}) - the targets conflict "
                    f"with the clip limits on this scene; locking the best "
                    f"clip-clean point instead"
                )
        return CBResult(c, b, False, cost, state["n"], status, _plant(),
                        _best_clean())

    def accepted_result(status):
        return CBResult(state["c"], state["b"], True,
                        cb_cost(state["bal"], cfg), state["n"], status, _plant(),
                        _best_clean())

    def _try_accept():
        """The current balance is in band. Return a final CBResult, or None to
        keep searching (verify failed / retries exhausted)."""
        if verify_at is None:
            return accepted_result("converged")
        if state["n"] >= budget:
            # No measurements left to verify with; lock the accepted point (it
            # WAS measured) rather than reverting to a stale best.
            return accepted_result("accepted-unverified")
        if state["verify_fails"] >= _CB_MAX_VERIFY_RETRIES:
            return None
        if take(state["c"], state["b"], "verify", measure=verify_at) is None:
            return best_result("measure-failed")
        pooled = state["bal"]
        # The pooled measurement has ~sqrt(frames) lower noise, so hold it to a
        # TIGHTER clip bound (half the accept margin) than the search reading:
        # this is what keeps borderline-clipped points from slipping through
        # acceptance on a lucky noise draw (review: ~1/3 of ep3 noise seeds
        # locked a cleanly-out-of-band point under the full margin).
        verify_clip_ok = (
            pooled.white_clip <= cfg.max_white_clip + _CB_CLIP_ACCEPT_MARGIN / 2.0
            and pooled.black_clip <= cfg.max_black_clip + _CB_CLIP_ACCEPT_MARGIN / 2.0)
        if cb_converged(pooled, cfg) and verify_clip_ok:
            return accepted_result("converged")
        state["verify_fails"] += 1
        return None

    if take(contrast, brightness, "baseline") is None:
        return best_result("no-measure")

    while True:
        bal = state["bal"]
        if bal is not None and cb_converged(bal, cfg):
            result = _try_accept()
            if result is not None:
                return result
        if state["since_improve"] >= _CB_PATIENCE:
            return best_result("patience")
        if state["n"] >= budget:
            break
        decision = _decide()
        if decision is None:
            break
        new_c, new_b, tag = decision
        if abs(new_c - state["c"]) < 1e-6 and abs(new_b - state["b"]) < 1e-6:
            break
        if take(new_c, new_b, tag) is None:
            return best_result("measure-failed")
    return best_result("budget-exhausted")