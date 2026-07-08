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


def cb_cost(balance, cfg):
    """
    Scalar "badness" of a measured balance, for best-so-far selection (lower = better).

    A static one-shot calibration must lock the lowest-cost CB it actually measured,
    not whatever the final iteration wrote. Terms:
      - median error vs target,
      - span error vs target (only when trustworthy -- no rail pinned),
      - clipping BEYOND the user's limit, weighted 2x (within-limit clipping is free,
        since the clip limits are the operator's "how much clipping I want" control).

    :param balance: CBBalance from compute_cb_balance
    :param cfg: CBControlConfig
    :return: non-negative float; 0.0 means on every target with no excess clipping
    """
    m_err = abs(cfg.target_median_fraction - balance.median_fraction)
    clip_excess = (max(0.0, balance.white_clip - cfg.max_white_clip)
                   + max(0.0, balance.black_clip - cfg.max_black_clip))
    span_trust = (balance.white_clip <= cfg.max_white_clip
                  and balance.black_clip <= cfg.max_black_clip)
    s_err = abs(cfg.target_contrast_span - balance.span_fraction) if span_trust else 0.0
    return m_err + s_err + 2.0 * clip_excess


# ===========================
# Probe -> jump -> secant calibration (measured-slope control)
# ===========================
#
# The damped-proportional law above crawls because it uses a fixed, UNMEASURED gain,
# and it mis-scales contrast because the normalized contrast knob is logarithmic --
# the SDK defines it as "contrast voltage converted to decibels then normalized", and
# the measured pixel span scales with the *linear* gain, so span-vs-knob is exponential.
# Brightness is a ~linear volt offset. The functions below measure each knob's real
# local slope from a one-shot probe, then take a deadbeat (secant) step -- in log space
# for contrast/span -- so the loop lands in a few measurements instead of oscillating.
#
#   contrast_delta / brightness_delta : size of the throwaway probe perturbation.
#   slope_floor_*  : minimum trusted |slope|; guards an 8-bit-quantized difference from
#                    producing a near-zero slope and an exploded deadbeat step.
#   min_knob_delta : the applied perturbation must be at least this large for a measured
#                    slope to be trusted (a probe near a bound can get squeezed smaller).
CBProbeConfig = namedtuple(
    "CBProbeConfig",
    ["contrast_delta", "brightness_delta", "slope_floor_contrast",
     "slope_floor_brightness", "min_knob_delta"],
)

# Outcome of one calibration run. `contrast`/`brightness` is the CB to lock (the
# lowest-cost point measured, or the converged point); `status` is a short tag.
CBResult = namedtuple(
    "CBResult",
    ["contrast", "brightness", "converged", "cost", "n_measurements", "status"],
)


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))


def cb_converged(balance, cfg):
    """
    True when a balance is within both clip limits AND the median band AND (when no
    rail is clipped) the span band. Factored out of next_cb_step so the probe/secant
    orchestrator uses the exact same acceptance test.
    """
    if balance.white_clip > cfg.max_white_clip or balance.black_clip > cfg.max_black_clip:
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


def run_cb_calibration(measure_at, contrast, brightness, cfg, probe, budget, note=None):
    """
    One-shot detector contrast/brightness calibration: clip-relief -> probe + jump ->
    secant fine-tune, locking the lowest-cost CB actually measured (best-so-far).

    Pure orchestration -- all hardware I/O is behind ``measure_at`` so this runs in
    tests against a synthetic detector. ``measure_at(c, b)`` must apply the CB, settle,
    and return ``(CBBalance | None, contrast_applied, brightness_applied)``; the applied
    values may differ from the request if the detector clamps them, and the orchestrator
    tracks the applied values as the true knob position.

    Decoupling note: span depends on gain only (offset cancels in a difference) so the
    contrast slope (measured in log span) is offset-independent; the median is gain-
    coupled, so a contrast move shifts it. We therefore make the LAST move a brightness
    re-center, and order the two knobs by span width so a probe never deepens clipping:
    wide span -> reduce contrast first; narrow span -> center brightness first.

    :param measure_at: callable(c, b) -> (CBBalance | None, c_applied, b_applied)
    :param contrast: starting normalized contrast in [0, 1]
    :param brightness: starting normalized brightness in [0, 1]
    :param cfg: CBControlConfig (targets / clip limits / bounds + clip-phase gains)
    :param probe: CBProbeConfig (probe deltas + slope floors)
    :param budget: max number of measurements (== cb_max_iterations)
    :param note: optional callable(str) for progress logging
    :return: CBResult
    """
    lo, hi = cfg.min_bound, cfg.max_bound
    budget = max(1, int(budget))
    jump_cap = hi - lo  # a jump is deadbeat: only the bounds limit it
    state = {"c": contrast, "b": brightness, "n": 0, "bal": None,
             "best_cost": float("inf"), "best_c": contrast, "best_b": brightness,
             "k_contrast": None, "k_brightness": None}

    def take(c_req, b_req, tag):
        bal, c_app, b_app = measure_at(c_req, b_req)
        state["n"] += 1
        state["c"], state["b"], state["bal"] = c_app, b_app, bal
        if bal is not None:
            cost = cb_cost(bal, cfg)
            if cost < state["best_cost"]:
                state["best_cost"] = cost
                state["best_c"], state["best_b"] = c_app, b_app
            if note is not None:
                note(f"Auto CB {tag}: contrast={c_app:.3f}, brightness={b_app:.3f}, "
                     f"median={bal.median_fraction:.3f}, span={bal.span_fraction:.3f}, "
                     f"wclip={bal.white_clip:.3f}, bclip={bal.black_clip:.3f}, "
                     f"cost={cost:.3f}")
        return bal

    def _clipping(bal):
        return bal.white_clip > cfg.max_white_clip or bal.black_clip > cfg.max_black_clip

    def declip_step(bal):
        # A too-wide occupied span clips a rail no matter where it is centred, so the
        # cure is less gain (lower contrast) -- chasing it with brightness alone just
        # trades the white rail for the black rail (a limit cycle). When the span is
        # acceptable the clip is a centring problem: relieve with the gated brightness law.
        if bal.span_fraction > cfg.target_contrast_span:
            return _clamp(state["c"] - cfg.contrast_step, lo, hi), state["b"]
        new_c, new_b, _conv = next_cb_step(bal, state["c"], state["b"], cfg)
        return new_c, new_b

    def best_result(status):
        return CBResult(state["best_c"], state["best_b"], False,
                        state["best_cost"], state["n"], status)

    def converged_result():
        return CBResult(state["c"], state["b"], True,
                        cb_cost(state["bal"], cfg), state["n"], "converged")

    # --- Probe / jump primitives (each returns False only on a measurement failure).
    def probe_brightness():
        if state["n"] >= budget:
            return True
        c0, b0 = state["c"], state["b"]
        median0 = state["bal"].median_fraction
        # Probe TOWARD the median target so the perturbation relieves, not deepens, any
        # rail pressure.
        mag = (probe.brightness_delta if cfg.target_median_fraction >= median0
               else -probe.brightness_delta)
        db = _signed_probe_delta(b0, mag, lo, hi)
        bal_b = take(c0, b0 + db, "probe_b")
        if bal_b is None:
            return False
        state["k_brightness"] = _measured_slope(
            median0, bal_b.median_fraction, b0, state["b"],
            log_space=False, floor=probe.slope_floor_brightness,
            min_knob_delta=probe.min_knob_delta)
        return True

    def jump_brightness():
        k = state["k_brightness"]
        if k is None or state["n"] >= budget or _clipping(state["bal"]):
            return True
        new_b = secant_step(state["bal"].median_fraction, cfg.target_median_fraction,
                            state["b"], k, jump_cap, lo, hi, log_space=False)
        if abs(new_b - state["b"]) > 1e-6:
            if take(state["c"], new_b, "jump_b") is None:
                return False
        return True

    def probe_contrast():
        if (state["n"] >= budget or _clipping(state["bal"])
                or state["bal"].span_fraction <= cfg.span_floor):
            return True
        c0, b0 = state["c"], state["b"]
        span0 = state["bal"].span_fraction
        # Probe TOWARD the span target (contrast raises span), again to avoid deepening a rail.
        mag = (probe.contrast_delta if cfg.target_contrast_span >= span0
               else -probe.contrast_delta)
        dc = _signed_probe_delta(c0, mag, lo, hi)
        bal_c = take(c0 + dc, b0, "probe_c")
        if bal_c is None:
            return False
        if bal_c.span_fraction > 0.0:
            state["k_contrast"] = _measured_slope(
                span0, bal_c.span_fraction, c0, state["c"],
                log_space=True, floor=probe.slope_floor_contrast,
                min_knob_delta=probe.min_knob_delta)
        return True

    def jump_contrast():
        k = state["k_contrast"]
        if (k is None or state["n"] >= budget or _clipping(state["bal"])
                or state["bal"].span_fraction <= cfg.span_floor):
            return True
        new_c = secant_step(state["bal"].span_fraction, cfg.target_contrast_span,
                            state["c"], k, jump_cap, lo, hi, log_space=True)
        if abs(new_c - state["c"]) > 1e-6:
            if take(new_c, state["b"], "jump_c") is None:
                return False
        return True

    if take(contrast, brightness, "baseline") is None:
        return best_result("no-measure")

    # --- Phase A: relieve clipping (span is a pinned-rail lie, so don't fill yet).
    while state["n"] < budget and _clipping(state["bal"]):
        new_c, new_b = declip_step(state["bal"])
        if abs(new_c - state["c"]) < 1e-6 and abs(new_b - state["b"]) < 1e-6:
            break  # clamped at a bound -> can't relieve further
        if take(new_c, new_b, "declip") is None:
            return best_result("measure-failed")
    if cb_converged(state["bal"], cfg):
        return converged_result()

    # --- Phase B: probe + deadbeat jump, ordered by span width, brightness re-center last.
    if not _clipping(state["bal"]) and state["n"] < budget:
        if state["bal"].span_fraction > cfg.target_contrast_span:
            steps = (probe_contrast, jump_contrast, probe_brightness, jump_brightness)
        else:
            steps = (probe_brightness, jump_brightness, probe_contrast, jump_contrast,
                     jump_brightness)
        for step in steps:
            if not step():
                return best_result("measure-failed")
            if cb_converged(state["bal"], cfg):
                return converged_result()
            if state["n"] >= budget:
                break

    # --- Phase C: fine-tune with the measured slopes (deadbeat, capped, one knob/meas).
    while state["n"] < budget:
        if cb_converged(state["bal"], cfg):
            return converged_result()
        bal = state["bal"]

        if _clipping(bal):
            new_c, new_b = declip_step(bal)
            if abs(new_c - state["c"]) < 1e-6 and abs(new_b - state["b"]) < 1e-6:
                break
            if take(new_c, new_b, "refine_declip") is None:
                return best_result("measure-failed")
            continue

        span_off = abs(cfg.target_contrast_span - bal.span_fraction) > cfg.contrast_tol
        median_off = abs(cfg.target_median_fraction - bal.median_fraction) > cfg.median_tol

        # Contrast first (span is offset-independent), then brightness.
        if span_off and state["k_contrast"] is not None and bal.span_fraction > cfg.span_floor:
            new_c = secant_step(bal.span_fraction, cfg.target_contrast_span,
                                state["c"], state["k_contrast"], cfg.contrast_step, lo, hi,
                                log_space=True)
            if abs(new_c - state["c"]) > 1e-6:
                if take(new_c, state["b"], "refine_c") is None:
                    return best_result("measure-failed")
                continue
        if median_off and state["k_brightness"] is not None:
            new_b = secant_step(bal.median_fraction, cfg.target_median_fraction,
                                state["b"], state["k_brightness"], cfg.brightness_step, lo, hi,
                                log_space=False)
            if abs(new_b - state["b"]) > 1e-6:
                if take(state["c"], new_b, "refine_b") is None:
                    return best_result("measure-failed")
                continue

        # No usable measured slope for the off knob -> fall back to the gated law.
        new_c, new_b, _conv = next_cb_step(bal, state["c"], state["b"], cfg)
        if abs(new_c - state["c"]) < 1e-6 and abs(new_b - state["b"]) < 1e-6:
            break
        if take(new_c, new_b, "refine_fallback") is None:
            return best_result("measure-failed")

    if cb_converged(state["bal"], cfg):
        return converged_result()
    return best_result("budget-exhausted")