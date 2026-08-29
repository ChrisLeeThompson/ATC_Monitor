"""
Real-Time Monitor (RTM) processing.
This module provides functions to convert RTM data into displayable images,
plus the hardware-free balance metric and control law behind the Auto
contrast/brightness (CB) detector correction.
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
# p2_fraction/p98_fraction: the robust histogram edges as fractions of full
# scale -- the quantities the edge-margin correction (run_cb_correction)
# steers. Appended with defaults so the many positional seven-field
# constructions in tests and simulators stay valid; the solver falls back to
# median +- span/2 when a fixture leaves them None.
CBBalance = namedtuple(
    "CBBalance",
    ["white_clip", "black_clip", "median_fraction", "n_valid", "vmin", "vmax",
     "span_fraction", "p2_fraction", "p98_fraction"],
    defaults=(None, None),
)
_CB_MIN_INTERIOR_FRACTION = 0.02   # interior must be >= 2% of valid pixels ...
_CB_MIN_INTERIOR_COUNT = 50        # ... and >= this many, else use all valid pixels.
                                   # Coupled to _CB_SPAN_BLIND_INTERIOR below: this
                                   # switch changes which population the median/span
                                   # describe, so moving it moves the regime the
                                   # empirical blindness threshold was calibrated in
                                   # (the 2026-08-23 frames sat at 2.4% interior,
                                   # just above this fallback)

# Declip / cost / acceptance tunables (internal; promotable to CBEdgeConfig if
# operators ever need them).
_CB_HARD_CLIP_FRACTION = 0.30    # clip fraction at/above which a rail dominates the
                                 # histogram: the frame is deterministic (bit-identical
                                 # statistics there are real, not a stale buffer) and
                                 # the cure is prescribed by which rail it is. The span
                                 # is still readable at this level -- only
                                 # _CB_SPAN_BLIND_INTERIOR below marks median/span
                                 # unreadable, and it is a stricter test
_CB_SPAN_BLIND_INTERIOR = 0.10   # fraction of valid pixels that must remain off both
                                 # rails for median/span to carry information. Below
                                 # it the reading is whatever narrow population
                                 # survives between the rails, and it stops tracking
                                 # the knobs at all. Calibrated on the two observed
                                 # regimes: the 2026-08-23 frozen frames held a 2.4%
                                 # interior and returned a bit-identical median/span
                                 # across the whole contrast range and both rails,
                                 # while the 2026-08-14 ep1 corner (20.1% interior,
                                 # span 0.992) still tracked and must keep its
                                 # offset-driven exemption. Note this is a stricter
                                 # test than _CB_HARD_CLIP_FRACTION: a merely
                                 # rail-dominated frame (30% pinned, 70% interior)
                                 # has a perfectly readable span. Coupled to
                                 # _CB_MIN_INTERIOR_FRACTION above: below that
                                 # switch the median/span are measured over all
                                 # valid pixels, not the interior, so this
                                 # threshold judges two physically different
                                 # readings on either side of it
# Clip hysteresis: the field 2026-08-14 limit cycle was declip acting at exactly the
# acceptance threshold while measurement noise straddled it (bclip 0.015-0.05 bounced
# a declip/recenter tug-of-war for 10+ measurements). The accept margin tolerates a
# small excess over the operator's clip limit (also ~where percentile pinning would begin
# to bias the span reading); the act threshold fires only well above it. Readings
# inside the deadband neither block acceptance nor trigger clip moves.
_CB_CLIP_ACCEPT_MARGIN = 0.01
_CB_CLIP_ACT_MARGIN = 0.04

# Termination / verification.
_CB_MAX_MEASURE_RECOVERIES = 3   # RTM recoveries (restart + re-measure) a single run may
                                 # spend on failed measurements before it stops trying;
                                 # failures beyond this still consume measurement budget
                                 # (no early abort: a permanently failed RTM grinds to
                                 # the budget and ends measure-failed)
# cb_edge_cost clip weights: white clipping saturates exactly the bright texture
# the downstream top-hat energy measures; black pixels are background the top-hat
# removes anyway. Weight white-clip excess above black accordingly.
_CB_COST_WHITE_CLIP_WEIGHT = 2.5
_CB_COST_BLACK_CLIP_WEIGHT = 1.5
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
    # Clip fractions: over all valid pixels (the operator's clip budget).
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
                     span_fraction, float(p_lo / white_level),
                     float(p_hi / white_level))
def cb_clip_acceptable(balance, cfg):
    """
    True when both clip fractions are within the operator limit plus the accept margin.

    This is the acceptance-side clip test (used by cb_edges_accepted). The
    margin exists because a 1-frame clip reading has noise of the same order
    as the limit itself; the pooled verify measurement re-checks any accepted
    point at lower noise.
    """
    return (balance.white_clip <= cfg.max_white_clip + _CB_CLIP_ACCEPT_MARGIN
            and balance.black_clip <= cfg.max_black_clip + _CB_CLIP_ACCEPT_MARGIN)


def cb_clips_actionable(white_clip, black_clip, cfg):
    """
    The clip-actionable test on raw fractions, for callers holding clip numbers
    without a CBBalance (a persisted telemetry row, a commit-check summary).

    :param white_clip: fraction of valid pixels pinned at the ceiling
    :param black_clip: fraction of valid pixels pinned at the floor
    :param cfg: CBEdgeConfig
    :return: bool
    """
    return (white_clip > cfg.max_white_clip + _CB_CLIP_ACT_MARGIN
            or black_clip > cfg.max_black_clip + _CB_CLIP_ACT_MARGIN)


def cb_clip_actionable(balance, cfg):
    """
    True when either clip fraction is far enough over the operator limit that a clip-
    relief move should fire. Readings between the accept and act margins are a
    deadband: not accepted as converged-clean, but not worth chasing either (the
    2026-08-14 field limit cycle lived entirely inside that band).
    """
    return cb_clips_actionable(balance.white_clip, balance.black_clip, cfg)


def cb_rail_dominated_clips(white_clip, black_clip):
    """
    The rail-dominance test on raw fractions (see cb_rail_dominated), for
    callers holding clip numbers without a CBBalance.

    :param white_clip: fraction of valid pixels pinned at the ceiling
    :param black_clip: fraction of valid pixels pinned at the floor
    :return: bool
    """
    return max(white_clip, black_clip) >= _CB_HARD_CLIP_FRACTION


def cb_interior_fraction(balance):
    """
    Fraction of valid pixels sitting off both rails -- the population the median
    and span are actually measured over (compute_cb_balance's ``interior``).

    :param balance: CBBalance
    :return: float in [0, 1]
    """
    return max(0.0, 1.0 - balance.white_clip - balance.black_clip)


def cb_span_blind(balance):
    """
    True when too little of the frame is off the rails for the median/span
    readings to carry information about the detector state.

    A clipped frame's median and span are computed over whatever survives between
    the rails. Squeeze that interior small enough and the readings stop being a
    measurement of the scene at all: on 2026-08-23 a 97.6%-white frame (2.4%
    interior) reported a bit-identical median 0.8824 / span 0.2395 at contrast
    0.657, 0.621, 0.550 and 0.122 -- and reported the same pair when the frame
    flipped to 97.5% black. Only the clip fractions moved.

    Callers that arbitrate on span must consult this first, or they will read a
    frozen constant as evidence. Note this is a stricter test than
    cb_rail_dominated: a frame with one rail at 30% still has a 70% interior and
    a perfectly usable span.

    :param balance: CBBalance, or None
    :return: bool (False for None -- nothing measured, nothing to judge)
    """
    if balance is None:
        return False
    return cb_interior_fraction(balance) < _CB_SPAN_BLIND_INTERIOR


def cb_rail_dominated(balance):
    """
    True when one rail dominates the frame -- at least _CB_HARD_CLIP_FRACTION of
    the valid pixels are pinned at the ceiling or the floor.

    Such a frame is deterministic: the pinned pixels sit at exactly the rail, so
    consecutive frames -- and small knob changes that leave the frame pinned --
    reproduce the same pixel statistics bit for bit. Any caller that infers "the
    RTM buffer did not refresh" from identical statistics must exempt these
    frames. (2026-08-23: a 99.1%-white-railed scene was rejected as a frozen
    buffer on the very first declip step, aborting the whole calibration and
    locking the saturated detector.)

    The search relies on the same determinism from the other side: it walks
    geometrically precisely so bit-identical railed frames cannot stall it.

    :param balance: CBBalance, or None
    :return: bool (False for None -- nothing measured, nothing to exempt)
    """
    if balance is None:
        return False
    return cb_rail_dominated_clips(balance.white_clip, balance.black_clip)
# Escape tuning. The escape is a retreat relative to the pinning rail, with
# steps that grow geometrically, rather than a jump to a fixed "safe" anchor
# point: a safe-anchor design failed the 2026-08-14 fitted plants, whose
# usable windows sit elsewhere.
_CB_ESCAPE_SEED = 0.05       # first escape step (both knobs)
_CB_ESCAPE_CAP = 0.40        # escalation ceiling
# Default knob slopes when no per-tool value is persisted. The signs are
# physics, not tuning: more gain widens the spread, more offset brightens
# the midpoint, on every detector. The magnitudes sit deliberately at the
# harsh end of everything measured: a too-high default only undershoots --
# small safe steps that _learn refines within a move or two -- while a
# too-low one overshoots straight into a rail (the ep8 replay saturated
# from a near-target start on modest defaults). k_c: the fitted harsh
# (Hydra) plant's true slope is gspan/20 = 12 decades/knob, and the
# on-tool learned values were 12.31 and 11.89. The "Hydra k_c ~2-6.5"
# figures the old 6.0 default was read from were lower bounds from
# clip-suppressed probes, not slope estimates (the field probes measured
# >= 6.6-8.5 decades/knob with the response flattened by clipping), so
# 6.0 sat below even the lower bound -- violating this comment's own
# harsh-end rule: every solve from a narrow-span baseline overshot the
# ~0.04-knob usable window into a rail (ep2 / PlantSweep gspan >= 240,
# 2026-08-26). A sign error -- the failure class behind field Run-4's
# stale-slope jumps to zero gain -- is impossible here by construction.
_CB_DEFAULT_K_C = 12.0       # d log10(spread) / d contrast
_CB_DEFAULT_K_B = 8.0        # d midpoint / d brightness
_CB_LEARN_TRUST = 0.03       # min |response| to refine a slope from a move:
                             # below it the sign is a coin flip of 1-frame
                             # noise (review lesson, 2026-08-15), so such a
                             # response neither teaches a slope nor
                             # contradicts one
_CB_STEP_CAP_C = 0.30        # anti-windup caps on one solve iteration;
_CB_STEP_CAP_B = 0.25        # halved per knob when a move lands on a rail
_CB_STEP_CAP_MIN = 0.01      # (the 2026-08-14 plants' usable window is
                             # narrower than any safe fixed step -- the
                             # campaign's core lesson), floored here
_CB_EDGE_TOL_DEFAULT = 0.12  # per-edge acceptance half-width; sits well
                             # above 1-frame edge noise (the 2026-08-14
                             # campaign died on gates at ~1 sigma of noise)
_CB_SPREAD_FLOOR = 0.02      # below this the image is flat; heavily penalized
_CB_FLAT_PENALTY = 1.0       # a flat image must never win "best" (field
                             # Run-2 locked contrast 0.0 on a cost tie-break)

# Public alias for run_metadata (the worker records the acceptance
# tolerance so traces stay interpretable).
CB_EDGE_TOL_DEFAULT = _CB_EDGE_TOL_DEFAULT

CBEdgeConfig = namedtuple(
    "CBEdgeConfig",
    ["white_level", "margin_lo", "margin_hi", "max_white_clip",
     "max_black_clip", "min_bound", "max_bound", "edge_tol"],
    defaults=[_CB_EDGE_TOL_DEFAULT],
)

# Outcome of run_cb_correction. best_clean is the lowest-cost clip-clean
# measured point as (contrast, brightness) or None -- the worker's
# commit-check revert target when the committed point turns out clipped.
CBResult = namedtuple(
    "CBResult",
    ["contrast", "brightness", "converged", "cost", "n_measurements",
     "status", "plant", "best_clean"],
)


def _cb_edges(bal):
    """The robust histogram edges of a balance, as fractions of full scale.

    Falls back to median +- span/2 for fixtures that construct CBBalance
    without the appended edge fields, clamped into [0, 1] -- the fallback
    can otherwise place an edge outside the physical range and fabricate
    edge error."""
    if bal.p2_fraction is not None and bal.p98_fraction is not None:
        return bal.p2_fraction, bal.p98_fraction
    half = bal.span_fraction / 2.0
    return (max(0.0, bal.median_fraction - half),
            min(1.0, bal.median_fraction + half))


def cb_edge_cost(balance, cfg):
    """Distance from the edge targets, plus clip excess (white weighted above
    black -- saturation destroys the texture the analysis measures) and a
    flat-image penalty so a zero-spread point can never outrank a moderately
    clipped one."""
    lo, hi = _cb_edges(balance)
    cost = abs(lo - cfg.margin_lo) + abs(hi - (1.0 - cfg.margin_hi))
    cost += _CB_COST_WHITE_CLIP_WEIGHT * max(
        0.0, balance.white_clip - cfg.max_white_clip)
    cost += _CB_COST_BLACK_CLIP_WEIGHT * max(
        0.0, balance.black_clip - cfg.max_black_clip)
    if (hi - lo) < _CB_SPREAD_FLOOR:
        cost += _CB_FLAT_PENALTY
    return cost


def cb_edges_accepted(balance, cfg, slack=1.0):
    """True when both edges sit within edge_tol of their targets and the clip
    fractions are inside the operator limits plus the accept margin.

    ``slack`` scales the edge tolerance; the verify re-check uses a slightly
    looser band so measurement noise on a borderline point cannot flicker
    accept/verify pairs for the rest of the budget."""
    lo, hi = _cb_edges(balance)
    tol = cfg.edge_tol * slack
    return (abs(lo - cfg.margin_lo) <= tol
            and abs(hi - (1.0 - cfg.margin_hi)) <= tol
            and balance.white_clip
            <= cfg.max_white_clip + _CB_CLIP_ACCEPT_MARGIN
            and balance.black_clip
            <= cfg.max_black_clip + _CB_CLIP_ACCEPT_MARGIN)


def run_cb_correction(measure_at, contrast, brightness, cfg, budget,
                      note=None, verify_at=None, recover=None,
                      init_k_contrast=None, init_k_brightness=None):
    """
    Edge-margin detector correction: escape saturation, then solve.

    :param measure_at: callable (contrast, brightness) -> (CBBalance | None,
        applied_contrast, applied_brightness); one measurement per call
    :param contrast, brightness: current knob positions (normalized)
    :param cfg: CBEdgeConfig
    :param budget: max measurements, verify included
    :param note: optional callable(str) for operator-readable progress
    :param verify_at: pooled-measure callable with measure_at's signature,
        used to confirm acceptance (defaults to measure_at)
    :param recover: optional callable() that restarts the RTM after a failed
        measurement (one retry per failure, bounded per run)
    :param init_k_contrast: seed for d log10(spread) / d contrast (per-tool,
        persisted); probed when absent
    :param init_k_brightness: seed for d midpoint / d brightness
    :return: CBResult -- converged=True only after a verified acceptance;
        otherwise the lowest-cost measured point with an honest status
    """
    lo_b, hi_b = cfg.min_bound, cfg.max_bound
    verify = verify_at if verify_at is not None else measure_at
    state = {
        "c": contrast, "b": brightness, "bal": None, "n": 0,
        "k_c": (init_k_contrast if init_k_contrast is not None
                and init_k_contrast > 0 else _CB_DEFAULT_K_C),
        "k_b": (init_k_brightness if init_k_brightness is not None
                and init_k_brightness > 0 else _CB_DEFAULT_K_B),
        "k_c_learned": False, "k_b_learned": False,
        "k_c_contradicted": False, "k_b_contradicted": False,
        "best_cost": float("inf"), "best_c": contrast, "best_b": brightness,
        "best_actionable": False,
        "best_clean_cost": float("inf"), "best_clean": None,
        "measure_fails": 0, "recoveries": 0, "last_good": None,
        "cap_c": _CB_STEP_CAP_C, "cap_b": _CB_STEP_CAP_B,
        "esc": _CB_ESCAPE_SEED, "esc_dir": 0.0,
        "last_dc": 0.0, "last_db": 0.0,  # the applied deltas of the last
                                         # move, for causal rail attribution
        "last_rail": 0.0,                # the rail the last relief fled
        "acc_db": 0.0, "acc_dmid": 0.0,  # sub-trust brightness residual
                                         # accumulators (see _learn)
    }

    def _plant():
        # Report only what this run measured: echoing a seed back would let
        # a wrong persisted slope survive forever.
        return {
            "k_brightness": state["k_b"] if state["k_b_learned"] else None,
            "k_contrast": state["k_c"] if state["k_c_learned"] else None,
            "k_brightness_contradicted": state["k_b_contradicted"],
            "k_contrast_contradicted": state["k_c_contradicted"],
        }

    def _result(status, converged=False):
        c, b, cost = state["best_c"], state["best_b"], state["best_cost"]
        if state["best_actionable"] and state["best_clean"] is not None:
            # Never lock a point the worker's commit check can only revert.
            c, b = state["best_clean"]
            cost = state["best_clean_cost"]
            status = f"{status}-infeasible"
            if note is not None:
                note("Auto CB infeasible: the lowest-cost point is actionably "
                     "clipped - the margins conflict with the clip limits on "
                     "this scene; locking the best clip-clean point instead")
        if (status != "no-measure" and state["measure_fails"]
                and state["measure_fails"] * 2 >= state["n"]):
            # A failed baseline keeps its distinct "no-measure" status: the
            # worker skips the whole commit path on it (nothing was measured,
            # so there is nothing to lock), and this majority-failed override
            # used to rewrite it to "measure-failed" at n=1, leaving that
            # skip unreachable.
            status = "measure-failed"
        return CBResult(c, b, converged, cost, state["n"], status, _plant(),
                        state["best_clean"])

    def take(c_req, b_req, tag, measure=None):
        measure = measure_at if measure is None else measure
        bal, c_app, b_app = measure(c_req, b_req)
        if (bal is None and recover is not None
                and state["recoveries"] < _CB_MAX_MEASURE_RECOVERIES):
            state["recoveries"] += 1
            recover()
            bal, c_app, b_app = measure(c_req, b_req)
        state["n"] += 1
        prev = (state["bal"], state["c"], state["b"])
        state["last_dc"] = c_app - state["c"]
        state["last_db"] = b_app - state["b"]
        state["c"], state["b"], state["bal"] = c_app, b_app, bal
        if bal is None:
            state["measure_fails"] += 1
            if state["last_good"] is not None:
                g_bal, g_c, g_b = state["last_good"]
                state["c"], state["b"], state["bal"] = g_c, g_b, g_bal
            return None, prev
        if not cb_span_blind(bal) and not cb_rail_dominated(bal):
            # Only a readable frame is worth rewinding to -- a blind or
            # rail-dominated one carries no trustworthy spread, and letting
            # it overwrite this slot disabled the rewind entirely (every
            # post-overshoot frame became its own "last good").
            state["last_good"] = (bal, c_app, b_app)
        cost = cb_edge_cost(bal, cfg)
        actionable = cb_clip_actionable(bal, cfg)
        if cost < state["best_cost"]:
            state["best_cost"] = cost
            state["best_c"], state["best_b"] = c_app, b_app
            state["best_actionable"] = actionable
        if not actionable and cost < state["best_clean_cost"]:
            state["best_clean_cost"] = cost
            state["best_clean"] = (c_app, b_app)
        if note is not None:
            lo, hi = _cb_edges(bal)
            note(f"Auto CB {tag}: contrast={c_app:.3f}, "
                 f"brightness={b_app:.3f}, edges={lo:.3f}..{hi:.3f}, "
                 f"wclip={bal.white_clip:.3f}, bclip={bal.black_clip:.3f}, "
                 f"cost={cost:.3f}")
        return bal, prev

    def _adopt_k_b(k):
        """Adopt or contradict a measured brightness slope. Shared by the
        direct learn and the sub-gate accumulator so accumulated evidence
        carries exactly the weight of direct evidence; any adoption or
        contradiction restarts the accumulation (stale half-runs must
        never blend into a later estimate)."""
        if k > 0:
            state["k_b"] = k
            state["k_b_learned"] = True
        else:
            state["k_b"] = _CB_DEFAULT_K_B
            state["k_b_learned"] = False
            state["k_b_contradicted"] = True
            state["cap_b"] = max(_CB_STEP_CAP_MIN,
                                 state["cap_b"] * 0.5)
        state["acc_db"] = state["acc_dmid"] = 0.0

    def _learn(bal, prev):
        """Refine the slope magnitudes from a rail-free move.

        Per-knob attribution assumes independence -- the spread responds to
        contrast, the midpoint to brightness -- which holds to first order;
        cross-term noise is filtered by the trust floor and by refusing
        negative slopes (the signs are physics: a measured response that
        opposes them is noise or a cross-term, and adopting it would aim
        every later solve backwards, the field Run-4 failure class; such a
        reading resets the slope to its default and halves the knob's step
        cap instead)."""
        p_bal, p_c, p_b = prev
        if p_bal is None or bal is None:
            return
        rail_free = (max(p_bal.white_clip, p_bal.black_clip)
                     < _CB_HARD_CLIP_FRACTION
                     and max(bal.white_clip, bal.black_clip)
                     < _CB_HARD_CLIP_FRACTION)
        if not rail_free:
            return
        dc, db = state["c"] - p_c, state["b"] - p_b
        p_lo, p_hi = _cb_edges(p_bal)
        lo, hi = _cb_edges(bal)
        p_spread, spread = max(p_hi - p_lo, 1e-6), max(hi - lo, 1e-6)
        # Joint attribution through the plant model (offset applied after
        # the gain stage): the spread responds to contrast alone, so k_c
        # comes straight from the spread ratio; and because that ratio is
        # measured, not modeled, the gain's contribution to the midpoint
        # change can be subtracted exactly, leaving the brightness slope
        # clean --
        # mid_new = mid_prev * r + k_b * db. No single-knob-move
        # requirement, so every solve iteration refines both slopes.
        if abs(dc) > 1e-6:
            d_log = math.log10(spread / p_spread)
            if abs(d_log) >= _CB_LEARN_TRUST:
                k = d_log / dc
                if k > 0:
                    state["k_c"] = k
                    state["k_c_learned"] = True
                else:
                    state["k_c"] = _CB_DEFAULT_K_C
                    state["k_c_learned"] = False
                    state["k_c_contradicted"] = True
                    state["cap_c"] = max(_CB_STEP_CAP_MIN,
                                         state["cap_c"] * 0.5)
        if abs(db) > 1e-6:
            r = spread / p_spread
            p_mid = (p_lo + p_hi) / 2.0
            d_mid_resid = (lo + hi) / 2.0 - p_mid * r
            if abs(d_mid_resid) >= _CB_LEARN_TRUST:
                _adopt_k_b(d_mid_resid / db)
            else:
                # Sub-gate residual accumulation. Each rail-free residual
                # is k_b*db + noise: same-sign residuals sum linearly
                # while the noise sum grows only as sqrt(n), so a true
                # k_b ~ 1.25 plant -- residual ~0.024 per 0.02 move,
                # forever under the per-move gate -- teaches within ~2
                # moves instead of never (the c066 brightness-creep
                # starvation). A direction flip restarts the run:
                # opposite-sign residuals are as likely noise as signal,
                # and letting them cancel would launder a zero response
                # into a slope. No symmetric k_c accumulator: a wrong k_c
                # surfaces as a rail landing, which _rail_evidence
                # converts to slope evidence.
                if state["acc_db"] * db < 0:
                    state["acc_db"] = state["acc_dmid"] = 0.0
                state["acc_db"] += db
                state["acc_dmid"] += d_mid_resid
                if (abs(state["acc_dmid"]) >= _CB_LEARN_TRUST
                        and abs(state["acc_db"]) >= 0.01):
                    _adopt_k_b(state["acc_dmid"] / state["acc_db"])

    def _rail_evidence(bal, prev):
        """A solve move that lands rail-dominated or blind is still slope
        evidence: the spread grew past what the frame can hold (which is
        why _learn's rail-free gate leaves the harsh-plant overshoot cycle
        unable to learn at all -- every solve landing is railed, so the
        wrong slope drives the next solve too). Sign-causal gate: with
        db <= 0 the offset moved away from the white rail, so a white-hard
        landing is the gain's doing (raw |dc| > |db| would mis-gate here --
        k_b's default makes db large in knob units while the spread
        response belongs to dc alone). The capacity bound can only
        underestimate k_c -- a genuine lower bound -- so adopting it still
        undershoots, the safe side, and an exact rail-free learn
        overwrites it later. The raise
        persists (k_c_learned) so a stale gentle persisted seed self-heals
        and overwrites itself the same session -- the 2026-08-26
        stale-seed 2-cycle's cure. Black landings and db > 0 landings
        never qualify; the x4 ceiling bounds a noise-corrupted p_hi; the
        max(x2, .) floor guarantees geometric progress on repeated
        re-entries."""
        p_bal, p_c, p_b = prev
        if p_bal is None or bal is None:
            return
        if cb_span_blind(p_bal) or cb_rail_dominated(p_bal):
            return
        white_hard = (bal.white_clip >= _CB_HARD_CLIP_FRACTION
                      or (cb_span_blind(bal)
                          and bal.white_clip >= bal.black_clip))
        dc, db = state["c"] - p_c, state["b"] - p_b
        if not white_hard or dc <= 1e-6 or db > 1e-6:
            return
        _p_lo, p_hi = _cb_edges(p_bal)
        k_lb = math.log10(max(1.0 - state["k_b"] * db, 1.0)
                          / max(p_hi, _CB_SPREAD_FLOOR)) / dc
        k_new = min(state["k_c"] * 4.0, max(state["k_c"] * 2.0, k_lb))
        if k_new > state["k_c"]:
            state["k_c"] = k_new
            state["k_c_learned"] = True  # persists -> stale seeds self-heal

    target_lo = cfg.margin_lo
    target_hi = 1.0 - cfg.margin_hi
    target_mid = (target_lo + target_hi) / 2.0
    target_spread = max(target_hi - target_lo, _CB_SPREAD_FLOOR)

    bal, _ = take(contrast, brightness, "baseline")
    if bal is None:
        return _result("no-measure")

    while state["n"] < budget:
        bal = state["bal"]
        if bal is None:
            # Rewound to last-good; re-measure from there.
            bal, prev = take(state["c"], state["b"], "re-measure")
            if bal is None:
                continue
        if cb_edges_accepted(bal, cfg):
            if state["n"] >= budget:
                break
            v_bal, _prev = take(state["c"], state["b"], "verify",
                                measure=verify)
            if v_bal is not None and cb_edges_accepted(v_bal, cfg,
                                                       slack=1.25):
                # Lock the verified point itself. _result picks its point by
                # cost alone, which can be an earlier, lower-cost measurement
                # that never passed verification (and its -infeasible switch
                # could then move the lock off the acceptance entirely) --
                # converged must mean "the knobs sit at the verified
                # acceptance".
                return CBResult(state["c"], state["b"], True,
                                cb_edge_cost(v_bal, cfg), state["n"],
                                "converged", _plant(), state["best_clean"])
            continue
        if cb_span_blind(bal):
            # A blind frame carries no measurement at all.
            lg = state["last_good"]
            if lg is not None:
                # A move from a readable frame landed blind: the usable
                # window is narrower than the step (the 2026-08-14
                # campaign's core lesson). Bisect between the readable
                # point and the blind landing -- re-solving from the
                # readable point just re-applies the same oversized move,
                # while the midpoint walk homes on the window edge
                # geometrically, per knob, whatever direction the overshoot
                # took.
                mid_c = _clamp((lg[1] + state["c"]) / 2.0, lo_b, hi_b)
                mid_b = _clamp((lg[2] + state["b"]) / 2.0, lo_b, hi_b)
                if (abs(mid_c - lg[1]) < 1e-3
                        and abs(mid_b - lg[2]) < 1e-3):
                    # The window edge is resolved to knob resolution; carry
                    # on solving from the readable side with tightened caps
                    # so the next solve cannot re-cross it.
                    state["cap_c"] = max(_CB_STEP_CAP_MIN,
                                         state["cap_c"] * 0.25)
                    state["cap_b"] = max(_CB_STEP_CAP_MIN,
                                         state["cap_b"] * 0.25)
                    state["bal"], state["c"], state["b"] = lg[0], lg[1], lg[2]
                    continue
                take(mid_c, mid_b, "bisect")
                continue
            # No readable point known yet (a saturated start): retreat both
            # knobs from the pinning rail at once -- cures a gain overload
            # and an offset overload alike without having to tell them
            # apart. Steps grow geometrically; crossing to the opposite
            # rail reverses at a quarter of the step, a bisection along the
            # retreat track that cannot strand outside a narrow window.
            direction = -1.0 if bal.white_clip >= bal.black_clip else 1.0
            if state["esc_dir"] and direction != state["esc_dir"]:
                state["esc"] = max(_CB_ESCAPE_SEED / 4.0,
                                   state["esc"] * 0.25)
            elif state["esc_dir"]:
                state["esc"] = min(_CB_ESCAPE_CAP, state["esc"] * 2.0)
            state["esc_dir"] = direction
            step = direction * state["esc"]
            new_c = _clamp(state["c"] + step, lo_b, hi_b)
            new_b = _clamp(state["b"] + step, lo_b, hi_b)
            if (abs(new_c - state["c"]) < 1e-6
                    and abs(new_b - state["b"]) < 1e-6):
                if note is not None:
                    note("Auto CB: the image is pinned at a rail even at "
                         "the knob bounds - the scene carries no usable "
                         "histogram")
                return _result("saturated")
            take(new_c, new_b, "escape")
            continue

        hard_w = bal.white_clip >= _CB_HARD_CLIP_FRACTION
        hard_b = bal.black_clip >= _CB_HARD_CLIP_FRACTION
        if hard_w or hard_b:
            # A rail-dominated frame's spread reading lies: the pinned
            # percentiles squeeze together and the solver would pump gain
            # deeper into the rail. The trustworthy signal is which rail is
            # overloaded; relieve it directly.
            rail_side = -1.0 if hard_w and not hard_b else 1.0
            prev_rail = state.get("last_rail")
            same_rail = (bool(prev_rail) and not (hard_w and hard_b)
                         and rail_side == prev_rail)
            if (prev_rail and not (hard_w and hard_b)
                    and rail_side != prev_rail):
                # Consecutive reliefs flipped the rail: the step crossed
                # the whole usable window. Halve the caps so the next
                # relief bisects into it instead of ping-ponging over it.
                state["cap_c"] = max(_CB_STEP_CAP_MIN, state["cap_c"] * 0.5)
                state["cap_b"] = max(_CB_STEP_CAP_MIN, state["cap_b"] * 0.5)
            state["last_rail"] = 0.0 if (hard_w and hard_b) else rail_side
            if hard_w and hard_b:
                # Bimodal: the occupied spread does not fit at this gain.
                # toward still needs a defined direction (relieve the heavier
                # rail) for the pinned-knob fallback below -- it previously
                # went unassigned on this path, an UnboundLocalError (or a
                # stale direction from an earlier single-rail pass) when the
                # compress move was a no-op at the contrast bound.
                toward = -1.0 if bal.white_clip >= bal.black_clip else 1.0
                dc, db, tag = -state["cap_c"], 0.0, "compress"
            else:
                toward = rail_side
                if (abs(state["last_dc"]) > abs(state["last_db"])
                        and state["last_dc"] * toward < 0):
                    # The last move was predominantly a gain move toward
                    # this rail: the gain caused it, so the gain relieves
                    # it (the v3.3.14 causal lesson -- laddering the other
                    # knob against a gain overshoot wanders).
                    if same_rail:
                        # Re-entering the rail the last relief fled, by
                        # the knob that caused it: full-cap retreat and
                        # full-slope solve were cycling over the window
                        # (the 2026-08-26 stale-seed 2-cycle). Halve the
                        # causal cap so the retreat bisects into it.
                        state["cap_c"] = max(_CB_STEP_CAP_MIN,
                                             state["cap_c"] * 0.5)
                    dc, db, tag = toward * state["cap_c"], 0.0, "declip"
                elif (rail_side < 0 and prev_rail == rail_side
                        and abs(state["last_dc"]) <= 1e-12
                        and state["last_db"] * toward > 0):
                    # An offset relief just fled this same white rail and
                    # the rail still stands: measured evidence the offset
                    # lacks clip authority here (the ModerateOvershoot
                    # band's regime -- its wclip barely answers to
                    # brightness), so the gain takes over. One bounded
                    # offset probe, never a ladder: on plants where the
                    # offset does have authority its first relief clears
                    # or flips the rail and this branch is never reached,
                    # which keeps the ep5-ep8 fitted plants
                    # (offset-curable clips) on their efficient reliefs.
                    # White side only: gain-down is the intrinsically safe
                    # retreat (less clipping), while a black-side gain-up
                    # is an expansion move that blows a harsh plant white
                    # from a standing black rail whose offset relief was
                    # only ever partial (the Hydra flip-bisection's rungs
                    # each leave the rail standing on purpose); the
                    # bottomed-out black case is already covered by the
                    # pinned-knob fallback below.
                    dc, db, tag = toward * state["cap_c"], 0.0, "declip"
                else:
                    if same_rail and state["last_db"] * toward < 0:
                        # Same-rail re-entry after the offset moved toward
                        # this rail: the offset caused it, so only its cap
                        # tightens. An away-move that is still railed is
                        # insufficient relief and must repeat at the full,
                        # unhalved cap -- the anti-stranding rule.
                        state["cap_b"] = max(_CB_STEP_CAP_MIN,
                                             state["cap_b"] * 0.5)
                    dc, db, tag = 0.0, toward * state["cap_b"], "declip"
            new_c = _clamp(state["c"] + dc, lo_b, hi_b)
            new_b = _clamp(state["b"] + db, lo_b, hi_b)
            if (abs(new_c - state["c"]) < 1e-6
                    and abs(new_b - state["b"]) < 1e-6):
                # The preferred knob is pinned; relieve with the other one
                # before giving up (field ep1: brightness bottomed out
                # while the hot gain still pinned the frame white).
                if abs(db) > abs(dc):
                    new_c = _clamp(state["c"] + toward * state["cap_c"],
                                   lo_b, hi_b)
                else:
                    new_b = _clamp(state["b"] + toward * state["cap_b"],
                                   lo_b, hi_b)
                if (abs(new_c - state["c"]) < 1e-6
                        and abs(new_b - state["b"]) < 1e-6):
                    return _result("no-move")
            nbal, prev = take(new_c, new_b, tag)
            if nbal is not None:
                _learn(nbal, prev)
            continue

        # last_rail deliberately survives readable frames: a solve that
        # re-enters the rail the last relief fled must be recognized as a
        # same-rail re-entry (the 2026-08-26 stale-seed 2-cycle), not as a
        # fresh rail with a fresh full cap.
        lo, hi = _cb_edges(bal)
        spread = max(hi - lo, 1e-6)
        mid = (lo + hi) / 2.0
        # Solve: contrast from the spread ratio, then brightness from the
        # residual midpoint error the gain move leaves behind -- the offset
        # is applied after the gain stage, so the chosen gain change scales the
        # midpoint predictably (10^(k_c * dc)) and solving brightness
        # against the raw midpoint would fight that coupling for the whole
        # budget. Each slope is a persisted or learned per-tool value, else
        # the sign-correct default; steps capped per knob.
        dc = _clamp(math.log10(target_spread / spread) / state["k_c"],
                    -state["cap_c"], state["cap_c"])
        new_c = _clamp(state["c"] + dc, lo_b, hi_b)
        mid_after_gain = mid * (10.0 ** (state["k_c"]
                                         * (new_c - state["c"])))
        db = _clamp((target_mid - mid_after_gain) / state["k_b"],
                    -state["cap_b"], state["cap_b"])
        new_b = _clamp(state["b"] + db, lo_b, hi_b)
        if (abs(new_c - state["c"]) < 1e-6
                and abs(new_b - state["b"]) < 1e-6):
            # The residual error is below the solvable resolution (or the
            # knobs are pinned at their bounds in the needed direction).
            return _result("no-move")
        nbal, prev = take(new_c, new_b, "solve")
        if nbal is not None:
            _learn(nbal, prev)
            _rail_evidence(nbal, prev)
    return _result("budget-exhausted")


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))
