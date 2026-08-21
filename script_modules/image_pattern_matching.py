"""
Image Pattern Matching Module

This module provides template matching functions for comparing RTM images over time.

The main function compares regions between a reference and current image using
tiled template matching to detect how much the pattern has changed.

Matching is performed using OpenCV's TM_SQDIFF_NORMED method, which normalizes
the squared difference by the two patches' energy (geometric mean of their sums
of squares), yielding scores in [0.0, 1.0]. That normalization cancels a
multiplicative scaling applied to BOTH patches together; a gain change BETWEEN
the reference and current frame is NOT cancelled (SQDIFF_NORMED(a, k*a) =
(1-k)^2/k), so a uniform inter-frame gain is estimated and divided out before
scoring (see estimate_interframe_gain; the caller decides -- a one-off step is
a CB/detector event, consecutive same-pattern corrections are refused as real
uniform change). Two situations never publish a raw
TM_SQDIFF_NORMED value: tiles where either side carries almost no signal (the
energy normalization is degenerate there; OpenCV clamps to exactly 1.0) are
scored by counting significantly-changed pixels instead, and tiles where the
metric structurally saturates (for same-size patches the true value is
r + 1/r - 2*rho, r = energy ratio, rho = cosine similarity, which exceeds 1.0
whenever rho <= (r + 1/r - 1)/2 -- OpenCV clamps it to exactly 1.0, discarding
all magnitude) are demoted to the same change-count measure, branch "cv2sat".
"""
import logging
import numpy as np
from cv2 import matchTemplate, minMaxLoc, TM_SQDIFF_NORMED
from skimage.util import img_as_ubyte
from math import sqrt


logger = logging.getLogger(__name__)


# A uint8 pixel above this level (~4% of full scale) counts as "significant"
# signal. Inputs arrive at TRUE amplitude (raw counts / 255, fixed scale --
# tophat_normalized on the foreground path, filter_images' with_match_image
# output on the grayscale path), so this gates on physical residual counts. Measured
# on the 2026-08-17 field campaign: noise ceiling 2-4 counts (even on single
# frames); faintest real residual texture 11-24 counts. 10 sits between those
# bands. This level ONLY routes a tile (absolute-scale vs cv2 scoring); it does
# not decide whether a tile changed -- that is SIGNIFICANT_CHANGE_LEVEL below,
# which is what keeps a dim-but-real scene from reading as settled.
SIGNIFICANT_PIXEL_LEVEL = 10

# A tile with this many significant pixels or fewer on EITHER side is "sparse":
# TM_SQDIFF_NORMED's energy normalization is degenerate there (a blank or
# near-blank patch makes OpenCV clamp the result to exactly 1.0 -- the on-tool
# stuck-at-1.0 failure, where one relocated stray pixel pinned the score all
# run). At fixed scale, stray counts sit below SIGNIFICANT_PIXEL_LEVEL, so a
# noise-only tile has 0 significant pixels; the faintest real features (~13 px
# above the level, concentrated in 1-2 tiles) put >= ~6 in their tile. 3 sits
# safely between.
SPARSE_TILE_PIXELS = 3

# A pixel whose absolute frame-to-frame DIFFERENCE exceeds this level (raw
# counts) counts as genuinely changed. Sparse tiles are scored by counting
# these, not by summing their energy and not by their absolute brightness:
# scoring on brightness alone made any change dimmer than SIGNIFICANT_PIXEL_LEVEL
# read as ~0 no matter how much of the crop it covered, so a dim scene still
# being milled could satisfy the completion criterion (an early-declare cliff at
# exactly 10 counts; a whole crop appearing at 10 counts scored 0.0015 while the
# same crop at 12 counts scored 1.0).
# Measured separation on the 2026-08-17 campaign (consecutive-batch top-hat
# crops, worst case single frames rather than batch means): milled-through tails
# that MUST settle peak at |diff| = 1-3 counts with ZERO pixels above 4, while
# active milling shows |diff| up to 70-95 counts with 460-1230 pixels above 4.
# 4 sits in that gap with an order of magnitude of margin on both sides. Tuning
# LOWER admits noise (score hangs high; completion blocked -- safe but annoying);
# HIGHER hides real change (declares complete early -- the dangerous direction).
SIGNIFICANT_CHANGE_LEVEL = 4

# Normalizer for sparse tiles: the count of significantly-changed pixels is
# scored against this fixed feature-scale area rather than the tile area, so a
# small real event stays decisive instead of being diluted by tile size.
# Counting is amplitude-independent, which matters at true amplitude: an energy
# sum under-reports a faint real event (156 px at 15 counts vanishing --
# Run-3 P1's measured residual texture -- sums to 0.008, silently reading "no
# change" on the very batch the last feature disappears), while its count
# saturates the clamp. Calibration: <= 3 strays -> 3/64 ~ 0.047 stays under the
# 0.05 threshold; a 13-px real event -> 13/64 ~ 0.20 stays decisive; >= 64
# changed px reads as maximum change.
SPARSE_ASYM_NORM_PIXELS = 64

# Minimum image dimension (pixels) for meaningful tiled template matching.
# Images smaller than this in either dimension produce tiles too small for
# reliable comparison. Returns None (score unavailable).
MIN_IMAGE_DIMENSION = 5

# Inter-frame gain normalization. TM_SQDIFF_NORMED does not cancel a gain
# change BETWEEN frames: SQDIFF_NORMED(a, k*a) = (1-k)^2/k, so a >=~25% CB /
# beam-current / detector step between consecutive batches scores above the
# completion threshold on settled dense tiles (resetting confirmations) and can
# trip the delay-phase match transition before milling starts. When the current
# frame is a uniform multiple of the reference over the jointly-significant
# pixels, that gain is divided out before scoring. The gates below make the
# estimator refuse anything that is not gain-like, so REAL change (a milling
# front, which is spatially heterogeneous and moves content across the
# significance level) is never normalized away:
GAIN_NORM_MIN_PIXELS = 16     # fewer jointly-significant px: tiles are sparse-
                              # dominated (immune to gain) and a median is noise
GAIN_NORM_MIN_SHIFT = 0.05    # |k-1| below this is within noise; don't touch
GAIN_NORM_RANGE = (0.6, 1.7)  # k outside: not a plausible gain step; leave the
                              # frames alone and let the score flag it
GAIN_NORM_MAX_SPREAD = 0.15   # IQR(ratios)/k above this: heterogeneous change
                              # (milling), not a uniform gain -- do not correct
GAIN_NORM_MAX_COUNT_RATIO = 1.25  # significant-pixel counts must agree within
                                  # this factor (gain moves few pixels across
                                  # the level; milling adds/removes many)


def _count_changed_pixels(last_tile, first_tile):
    """Count pixels whose absolute frame-to-frame difference exceeds
    SIGNIFICANT_CHANGE_LEVEL (raw counts). Shared by the sparse branch and the
    cv2 saturation demotion."""
    diff = np.abs(last_tile.astype(np.float32) - first_tile.astype(np.float32))
    return int(np.count_nonzero(diff > SIGNIFICANT_CHANGE_LEVEL))


def estimate_interframe_gain(first_image, last_image):
    """
    Estimate the uniform multiplicative gain of the current frame relative to
    the reference, over jointly-significant pixels.

    Returns the gain k when -- and only when -- the frames differ by a
    confident, uniform multiplicative factor worth correcting; otherwise None.
    Inputs are float images in [0.0, 1.0] at fixed absolute scale.
    """
    ref = np.asarray(first_image, dtype=np.float32)
    cur = np.asarray(last_image, dtype=np.float32)
    level = SIGNIFICANT_PIXEL_LEVEL / 255.0
    ref_sig = ref > level
    cur_sig = cur > level
    n_ref = int(np.count_nonzero(ref_sig))
    n_cur = int(np.count_nonzero(cur_sig))
    if min(n_ref, n_cur) < GAIN_NORM_MIN_PIXELS:
        return None
    if max(n_ref, n_cur) > GAIN_NORM_MAX_COUNT_RATIO * min(n_ref, n_cur):
        return None
    mask = ref_sig & cur_sig
    if int(np.count_nonzero(mask)) < GAIN_NORM_MIN_PIXELS:
        return None
    ratios = cur[mask] / ref[mask]
    k = float(np.median(ratios))
    if not (GAIN_NORM_RANGE[0] <= k <= GAIN_NORM_RANGE[1]):
        return None
    if abs(k - 1.0) < GAIN_NORM_MIN_SHIFT:
        return None
    q1, q3 = np.percentile(ratios, [25.0, 75.0])
    if (q3 - q1) / k > GAIN_NORM_MAX_SPREAD:
        return None
    return k


def calculate_match_score(first_image, last_image, min_splits=3, target_tile_size=100,
                          log_tag="", interframe_gain=None):
    """
    Calculate a pattern match score between two images using tiled template matching.

    This function splits both images into NxN tiles and compares corresponding tiles.
    The maximum match score across all tiles is returned, providing sensitivity to
    localized changes in any region of the image.

    The number of tiles adapts to image size:
    - Larger images get more tiles for finer resolution
    - Smaller images use fewer tiles (minimum of min_splits x min_splits)
    - This prevents over-splitting small crops or under-splitting large crops

    Workflow:
    1. Convert images from [0.0, 1.0] float to uint8 (required by cv2.matchTemplate)
    2. Calculate optimal number of splits based on image area and target tile size
    3. Split images into NxN grid
    4. Compare each tile pair: TM_SQDIFF_NORMED for tiles with real content on both
       sides; change-count scoring for sparse tiles (below)
    5. Return the maximum match score across all tiles

    Input images are expected in [0.0, 1.0] range at FIXED absolute scale
    (counts / 255: tophat_normalized output on the foreground path,
    filter_images' with_match_image output on the grayscale path), so the
    conversion to uint8 restores raw counts without any contrast stretching.

    TM_SQDIFF_NORMED normalizes by patch energy (geometric mean of the two patches'
    sums of squares), producing values in [0.0, 1.0]. 0.0 = perfect match, 1.0 =
    maximum difference. The normalization cancels a scaling applied to both frames
    together only; a uniform gain change BETWEEN the frames is estimated and
    divided out before scoring (never when the change is heterogeneous, i.e. real
    -- see estimate_interframe_gain). A tile whose TM_SQDIFF_NORMED value
    saturates at >= 1.0 (structurally possible at low cosine similarity; OpenCV
    clamps it to exactly 1.0, discarding magnitude) is demoted to the change-count
    score, branch "cv2sat".

    Note on sparse tiles:
    When either side of a tile pair carries almost no signal (a milled-through
    region that is blank except stray residual counts), the energy normalization is
    degenerate: OpenCV clamps the result to exactly 1.0 both for a zero-energy patch
    and for two isolated spikes in different places, falsely reporting maximum
    difference forever. Such tiles are detected by counting significant pixels and
    scored on absolute scale instead: the count of pixels whose frame-to-frame
    difference exceeds SIGNIFICANT_CHANGE_LEVEL, over SPARSE_ASYM_NORM_PIXELS. That
    measure is independent of how BRIGHT the content is, so a genuine appearance /
    disappearance stays decisive even in a dim regime, while stray-count churn
    (which changes nothing by more than a count or two) scores ~0.

    :param first_image: 2D numpy array in [0.0, 1.0] range (reference image)
    :param last_image: 2D numpy array in [0.0, 1.0] range (current image)
    :param min_splits: Minimum number of splits per dimension (default: 3, the
        on-tool-validated value; kept in sync with ProcessingParameters)
    :param target_tile_size: Target size for each tile dimension in pixels (default: 100)
    :param log_tag: Optional prefix (e.g. "Pattern 1: ") for log attribution
    :param interframe_gain: Caller-supplied uniform gain of last_image relative
        to first_image (from estimate_interframe_gain), divided out before
        scoring; None = no correction. The DECISION to correct lives with the
        caller because it needs cross-batch state: a genuine CB/detector step
        is a one-off event, while a sustained same-direction "gain" on
        consecutive batches is real uniform milling change that must NOT be
        normalized away (the worker refuses consecutive corrections)
    :return: Match score as float (0.0 = perfect match, 1.0 = maximum difference),
        or None when no score can be computed (shape mismatch / image too small /
        no valid tiles). Callers must treat None as "score unavailable", NEVER as a
        match -- a 0.0 here would satisfy the completion criterion and could stop
        milling early on a broken input.
    """
    # Guard: images must have identical dimensions for tile-based comparison
    if first_image.shape != last_image.shape:
        logger.warning(
            f"{log_tag}Shape mismatch between reference {first_image.shape} and "
            f"current {last_image.shape} - score skipped (cannot compare)"
        )
        return None

    # Guard: images too small for meaningful tiled comparison
    height, width = first_image.shape[:2]
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        logger.warning(
            f"{log_tag}Image too small for tiled matching ({width}x{height}, "
            f"minimum {MIN_IMAGE_DIMENSION}px per dimension) - score skipped"
        )
        return None

    # Inter-frame gain normalization (caller-decided; see the param docstring).
    # Applied by scaling the BRIGHTER side down -- never up -- so no value can
    # exceed full scale: dividing the current frame by k < 1 would inflate
    # near-full-scale pixels past 1.0 and the necessary clip would erase real
    # change against the ceiling (TM_SQDIFF_NORMED is identical either way).
    if interframe_gain is not None:
        k = float(interframe_gain)
        if k >= 1.0:
            last_image = np.asarray(last_image, dtype=np.float32) / k
        else:
            first_image = np.asarray(first_image, dtype=np.float32) * k
        logger.info(
            f"{log_tag}Uniform inter-frame gain {k:.3f} normalized out "
            f"before matching"
        )

    # Convert to uint8 for cv2.matchTemplate
    # Input images are already in [0.0, 1.0] at fixed absolute scale,
    # so img_as_ubyte maps directly to 0-255 without any contrast stretching.
    last_image_uint8 = img_as_ubyte(last_image)
    first_image_uint8 = img_as_ubyte(first_image)

    # Calculate optimal number of splits based on image area and target tile size
    num_splits = _calculate_num_splits(last_image_uint8.shape, min_splits, target_tile_size)

    # Split images into tiles
    last_tiles = _split_image_into_tiles(last_image_uint8, num_splits)
    first_tiles = _split_image_into_tiles(first_image_uint8, num_splits)

    # Compare all tile pairs and collect (score, row, col, branch) so the
    # published max is attributable in the log: a 1.0 from "asym" (real content
    # appeared/disappeared) and a 1.0 from "cv2" (degenerate energy input)
    # mean different things when triaging a run.
    tile_scores: list[tuple[float, int, int, str]] = []
    sparse_tiles = 0

    for i in range(num_splits):
        for j in range(num_splits):
            # Get corresponding tiles
            last_tile = last_tiles[i][j]
            first_tile = first_tiles[i][j]

            # Guard: skip empty tiles (can occur if array_split produces
            # zero-dimension slices at edges)
            if last_tile.size == 0 or first_tile.size == 0:
                continue

            # Sparse gate: if either side is empty/noise-like, TM_SQDIFF_NORMED's
            # energy normalization is degenerate (OpenCV clamps to exactly 1.0 for
            # a blank patch or two spikes in different places). Score such tiles on
            # absolute scale instead -- an energy-normalized score cannot tell a
            # 1-px noise spike from a real feature.
            sig_last = int(np.count_nonzero(last_tile > SIGNIFICANT_PIXEL_LEVEL))
            sig_first = int(np.count_nonzero(first_tile > SIGNIFICANT_PIXEL_LEVEL))
            if sig_last <= SPARSE_TILE_PIXELS or sig_first <= SPARSE_TILE_PIXELS:
                # Score by COUNTING significantly-changed pixels against a fixed
                # feature-scale area. This is the same measure whether the tile
                # is empty on both sides (stray-pixel churn -> 0 changed -> ~0)
                # or content appeared/disappeared on one side, so a real event
                # is never missed for being dim: what matters is how much of the
                # tile changed, not how bright it is (see SIGNIFICANT_CHANGE_LEVEL).
                changed = _count_changed_pixels(last_tile, first_tile)
                score = min(1.0, changed / SPARSE_ASYM_NORM_PIXELS)
                # Branch label is diagnostic only (both formulas are identical):
                # "sym" = nothing on either side, "asym" = content on one side.
                branch = ("sym"
                          if (sig_last <= SPARSE_TILE_PIXELS
                              and sig_first <= SPARSE_TILE_PIXELS)
                          else "asym")
                sparse_tiles += 1
            else:
                # Both sides carry real content: energy-normalized matching
                # (0 = perfect match). Tiles are same-size, so the result is a
                # single value, not a translation search.
                result = matchTemplate(last_tile, first_tile, TM_SQDIFF_NORMED)
                score = float(minMaxLoc(result)[0])
                branch = "cv2"
                if score >= 1.0:
                    # Structural saturation: the metric's true value here is
                    # r + 1/r - 2*rho (>= 1.0 whenever cosine similarity rho is
                    # low enough), and OpenCV clamps it to exactly 1.0 --
                    # discarding all magnitude. Substitute the module's own
                    # change-count measure so the tail stays graded, FLOORED at
                    # the smallest count field evidence covers: every one of
                    # the 465 saturated tiles in the 2026-08-17 campaign had
                    # >= 6 changed px (honest score >= 0.094 = ~2x threshold).
                    # A saturated dense tile provably had a large relative
                    # energy change, so a tiny count (e.g. a compact 4-px
                    # cluster vanishing) must not map BELOW threshold -- the
                    # floor keeps such an event blocking for the one batch it
                    # is visible, exactly as the pre-demotion clamp did, while
                    # every field-observed tile's score is unchanged.
                    changed = _count_changed_pixels(last_tile, first_tile)
                    score = min(1.0, max(changed, 6) / SPARSE_ASYM_NORM_PIXELS)
                    branch = "cv2sat"

            tile_scores.append((score, i, j, branch))

    # Return the maximum match score across all tiles
    if not tile_scores:
        logger.warning(f"{log_tag}No valid tiles for comparison - score skipped")
        return None

    max_match_score, max_row, max_col, max_branch = max(
        tile_scores, key=lambda entry: entry[0])

    # Per-tile dump capped: a large uncropped frame can produce 100+ tiles,
    # and an unbounded INFO line per batch per pattern bloats a (possibly
    # network-hosted) log. Above the cap, log only the highest scores.
    if len(tile_scores) <= 25:
        tile_summary = ", ".join(
            f"{score:.4f}/{branch}" for score, _, _, branch in tile_scores)
    else:
        top = sorted(tile_scores, key=lambda entry: entry[0], reverse=True)[:5]
        tile_summary = f"top 5 of {len(tile_scores)}: " + ", ".join(
            f"{score:.4f}/{branch}@({row},{col})" for score, row, col, branch in top)
    logger.info(
        f"{log_tag}Match score {max_match_score:.4f} from tile "
        f"({max_row},{max_col}) [{max_branch}]; {len(tile_scores)} tiles, "
        f"{sparse_tiles} sparse-fallback: [{tile_summary}]"
    )

    return max_match_score


def _calculate_num_splits(image_shape, min_splits=3, target_tile_size=100):
    """
    Calculate the optimal number of splits for an image based on its area.

    This ensures:
    - Small cropped images don't get over-divided into tiny tiles
    - Large images get sufficient tiling for detailed comparison
    - Splits never exceed the smallest image dimension (prevents empty tiles)

    The algorithm aims to create tiles of approximately target_tile_size x target_tile_size
    pixels (default 100x100), which provides good template matching resolution without
    over-segmentation.

    Formula:
    - area = width x height
    - target_area = target_tile_size^2  (e.g., 100^2 = 10,000 px per tile)
    - split_factor = area / target_area
    - num_splits = max(min_splits, round(sqrt(split_factor)))

    Examples with target_tile_size=100:
    - 100x100 image (10,000 px) -> split_factor=1 -> sqrt(1)=1 -> use min_splits (3)
    - 256x256 image (65,536 px) -> split_factor=6.5 -> sqrt~2.5 -> round to 3 -> use min_splits (3)
    - 512x512 image (262,144 px) -> split_factor=26.2 -> sqrt~5.1 -> round to 5 -> use 5
    - 1024x1024 image (1,048,576 px) -> split_factor=104.9 -> sqrt~10.2 -> round to 10 -> use 10

    :param image_shape: Tuple (height, width) from image.shape
    :param min_splits: Minimum number of splits (default: 3, kept in sync with
        ProcessingParameters.min_pattern_splits)
    :param target_tile_size: Target dimension for each tile in pixels (default: 100)
    :return: Number of splits per dimension as int
    """
    height, width = image_shape
    area = width * height

    # Calculate split factor based on target tile area
    target_area = target_tile_size ** 2
    split_factor = area / target_area

    # Take square root and round
    sqrt_factor = round(sqrt(split_factor))

    # Use minimum or calculated value, whichever is larger
    num_splits = max(min_splits, sqrt_factor)

    # Cap at smallest image dimension to prevent empty tiles from array_split
    num_splits = min(num_splits, height, width)

    # Ensure at least 1 split (degenerate images)
    num_splits = max(1, num_splits)

    logger.debug(f"Image width, height: Width: {width}, Height: {height}")

    return num_splits


def _split_image_into_tiles(image, num_splits):
    """
    Split an image into NxN tiles.

    :param image: 2D numpy array
    :param num_splits: Number of splits per dimension
    :return: 2D list of tiles [row][col]
    """
    # Split vertically (into rows)
    rows = np.array_split(image, num_splits, axis=0)

    # Split each row horizontally (into columns)
    tiles = []
    for row in rows:
        cols = np.array_split(row, num_splits, axis=1)
        tiles.append(cols)

    logger.debug(f"Number of rows, columns: Rows: {len(rows)}, Columns: {len(cols)}")

    return tiles