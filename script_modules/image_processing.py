"""
Image Processing Module

This module provides image filtering and thresholding functions for processing 
RTM images.

Main functions:
- filter_images(): Complete background subtraction workflow
- threshold_adaptive(): Adaptive binarization for white pixel counting
"""

import logging

import numpy as np
from skimage import filters, util, morphology, exposure


logger = logging.getLogger(__name__)

# Dynamic-range floor below which a filtered frame is treated as uniform
# (fully-milled / blank). Below this, rescale_intensity would amplify pure
# floating-point noise to span [0, 1] and corrupt every downstream metric.
_UNIFORM_FRAME_EPS = 1e-9

# A frame whose dynamic range is carried by this many pixels or fewer is noise,
# not signal, and must not be per-frame rescaled: rescaling would stretch 1-2
# stray counts to full scale (observed on-tool: ONE residual count in a 7614-px
# crop became a 255 spike, corrupting the display / mean-pixel / white-pixel
# metrics). The faintest real features seen on-tool are ~13+ pixels (0.17%
# coverage), well above this floor, so genuine faint signal always takes the
# rescale path. Both guards protect the RESCALED (display/metrics) output only;
# the fixed-scale match image (with_match_image=True) is never rescaled and
# needs no guard.
_SPARSE_CONTENT_PIXELS = 3
_RAW_FULL_SCALE = 255.0

# One-time-warning latch for tophat_normalized inputs exceeding full scale.
_tophat_clip_warned = False


# =======================
# Core Filtering Workflow
# =======================

def filter_images(image_list, gaussian_sigma=1.0, apply_dilation=True,
                  with_match_image=False):
    """
    Process a batch of images using background subtraction to isolate temporal changes.

    This is the main filtering workflow for RTM monitoring. It removes static background
    features and highlights areas that have changed during patterning.

    The filtering workflow:
    1. Average all images in the batch (creates background reference)
    2. Apply Gaussian blur to the average
    3. Optionally apply morphological dilation to the average
    4. Apply Gaussian blur to the last (current) image
    5. Optionally apply morphological dilation to the last image
    6. Subtract the processed average from the processed last image
    7. Invert the result
    8. Rescale to 0.0-1.0 range

    The output is always normalized to [0.0, 1.0] so all downstream consumers
    (display, thresholding, mean pixel calculation) receive data in a
    consistent, predictable range.

    With ``with_match_image=True`` a SECOND image is returned for template
    matching: the SIGNED change map (current - background, same blur/dilation
    processing) at FIXED absolute scale, counts / _RAW_FULL_SCALE clipped to
    [-1, 1]. Unlike the display output it is never per-frame stretched, so
    identical scenes produce identical match images and residual noise counts
    stay at true amplitude -- the property the matcher's raw-count calibration
    (SIGNIFICANT_PIXEL_LEVEL / SIGNIFICANT_CHANGE_LEVEL) depends on. The sign
    is preserved because a contrast INVERSION (a feature flipping from
    brighter-than-background to dimmer, e.g. charging oscillation or
    breakthrough) has an identical magnitude map and would otherwise read as
    "no change" (adversarial review 2026-08-18). Fold it into the matcher's
    [0, 1] domain with signed_to_match_image() AFTER cropping. The
    uniform/sparse guards below do not apply to it: they exist to protect the
    rescale step, and the match image has none.

    :param image_list: List of 2D numpy arrays (images from RTM, raw counts)
    :param gaussian_sigma: Standard deviation for Gaussian kernel (default: 1.0)
    :param apply_dilation: Whether to apply morphological dilation (default: True)
    :param with_match_image: Also return the fixed-scale signed change map
        (default: False)
    :return: Filtered 2D numpy array in [0.0, 1.0] range highlighting temporal
        changes; or a tuple ``(filtered, match_image)`` when with_match_image is
        True, where match_image is float32 in [-1.0, 1.0] at fixed absolute
        scale (crop it, then fold with signed_to_match_image for the matcher)
    """
    if not image_list:
        raise ValueError("image_list cannot be empty")

    last_image = image_list[-1]

    # Create background reference by averaging all images
    background = blend_images(image_list)
    # Pre-blur sparse-content check: how many pixels actually differ between the
    # current frame and the batch mean, measured in RAW counts BEFORE the Gaussian
    # smears a single stray count over ~20 px (which would defeat a count-based
    # floor applied after filtering).
    raw_dev = np.abs(np.asarray(last_image, dtype=float)
                     - np.asarray(background, dtype=float))
    raw_dev_max = float(raw_dev.max())
    raw_content_sparse = (
        raw_dev_max < _UNIFORM_FRAME_EPS
        or int(np.count_nonzero(raw_dev > 0.1 * raw_dev_max)) <= _SPARSE_CONTENT_PIXELS
    )
    background = apply_gaussian_filter(background, gaussian_sigma)
    if apply_dilation:
        background = morphological_dilation(background)

    # Process the current (last) image
    current = apply_gaussian_filter(last_image, gaussian_sigma)
    if apply_dilation:
        current = morphological_dilation(current)

    # Subtract background and invert to highlight changes
    diff_image = subtract_images(current, background)
    filtered_image = invert_image(diff_image)

    # Fixed-scale SIGNED match image: (current - background) in raw counts
    # mapped by full scale, never stretched. Computed before the guards --
    # they protect the rescale path only, and without a rescale there is
    # nothing to amplify: 1-3 count strays stay at 1-3 counts and score ~0 in
    # the matcher.
    match_image = (
        np.clip(diff_image / _RAW_FULL_SCALE, -1.0, 1.0).astype(np.float32)
        if with_match_image else None
    )

    # Guard against a near-uniform (e.g. fully-milled / blank) frame: per-frame
    # rescale_intensity below would stretch pure floating-point noise to span
    # [0, 1], producing garbage that corrupts mean-pixel / white-pixel / match
    # metrics and can trigger the multi-Otsu raise. Return a flat zero frame.
    if float(np.ptp(filtered_image)) < _UNIFORM_FRAME_EPS:
        zero = np.zeros_like(filtered_image)
        return (zero, match_image) if with_match_image else zero

    # Sparse-content guard (measured pre-blur, above): a frame whose only change
    # vs the batch mean is a couple of stray counts is noise, not signal --
    # rescaling would promote those strays (smeared by the Gaussian) to full
    # scale, which is what pinned the match score at the degenerate 1.0 on-tool.
    if raw_content_sparse:
        zero = np.zeros_like(filtered_image)
        return (zero, match_image) if with_match_image else zero

    # Normalize to [0.0, 1.0] so downstream consumers get a consistent range
    filtered_image = exposure.rescale_intensity(filtered_image, out_range=(0.0, 1.0))

    return (filtered_image, match_image) if with_match_image else filtered_image


def signed_to_match_image(signed_change_map):
    """
    Fold a signed fixed-scale change map ([-1, 1], from filter_images'
    with_match_image output, cropped to the region of interest) into the
    matcher's [0, 1] domain: positive change in the left half, negative change
    in the right half.

    Splitting by sign instead of taking |diff| preserves polarity (a contrast
    inversion moves content between halves and reads as change) without a
    mid-scale pedestal, which would crush TM_SQDIFF_NORMED's sensitivity
    (score ~ n*delta^2 / (N*pedestal^2)) and defeat the sparse-tile gate.
    Each half stays at true amplitude, so the matcher's raw-count calibration
    holds per half.

    :param signed_change_map: 2D float array in [-1.0, 1.0] at counts/255 scale
    :return: float32 2D array in [0.0, 1.0], twice the input width
    """
    signed = np.asarray(signed_change_map, dtype=np.float32)
    return np.hstack([
        np.clip(signed, 0.0, 1.0),
        np.clip(-signed, 0.0, 1.0),
    ]).astype(np.float32)


# ===============================
# Individual Filtering Operations
# ===============================

def blend_images(image_list):
    """
    Calculate the average of multiple images to create a background reference.

    This is used to create a static background from a batch of images,
    which can then be subtracted to isolate temporal changes.

    Uses np.stack + np.mean to avoid N-1 intermediate array allocations
    that occur with Python's sum() on numpy arrays.

    :param image_list: List of 2D numpy arrays
    :return: Blended image as 2D numpy array
    """
    if not image_list:
        raise ValueError("image_list cannot be empty")

    return np.mean(np.stack(image_list), axis=0)


def apply_gaussian_filter(image, sigma=1.0):
    """
    Apply Gaussian blur to smooth an image and reduce noise.

    :param image: 2D numpy array
    :param sigma: Standard deviation for Gaussian kernel (controls blur strength)
    :return: Filtered 2D numpy array
    """
    return filters.gaussian(image, sigma)


def morphological_dilation(image, selem=None):
    """
    Apply morphological dilation to expand bright regions in an image.

    Dilation is useful for connecting nearby features and filling small gaps.

    :param image: 2D numpy array
    :param selem: Structuring element (None = default disk)
    :return: Dilated 2D numpy array
    """
    return morphology.dilation(image, selem)


def subtract_images(image1, image2):
    """
    Subtract one image from another to isolate differences.

    This is the core of background subtraction: subtracting a static
    background reference from the current image reveals what has changed.

    :param image1: 2D numpy array (current image)
    :param image2: 2D numpy array (background to subtract)
    :return: Difference image as 2D numpy array
    """
    return np.subtract(image1, image2)


def invert_image(image):
    """
    Invert an image (flip pixel intensities).

    :param image: 2D numpy array
    :return: Inverted 2D numpy array
    """
    return util.invert(image)


# ===========================
# Thresholding / Binarization
# ===========================

def threshold_multiotsu(image, num_classes=2):
    """
    Apply multi-Otsu thresholding to create a binary image.

    Multi-Otsu automatically determines optimal threshold values by analyzing
    the image histogram. For num_classes=2, it creates a simple binary threshold.

    This is used during the "delay" phase when we don't yet have a reference
    mean pixel value.

    :param image: 2D numpy array
    :param num_classes: Number of classes for multi-Otsu (default: 2 for binary)
    :return: Binary threshold image (boolean array)
    """
    thresholds = calculate_otsu_threshold_values(image, num_classes=num_classes)
    threshold_image = image >= thresholds[-1]
    return threshold_image

def calculate_otsu_threshold_values(image, num_classes=2):
    """
    Calculate multi-Otsu threshold values without applying them.

    Returns the raw threshold values that optimally separate the image
    histogram into the specified number of classes. Useful for capturing
    a data-driven threshold at a specific point in time.

    :param image: 2D numpy array
    :param num_classes: Number of classes for multi-Otsu (default: 2)
    :return: Array of threshold values (length = num_classes - 1)
    """
    try:
        return filters.threshold_multiotsu(image, classes=num_classes)
    except ValueError:
        # skimage raises when the frame has fewer distinct levels than classes
        # (a near-uniform / fully-milled ROI). Fall back to the image mean as a
        # single threshold so the monitoring loop survives the degenerate frame
        # instead of aborting (a dead monitor fails to stop milling at criteria).
        fallback = float(np.mean(image))
        logger.warning(
            "multi-Otsu failed on a near-uniform frame; falling back to mean "
            "threshold (%.6f)", fallback
        )
        return np.full(num_classes - 1, fallback, dtype=float)


def frozen_threshold_for_boundary(image, num_classes, boundary_index):
    """
    Pick a single multi-Otsu threshold value by class-boundary index.

    Multi-Otsu returns ``num_classes - 1`` ascending boundary values. This selects
    one of them to use as a frozen binary cutoff (``image >= value``). The index
    chooses how inclusive the binary is:

    - ``-1`` : boundary of the brightest class only (most selective; original behavior)
    - ``0``  : lowest boundary (most inclusive; keeps faint mid-grey features)
    - middle : a balance between the two

    The index is clamped into range so a degenerate (near-uniform) frame, where the
    multi-Otsu fallback returns fewer/identical values, can't raise.

    :param image: 2D numpy array
    :param num_classes: Number of multi-Otsu classes (>= 2)
    :param boundary_index: Which boundary to select (supports negative indexing)
    :return: Selected threshold value as float
    """
    thresholds = calculate_otsu_threshold_values(image, num_classes=num_classes)
    n = len(thresholds)
    idx = boundary_index if -n <= boundary_index < n else -1
    return float(thresholds[idx])


def threshold_fixed(image, threshold_value):
    """
    Apply fixed threshold to create a binary image.

    Pixels with values >= threshold_value become True (white),
    pixels below become False (black).

    This is used after the "delay" phase when we have a calculated reference
    mean pixel value to use as the threshold.

    :param image: 2D numpy array
    :param threshold_value: Threshold value
    :return: Binary threshold image (boolean array)
    """
    threshold_image = image >= threshold_value
    return threshold_image


def threshold_adaptive(image, delay_active, mean_pixel_at_delay=None):
    """
    Apply adaptive thresholding based on monitoring state.

    This is the main thresholding method used in the monitoring workflow.
    It automatically selects the appropriate thresholding method based on
    whether delay processing is active:

    - When delay is active: Uses multi-Otsu (automatic threshold detection)
    - When delay is off: Uses fixed threshold based on reference mean value

    :param image: 2D numpy array
    :param delay_active: Boolean indicating if delay processing is active
    :param mean_pixel_at_delay: Mean pixel value when delay was deactivated
                                (required if delay_active=False)
    :return: Binary threshold image (boolean array)
    """
    if delay_active:
        return threshold_multiotsu(image, num_classes=2)
    else:
        if mean_pixel_at_delay is None:
            raise ValueError("mean_pixel_at_delay must be provided when delay is not active")
        return threshold_fixed(image, mean_pixel_at_delay)


# =======================
# Top-hat foreground isolation (background-level agnostic)
# =======================
#
# A morphological white top-hat estimates the local background by an opening with a
# disk structuring element and subtracts it, so only bright features SMALLER than the
# disk survive. This isolates the bright, sharply-textured foreground regardless of the
# absolute background level or slowly-varying static detail (e.g. a grid bar), which a
# single global threshold cannot do. Used as a CONTINUOUS energy metric (mean residual)
# rather than a binary count, so it has no threshold to freeze or drift.


def white_tophat_map(image, radius=5):
    """
    White top-hat foreground map: bright, sharply-textured features survive; the local
    and slowly-varying static background is removed.

    Compute this on the FULL (uncropped) frame so the disk has real surrounding context
    and there is no crop-boundary artifact in the local-background estimate; crop the
    RESULT afterwards if a region-scoped metric is wanted.

    :param image: 2D numpy array (raw detector counts)
    :param radius: disk structuring-element radius = the maximum foreground feature scale
    :return: top-hat residual image (same dtype/shape as input)
    """
    return morphology.white_tophat(np.asarray(image), morphology.disk(int(radius)))


def tophat_energy(tophat_map, full_scale=255.0):
    """
    Continuous foreground energy = mean top-hat residual as a percentage of full-scale.

    Decays toward ~0 as the foreground is milled away, on a scale that is consistent
    across very different background levels (so one completion threshold serves all).

    :param tophat_map: output of white_tophat_map (for a region, crop it first)
    :param full_scale: detector full-scale used to normalize (raw counts)
    :return: float energy in [0, 100]
    """
    fs = float(full_scale) if full_scale else 255.0
    return float(np.asarray(tophat_map).mean() / fs * 100.0)


def tophat_display(tophat_map):
    """Rescale a top-hat residual map to a uint8 image for display / saving."""
    return exposure.rescale_intensity(tophat_map, out_range=(0, 255)).astype(np.uint8)


def tophat_normalized(tophat_map, full_scale=_RAW_FULL_SCALE):
    """
    Map a top-hat residual map to a float image in [0.0, 1.0] for template
    matching (calculate_match_score expects [0, 1] input), at FIXED absolute
    scale: counts / full_scale, clipped.

    Fixed scale is load-bearing: identical scenes must produce identical match
    images. The former per-frame rescale_intensity stretched whatever range each
    frame happened to have, so on a nearly-milled crop (4-50 pixels of 1-3 count
    residual noise -- the 2026-08-17 on-tool regime) consecutive batches became
    independently amplified, uncorrelated noise fields and the match score hung
    at/near 1.0 instead of settling, blocking completion. A ~3-count sparse gate
    (2026-08-03) had already been added for the <= 3 pixel version of the same
    amplifier; the fixed scale removes the amplifier itself.

    Matching loses nothing: TM_SQDIFF_NORMED is invariant to a multiplicative
    scaling applied to BOTH frames together (a gain change between the reference
    and current frame is NOT cancelled -- SQDIFF_NORMED(a, k*a) = (1-k)^2/k --
    which calculate_match_score handles separately by estimating and dividing
    out the inter-frame gain). Confirmed by replaying the 2026-08-17 field
    campaign, where milling-active scores are essentially unchanged while the
    noise regimes drop to ~0. Real residual counts (measured faintest texture:
    11-24 counts) survive above the matcher's SIGNIFICANT_PIXEL_LEVEL; noise
    (measured ceiling: 2-4 counts, even single-frame) stays below it.

    :param tophat_map: output of white_tophat_map (crop it to the region first)
    :param full_scale: detector full-scale in raw counts (255 for 8-bit; values
        above it clip -- acceptable for a change metric, revisit for >8-bit
        detectors together with tophat_energy's matching assumption)
    :return: float32 2D array in [0.0, 1.0]
    """
    arr = np.asarray(tophat_map, dtype=np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    if peak > float(full_scale):
        # One-time alert: values above full scale clip to 1.0, making the
        # matcher blind to change in clipped regions (>8-bit detector?).
        global _tophat_clip_warned
        if not _tophat_clip_warned:
            _tophat_clip_warned = True
            logger.warning(
                "tophat_normalized: input peak %.1f exceeds full_scale %.1f - "
                "clipping at 1.0; match scoring is blind to change in clipped "
                "regions (is this a >8-bit detector?)", peak, float(full_scale)
            )
    return np.clip(arr / float(full_scale), 0.0, 1.0).astype(np.float32)