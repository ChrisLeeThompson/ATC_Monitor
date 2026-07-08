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


# =======================
# Core Filtering Workflow
# =======================

def filter_images(image_list, gaussian_sigma=1.0, apply_dilation=True):
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
    (pattern matching, display, thresholding, mean pixel calculation) receive
    data in a consistent, predictable range.

    :param image_list: List of 2D numpy arrays (images from RTM)
    :param gaussian_sigma: Standard deviation for Gaussian kernel (default: 1.0)
    :param apply_dilation: Whether to apply morphological dilation (default: True)
    :return: Filtered 2D numpy array in [0.0, 1.0] range highlighting temporal changes
    """
    if not image_list:
        raise ValueError("image_list cannot be empty")

    last_image = image_list[-1]

    # Create background reference by averaging all images
    background = blend_images(image_list)
    background = apply_gaussian_filter(background, gaussian_sigma)
    if apply_dilation:
        background = morphological_dilation(background)

    # Process the current (last) image
    current = apply_gaussian_filter(last_image, gaussian_sigma)
    if apply_dilation:
        current = morphological_dilation(current)

    # Subtract background and invert to highlight changes
    filtered_image = subtract_images(current, background)
    filtered_image = invert_image(filtered_image)

    # Guard against a near-uniform (e.g. fully-milled / blank) frame: per-frame
    # rescale_intensity below would stretch pure floating-point noise to span
    # [0, 1], producing garbage that corrupts mean-pixel / white-pixel / match
    # metrics and can trigger the multi-Otsu raise. Return a flat zero frame.
    if float(np.ptp(filtered_image)) < _UNIFORM_FRAME_EPS:
        return np.zeros_like(filtered_image)

    # Normalize to [0.0, 1.0] so downstream consumers get a consistent range
    filtered_image = exposure.rescale_intensity(filtered_image, out_range=(0.0, 1.0))

    return filtered_image


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


def tophat_normalized(tophat_map):
    """
    Rescale a top-hat residual map to a float image in [0.0, 1.0] for template
    matching (calculate_match_score expects [0, 1] input).

    Per-region rescale maximizes tile variance so the matcher does not collapse
    foreground tiles into its near-uniform "perfect match" case. Mirrors the
    near-uniform guard in filter_images: a fully-milled / blank region (no real
    foreground) has no dynamic range to stretch, so return a flat zero frame
    rather than amplifying floating-point noise to span [0, 1].

    :param tophat_map: output of white_tophat_map (crop it to the region first)
    :return: float32 2D array in [0.0, 1.0]
    """
    arr = np.asarray(tophat_map, dtype=np.float32)
    if float(np.ptp(arr)) < _UNIFORM_FRAME_EPS:
        return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, out_range=(0.0, 1.0)).astype(np.float32)