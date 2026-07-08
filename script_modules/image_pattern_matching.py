"""
Image Pattern Matching Module

This module provides template matching functions for comparing RTM images over time.

The main function compares regions between a reference and current image using
tiled template matching to detect how much the pattern has changed.

Matching is performed using OpenCV's TM_SQDIFF_NORMED method, which normalizes
scores to [0.0, 1.0] and is insensitive to global brightness drift between
reference and current images.
"""
import logging
import numpy as np
from cv2 import matchTemplate, minMaxLoc, TM_SQDIFF_NORMED
from skimage.util import img_as_ubyte
from math import sqrt


logger = logging.getLogger(__name__)


# Variance threshold for uniform tile detection (uint8 pixel values).
# Tiles where both reference and current have variance below this value
# are considered uniform (e.g., fully milled away) and assigned a perfect
# match score of 0.0, avoiding the numerically unstable SQDIFF_NORMED
# division-by-zero that produces false 1.0 scores.
UNIFORM_TILE_VARIANCE_THRESHOLD = 1.0

# Minimum image dimension (pixels) for meaningful tiled template matching.
# Images smaller than this in either dimension produce tiles too small for
# reliable comparison. Returns 0.0 (no meaningful difference detectable).
MIN_IMAGE_DIMENSION = 5


def calculate_match_score(first_image, last_image, min_splits=5, target_tile_size=100):
    """
    Calculate a pattern match score between two images using tiled template matching.

    This function splits both images into NxN tiles and compares corresponding tiles
    using TM_SQDIFF_NORMED template matching. The maximum match score across all tiles
    is returned, providing sensitivity to localized changes in any region of the image.

    The number of tiles adapts to image size:
    - Larger images get more tiles for finer resolution
    - Smaller images use fewer tiles (minimum of min_splits x min_splits)
    - This prevents over-splitting small crops or under-splitting large crops

    Workflow:
    1. Convert images from [0.0, 1.0] float to uint8 (required by cv2.matchTemplate)
    2. Calculate optimal number of splits based on image area and target tile size
    3. Split images into NxN grid
    4. Compare each tile pair using normalized squared difference matching
    5. Return the maximum match score across all tiles

    Input images are expected in [0.0, 1.0] range (as output by filter_images()),
    so the conversion to uint8 is a direct mapping without any contrast stretching.

    TM_SQDIFF_NORMED normalizes by patch energy, producing values in [0.0, 1.0].
    0.0 = perfect match, 1.0 = maximum difference. Insensitive to global intensity
    shifts between reference and current images.

    Note on uniform tiles:
    Tiles with near-zero variance (e.g., fully milled regions that are mostly black)
    produce numerically unstable results because the normalization denominator
    approaches zero. OpenCV returns 1.0 for these degenerate cases, falsely indicating
    maximum difference between identical uniform tiles. A variance guard detects these
    tiles and assigns them a perfect match score of 0.0.

    :param first_image: 2D numpy array in [0.0, 1.0] range (reference image)
    :param last_image: 2D numpy array in [0.0, 1.0] range (current image)
    :param min_splits: Minimum number of splits per dimension (default: 5)
    :param target_tile_size: Target size for each tile dimension in pixels (default: 100)
    :return: Match score as float (0.0 = perfect match, 1.0 = maximum difference)
    """
    # Guard: images must have identical dimensions for tile-based comparison
    if first_image.shape != last_image.shape:
        logger.warning(
            f"Shape mismatch between reference {first_image.shape} and "
            f"current {last_image.shape} - returning 0.0 (cannot compare)"
        )
        return 0.0

    # Guard: images too small for meaningful tiled comparison
    height, width = first_image.shape[:2]
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        logger.warning(
            f"Image too small for tiled matching ({width}x{height}, "
            f"minimum {MIN_IMAGE_DIMENSION}px per dimension) - returning 0.0"
        )
        return 0.0

    # Convert to uint8 for cv2.matchTemplate
    # Input images are already in [0.0, 1.0] range from filter_images(),
    # so img_as_ubyte maps directly to 0-255 without any contrast stretching.
    last_image_uint8 = img_as_ubyte(last_image)
    first_image_uint8 = img_as_ubyte(first_image)

    # Calculate optimal number of splits based on image area and target tile size
    num_splits = _calculate_num_splits(last_image_uint8.shape, min_splits, target_tile_size)

    # Split images into tiles
    last_tiles = _split_image_into_tiles(last_image_uint8, num_splits)
    first_tiles = _split_image_into_tiles(first_image_uint8, num_splits)

    # Compare all tile pairs and collect match scores
    match_scores = []
    skipped_tiles = 0

    for i in range(num_splits):
        for j in range(num_splits):
            # Get corresponding tiles
            last_tile = last_tiles[i][j]
            first_tile = first_tiles[i][j]

            # Guard: skip empty tiles (can occur if array_split produces
            # zero-dimension slices at edges)
            if last_tile.size == 0 or first_tile.size == 0:
                skipped_tiles += 1
                continue

            # Guard: skip tiles with near-zero variance (uniform regions)
            # where SQDIFF_NORMED normalization is numerically unstable.
            # Two identical uniform tiles are a perfect match (0.0).
            if (np.var(last_tile) < UNIFORM_TILE_VARIANCE_THRESHOLD and
                    np.var(first_tile) < UNIFORM_TILE_VARIANCE_THRESHOLD):
                match_scores.append(0.0)
                skipped_tiles += 1
                continue

            # Perform template matching (0 = perfect match)
            result = matchTemplate(last_tile, first_tile, TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = minMaxLoc(result)

            # Collect the minimum value (best match for this tile)
            match_scores.append(min_val)

    # Return the maximum match score across all tiles
    if not match_scores:
        logger.warning("No valid tiles for comparison - returning 0.0")
        return 0.0

    max_match_score = float(max(match_scores))

    logger.info(f"Match scores ({len(match_scores)} tiles, {skipped_tiles} uniform): {match_scores}")

    return max_match_score


def _calculate_num_splits(image_shape, min_splits=5, target_tile_size=100):
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
    - 100x100 image (10,000 px) -> split_factor=1 -> sqrt(1)=1 -> use min_splits (5)
    - 256x256 image (65,536 px) -> split_factor=6.5 -> sqrt~2.5 -> round to 3 -> use min_splits (5)
    - 512x512 image (262,144 px) -> split_factor=26.2 -> sqrt~5.1 -> round to 5 -> use 5
    - 1024x1024 image (1,048,576 px) -> split_factor=104.9 -> sqrt~10.2 -> round to 10 -> use 10

    :param image_shape: Tuple (height, width) from image.shape
    :param min_splits: Minimum number of splits (default: 5)
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

    logger.info(f"Image width, height: Width: {width}, Height: {height}")

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

    logger.info(f"Number of rows, columns: Rows: {len(rows)}, Columns: {len(cols)}")

    return tiles