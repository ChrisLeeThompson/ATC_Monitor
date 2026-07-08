"""
Image Cropping Module

This module provides cropping utilities for RTM images.
"""

import numpy as np
from PySide6.QtCore import QRect


def crop_to_rect(image, crop_rect):
    """
    Crop an image to a specified rectangle.
    
    This is used to apply the user-defined crop rectangle from the RTM plot widget.
    The rectangle defines the region of interest for analysis.
    
    :param image: 2D numpy array
    :param crop_rect: QRect or tuple (x, y, width, height) defining crop region
    :return: Cropped 2D numpy array
    """
    # Handle QRect or tuple
    if isinstance(crop_rect, QRect):
        x = crop_rect.x()
        y = crop_rect.y()
        width = crop_rect.width()
        height = crop_rect.height()
    else:
        x, y, width, height = crop_rect
    
    # Convert to integers and ensure valid bounds
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)
    
    # Get image dimensions
    image_height, image_width = image.shape
    
    # Clamp coordinates to image bounds
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    
    # Clamp dimensions to stay within image
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    
    # Enforce minimum 1px in each dimension to prevent empty arrays
    width = max(1, width)
    height = max(1, height)
    
    # Crop the image
    cropped = image[y:y + height, x:x + width]
    
    return cropped