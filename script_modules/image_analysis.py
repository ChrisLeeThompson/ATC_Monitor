"""
Image Analysis Module

This module provides analysis methods for RTM images.
"""

import numpy as np


def calculate_mean_pixel_value(image):
    """
    Calculate the mean pixel value of an image.
    
    This metric is used to track the overall brightness/darkness of RTM images
    over time. Typically used on cropped regions of interest.
    
    :param image: 2D numpy array
    :return: Mean pixel value as float
    """
    mean_value = np.mean(image)
    return mean_value


def calculate_slope_of_mean_pixel_values(mean_pixel_list, num_points=4):
    """
    Calculate the slope (first derivative) of mean pixel values using gradient method.
    
    :param mean_pixel_list: List of mean pixel values over time
    :param num_points: Number of most recent gradient points to average (default=4)
    :return: Average slope as float, or None if insufficient data
    """
    if len(mean_pixel_list) < num_points:
        return None
    
    # Calculate gradient (point-to-point differences)
    gradients = np.gradient(mean_pixel_list[-num_points:])
    
    # Average the gradients
    avg_slope = np.mean(gradients)
    
    return round(float(avg_slope), 4)


def calculate_slope_linear_regression(mean_pixel_list, fit_number=3):
    """
    Calculate the slope of mean pixel values using linear regression.
    
    :param mean_pixel_list: List of mean pixel values over time
    :param fit_number: Number of most recent points to fit (default=3)
    :return: Slope as float, or None if insufficient data
    """
    if len(mean_pixel_list) < fit_number:
        return None
    
    # Get the last 'fit_number' of points
    recent_values = mean_pixel_list[-fit_number:]
    
    # Create x values (time indices)
    x = np.arange(len(recent_values))
    y = np.array(recent_values)
    
    # Perform linear regression
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    
    return slope


def calculate_num_white_pixels(binary_image, image_height=None, image_width=None, return_percentage=False):
    """
    Calculate the number (or percentage) of white pixels in a binary image.
    
    :param binary_image: 2D numpy array (binary or grayscale)
    :param image_height: Optional, height for percentage calculation (uses image shape if None)
    :param image_width: Optional, width for percentage calculation (uses image shape if None)
    :param return_percentage: If True, returns percentage instead of count
    :return: Number of white pixels (int) or percentage (float) if return_percentage=True
    """
    # Count white pixels (assuming white = 255 or True)
    # Handle both boolean arrays and uint8 with value 255
    if binary_image.dtype == bool:
        num_white_pixels = np.sum(binary_image)
    else:
        # For grayscale, count pixels above threshold (typically 127 for binary images)
        num_white_pixels = np.sum(binary_image > 127)
    
    if return_percentage:
        # Get image dimensions
        if image_height is None or image_width is None:
            img_height, img_width = binary_image.shape
        else:
            img_height, img_width = image_height, image_width
        
        total_pixels = img_height * img_width
        percentage = (num_white_pixels / total_pixels) * 100.0
        return percentage
    
    return int(num_white_pixels)