import cv2 as cv
import numpy as np
from skimage.segmentation import quickshift

def split_and_merge_segmentation(image):
    """Fixed segmentation implementation"""
    segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    
    hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for seg_val in np.unique(segments):
        segment_mask = (segments == seg_val).astype(np.uint8)
        
        segment_pixels = cv.bitwise_and(image, image, mask=segment_mask)
        segment_pixels = segment_pixels[np.where(segment_mask)]
        
        if len(segment_pixels) > 0:
            segment_pixels = segment_pixels.reshape(-1, 1, 3)

            if region_predicate(segment_pixels):
                hand_mask = cv.bitwise_or(hand_mask, segment_mask)
    contours, _ = cv.findContours(hand_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(hand_mask)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(final_mask, [largest_contour], -1, 255, cv.FILLED)
        return final_mask
    return hand_mask
    
def region_predicate(region_pixels):
    rgb_mask = get_rgb_mask(region_pixels)
    ratio = np.count_nonzero(rgb_mask) / region_pixels.shape[0]
    return ratio > 0.5

def get_rgb_mask(region_pixels):
    B = region_pixels[:, :, 0].astype(np.int16)
    G = region_pixels[:, :, 1].astype(np.int16)
    R = region_pixels[:, :, 2].astype(np.int16)
    cond1 = (R > 95) & (G > 40) & (B > 20)
    cond2 = (R - G > 30) & (R - B > 30)
    cond3 = (R > G) & (R > B)
    rgb_mask = np.where(cond1 & cond2 & cond3, 255, 0).astype(np.uint8)
    return rgb_mask
#Write the main body and function calls here (feel free to add more functions as you like)

