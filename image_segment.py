import numpy as np
import cv2 as cv
from typing import Tuple, Dict
from numpy.typing import ArrayLike

def SegmentHand(Img: ArrayLike) -> ArrayLike:
    """
    Parameters:
        Img: is the original hand image on any background

    Returns:
        A NumPy array of the binary image that contains the hand mask only
    """
    pass


def DetectFaces(Img, OutPath) -> Tuple[Dict, Dict]:
    """
    Parameters:
        Img: is the original colored image with one or more faces
        OutPath: folder path to save ALL output faces with marked eyes on it.

    Returns:
        tuple containing 
            - Faces: returned struct containing all detected faces after resizing them to 
                        the same size AND marking each of the left eye and right eye
            - Eyes: returned struct containing all resized faces after detecting and marking each of the left eye and right eye
    """
    pass
