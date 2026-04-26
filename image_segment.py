import numpy as np
import cv2 as cv
from typing import Tuple, Dict
from numpy.typing import ArrayLike
from splitmerge import split_and_merge_segmentation
import dlib as dl
import os

def SegmentHand(Img: ArrayLike) -> ArrayLike:
    hand_mask = split_and_merge_segmentation(Img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (17, 17))
    hand_mask = cv.morphologyEx(hand_mask, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    hand_mask = cv.morphologyEx(hand_mask, cv.MORPH_OPEN, kernel)
    return hand_mask


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
    detector = dl.get_frontal_face_detector()
    predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")
    kernel = np.array([[0,-1,0],
                      [-1,5,-1],
                    [0,-1,0]])
    Img = cv.filter2D(Img, -1, kernel)
    img_gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    faces = detector(img_gray, 2)
    Faces = []
    Eyes = []
    os.makedirs(OutPath, exist_ok=True)
    for i,face in enumerate(faces):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = Img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (128, 128))
        landmarks = predictor(img_gray, face)
        scale_x = 128 / w
        scale_y = 128 / h
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)]
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42,48)]
        for (ex, ey) in left_eye:
            ex_rel = int((ex - x) * scale_x)
            ey_rel = int((ey - y) * scale_y)
            cv.circle(face_img, (ex_rel, ey_rel), 2, (0, 255, 0), 1)
        for (ex, ey) in right_eye:
            ex_rel = int((ex - x) * scale_x)
            ey_rel = int((ey - y) * scale_y)
            cv.circle(face_img, (ex_rel, ey_rel), 2, (255,0, 0), 1)
        out_file = os.path.join(OutPath, f"{i}.png")
        cv.imwrite(out_file, face_img)
        Faces.append(face_img)
        Eyes.append({"left": left_eye, "right": right_eye})
    return Faces, Eyes
