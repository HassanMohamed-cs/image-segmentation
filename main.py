import matplotlib.pyplot as plt
import cv2 as cv
from image_segment import SegmentHand, DetectFaces
def Problem_1():
    case_choice = int(input("Test Case #: "))
    input_image = cv.imread(f"../../Imgs/Prob1/{case_choice}.jpg")
    output_image = SegmentHand(input_image)
    input_image_rgb = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(input_image_rgb)
    ax[1].imshow(output_image, cmap='gray')
    plt.show()

def ApplyAll(inputPaths_File, outputPath):
    pass

def Problem_2():
    case_choice = int(input("Test Case #: "))
    image = cv.imread(f"../../Imgs/Prob2/{case_choice}.jpg")
    Faces, Eyes = DetectFaces(image, "../../Imgs/Prob2/Sample/")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print(Eyes)
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, len(Faces) + 1, 1)
    plt.imshow(image)
    plt.axis('off')
    for i, face in enumerate(Faces):
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        plt.subplot(1, len(Faces) + 1, i + 2)
        plt.imshow(face)
        plt.axis('off')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Right Eye',
                           markerfacecolor='blue', markersize=8)
    green_patch = plt.Line2D([0], [0], marker='o', color='w', label='Left Eye',
                             markerfacecolor='green', markersize=8)
    fig.legend(handles=[red_patch, green_patch],
               loc='lower center', ncol=2)
    plt.show()


Problem_1()
Problem_2()