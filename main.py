import matplotlib.pyplot as plt
import cv2 as cv
import image_segment
from splitmerge import split_and_merge_segmentation
def Problem_1():
    #case_choice = int(input("Test Case #: "))

    # Open input image based on Case Choice
    input_image = cv.imread("../Imgs/Prob1/Sample/hand.jpg")
    output_image = split_and_merge_segmentation(input_image)
    input_image_rgb = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    ## Plot both images side-by-side
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(input_image_rgb)
    ax[1].imshow(output_image, cmap='gray')
    plt.show()

def ApplyAll(inputPaths_File, outputPath):
    pass

def Problem_2():
    case_choice = int(input("Test Case #: "))

    # Your Code


Problem_1()
#Problem_2()