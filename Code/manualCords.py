import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def preProccess(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    y, x= image.shape[:2]
    if (x >1024 and y > 768):
        y = int(y/(x/1024))
        image =  cv.resize(image, ( 1024, y ))
    return image

def display(image, windowName):
    cv.imshow(windowName,image)

def cord (image, x, y,xx,yy,xxx,yyy,xxxx,yyyy):
    cv.circle(image, (x, y), 3, (255,0,0), -1)
    cv.circle(image, (xx, yy), 3, (0,255,0), -1)
    cv.circle(image, (xxx, yyy), 3, (0,0,255), -1)
    cv.circle(image, (xxxx, yyyy), 3, (0,255,255), -1)


#Pre-proccesing
folder = "C://Users//Owen Pantry//project//pictures//Boards//"
imageN = "Europe//table//wood//trains//IMG_2103.jpg"
img = preProccess(folder+imageN)

cord(img, 216,74,915,168,940,669,56,538)


display(img, "image") 
cv.waitKey(0)
cv.destroyAllWindows()




