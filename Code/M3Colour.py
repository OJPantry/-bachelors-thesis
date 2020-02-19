import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import timeit

def closing(image, k):
    kernel = cv.getStructuringElement( cv.MORPH_RECT, (k,k) )
    image = cv.morphologyEx( image, cv.MORPH_CLOSE, kernel)
    return image

def opening(image, k):
    kernel = cv.getStructuringElement( cv.MORPH_RECT, (k,k) )
    image = cv.morphologyEx( image, cv.MORPH_OPEN, kernel)
    return image
def preProccess(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    y, x= image.shape[:2]
    if (x >1024 and y > 768):
        y = int(y/(x/1024))
        image =  cv.resize(image, ( 1024, y ))
    return image

def perspective(img, pts):
    rect = np.zeros((4, 2), dtype="float32")
    xy = sorted(pts[:,0,:], key=lambda row: row[0])
    L = xy[:2]
    R = xy[2:]
    rect[0], rect[3] = sorted(L, key=lambda row: row[1])
    rect[1],rect[2] = sorted(R, key=lambda row: row[1])

    (tl, tr, br, bl) = rect.astype(int)
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

start = timeit.default_timer()

#Pre-proccessing
folder = "C://Users//Owen Pantry//project//pictures//Boards//"
imageN = "Europe//table//wood//Board.jpg"
img = preProccess(folder+imageN)
noise = cv.fastNlMeansDenoising(img, None,11, 15)#NL-means algorithm
hsv = cv.cvtColor( img,cv.COLOR_BGR2HSV)


#creating a mask
lower =  np.asarray([85, 1, 1]) # double hue opencv
upper = np.asarray([135, 190, 200])
mask = cv.inRange(hsv, lower, upper)
mask = closing(mask, 13)

#applying mask to input image
res = cv.bitwise_and(img, img, mask= mask)

#Contour finding
contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]

for cnt in contours:
    epsilon = 0.04*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)    
    cv.drawContours(img, [approx], 0, (0,255,0), 3)
    if  len(approx) == 4  :
        warped = perspective(img,approx)   #perspective transform
        cv.imshow('line ',warped) 



stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in %s"%execution_time)

cv.imshow('image',img)
cv.imshow('mask',mask)
cv.imshow('res',res)

cv.waitKey(0)
cv.destroyAllWindows()