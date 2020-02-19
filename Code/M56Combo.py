import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import timeit

def preProccess(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    y, x= image.shape[:2]
    if (x >1024 and y > 768):
        y = int(y/(x/1024))
        image =  cv.resize(image, ( 1024, y ))
    return image

def display(image, windowName):
    cv.imshow(windowName,image)

def gausblur(image, k):
    image = cv.GaussianBlur(image, (k, k), 0)
    return image

def grayscale(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return image

def sobelOp(image, depth, k, scale, delta):
    grad_x = cv.Sobel(image, depth, 1, 0, ksize=k, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(image, depth, 0, 1, ksize=k, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def cannyEdge(image, thresh1, thresh2, aperture):
   image  = cv.Canny(image, thresh1, thresh2, apertureSize = aperture )
   return image

def closing(image, k):
    kernel = cv.getStructuringElement( cv.MORPH_RECT, (k,k) )
    image = cv.morphologyEx( image, cv.MORPH_CLOSE, kernel)
    return image

def opening(image, k):
    kernel = cv.getStructuringElement( cv.MORPH_RECT, (k,k) )
    image = cv.morphologyEx( image, cv.MORPH_OPEN, kernel)
    return image

def contourFind(image, dst, n, precision, minArea, colour):
    contours, _ = cv.findContours(image,cv.RETR_TREE,cv.CHAIN_APPROX_TC89_L1)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:n]
    board = None
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > minArea:
            epsilon = precision*cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            cv.drawContours(dst, [approx], 0, colour, 3)
                
            if  len(approx) == 4  :
                board = approx
                cv.drawContours(dst, [approx], 0, colour, 3)
                return board, dst
    return None, dst



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

#Edge detection M5 = M1+4
gray = grayscale(img)
ret, _= cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

sobel = grayscale(sobelOp(noise, cv.CV_16S, 3, 1, 0))
_,sobel = cv.threshold(sobel,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

edge = cannyEdge(noise,(ret*0.7),(ret*1),3)
edge = closing(edge, 3)

comboS = cv.bitwise_and(sobel, edge)
_, comboS = cv.threshold(comboS,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
comboS = closing(comboS, 3)

#Edge detection M6 = M3 + 5
hsv = cv.cvtColor( img,cv.COLOR_BGR2HSV)
lower =  np.asarray([85, 1, 1]) # double hue opencv
upper = np.asarray([135, 190, 200])
mask = cv.inRange(hsv, lower, upper)
mask = closing(mask, 9)
comboS2 =  cv.bitwise_or(comboS, mask)
img2 = img.copy()


#Contour finding
boardPts, img = contourFind(comboS,img, 1, 0.025, 15000, (0,255,0))
boardPts2, img2 = contourFind(comboS2,img2, 3, 0.025, 15000, (0, 0,255))

#Perspective Warp
if not (boardPts is None):
    board = perspective(img,boardPts)
    display(board, "combo 1")

if not (boardPts2 is None):
    board2 = perspective(img2,boardPts2)
    display(board2, "combo 2")

stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in %s"%execution_time)

display(comboS, "combo1")
display(comboS2, "combo2")
display(img, "combo  1") 
display(img2, "combo  2") 
  
cv.waitKey(0)
cv.destroyAllWindows()