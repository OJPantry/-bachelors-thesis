import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
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

def grayscale(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return image


def LapOp(image, depth, k):
    image = cv.Laplacian(image,depth,k)
    image = cv.convertScaleAbs(image)
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



#Edge detection
lap = grayscale(LapOp(noise, cv.CV_16S, 9))
_, lapT = cv.threshold(lap,30,255,cv.THRESH_TOZERO)
lapT = closing(lapT, 3)



#contour finding
boardPts, img = contourFind(lapT,img, 3, 0.025, 15000, (0, 0,255))

#perspective warp
if not (boardPts is None):
    board = perspective(img,boardPts)
    display(board, "warp")

display(lapT, "lap Torezo")
display(img, "image lap") 
  

stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in %s"%execution_time)

cv.waitKey(0)
cv.destroyAllWindows()

