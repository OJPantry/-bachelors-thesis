import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
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

def getBoard(img, pts):
    (tl, tr, br, bl) = pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped
#resizes input image
def resize(img, w):
    dim= (w, 678)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

# returns dominant colours through k means       
def dominantColors(image, clusters=3):
	img = cv.imread(image)
	y, x= img.shape[:2]
	r = x/y
	x = int(200*r)
	y = int(200*r)
	img =  cv.resize(img, ( x, y ))
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	kmeans = KMeans(n_clusters = clusters)
	kmeans.fit(img)	
	colours = kmeans.cluster_centers_
	labels = kmeans.labels_
	return colours.astype(int)	

#returns expected colour range of a given tile        
def checkLocC(n):
    if (locCols[n] == 'lg'):
        colL = lg[0]
        colH = lg[1]
    elif (locCols[n] == 'y'):
        colL = y[0]
        colH = y[1]
    elif (locCols[n] == 'g'):
        colL =g[0]
        colH = g[1]
    elif (locCols[n] == 'b'):
        colL = b[0]
        colH = b[1]
    elif (locCols[n] == 'r'):
        colL = r[0]
        colH = r[1]
    else:
        colL = w[0]
        colH = w[1]
    return colL, colH

#returns wether game piece on a game given game tile
def checkLoc(n, board):
    r,g,b =tileColour(board,locs[n])
    colL, colH = checkLocC(n)
    h,s,v = rgb2hsv(r,g,b)
    if h >= colL[0] and h<= colH[0]:
        if s >= colL[1] and s<= colH[1]:
            if v >= colL[2] and v<= colH[2]:
                return False
            else:
                return True
        else:
            return True
    else:
        return True

#COnverts RGB to HSV values
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

#returns dominant colour of tile at provided ROI
def tileColour(image, polygon):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros((height, width))
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.fillPoly(mask, [polygon], (255))
    dst = cv.bitwise_and(image,image,mask = mask)
    bg = ~mask
    bg= cv.bitwise_not(dst,dst,mask = bg)

    rect = cv.boundingRect(polygon) # returns (x,y,w,h) of the rect
    crop = bg[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    cv.imwrite("dst.jpg",crop)
    colors = dominantColors('dst.jpg', 2)

    if np.sum(colors[0]) >=  np.sum(colors[1]):
        colors = colors[1]
    else:
        colors = colors[0]

    return colors

def selectBoard(n):
    if n ==96:
        bpts = np.array([[173,39],[960,41],[1023,572],[134,586]], dtype="float32")#96
        trains = np.array([3,6])
        imageN = "trains//IMG_2096.jpg"
        return bpts, trains, (imageN)
    elif n ==97:
        bpts = np.array([[227,60],[893,186],[851,678],[44,479]], dtype="float32")#97
        trains = np.array([3,6,10,11])
        imageN = "trains//IMG_2097.jpg"
        return bpts, trains, (imageN)
    elif n ==95:
        bpts = np.array([[182,140],[865,136],[966,603],[109,623]], dtype="float32")#97
        trains = np.array([])
        imageN = "trains//IMG_2095.jpg"
        return bpts, trains, (imageN)
    elif n ==94:
        bpts = np.array([[202,209],[837,188],[966,620],[140,671]], dtype="float32")#97
        trains = np.array([])
        imageN = "trains//IMG_2094.jpg"
        return bpts, trains, (imageN)
    elif n ==93:
        bpts = np.array([[507,139],[952,413],[516,721],[190,255]], dtype="float32")#9
        trains = np.array([])
        imageN = "IMG_1993.jpg"
        return bpts, trains, (imageN)
    elif n ==92:
        bpts = np.array([[96,265],[668,79],[961,320],[234,691]], dtype="float32")#9
        trains = np.array([])
        imageN = "IMG_1994.jpg"
        return bpts, trains, (imageN)
    elif n ==91:
        bpts = np.array([[162,152],[857,123],[1004,581],[107,656]], dtype="float32")#9
        trains = np.array([])
        imageN = "IMG_1995.jpg"
        return bpts, trains, (imageN)
    elif n ==90:
        bpts = np.array([[170,237],[741,143],[952,458],[204,660]], dtype="float32")#9
        trains = np.array([])
        imageN = "IMG_1996.jpg"
        return bpts, trains, (imageN)
    elif n ==98:
        bpts = np.array([[150,90],[876,129],[941,634],[33,592]], dtype="float32")#98
        trains = np.array([3,6,10,11,5])
        imageN = "trains//IMG_2098.jpg"
        return bpts, trains, (imageN)
    elif n ==99:
        bpts = np.array([[110,77],[876,83],[970,605],[29,622]], dtype="float32")#99
        trains = np.array([1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        imageN = "trains//IMG_2099.jpg"
        return bpts, trains, (imageN)
    elif n ==100:
        bpts = np.array([[161,106],[855,113],[969,577],[56, 584]], dtype="float32")#100
        trains = np.array([1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        imageN = "trains//IMG_2100.jpg"
        return bpts, trains, (imageN)
    elif n ==101:
        bpts = np.array([[305,66],[925,218],[878,674],[81,411]], dtype="float32")#101
        trains = np.array([1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        imageN = "trains//IMG_2101.jpg"
        return bpts, trains, (imageN)
    elif n ==102:
        bpts = np.array([[88,182],[781,44],[990,467],[141,704]], dtype="float32")#102
        trains = np.array([1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        imageN = "trains//IMG_2102.jpg"
        return bpts, trains, (imageN)
    elif n ==103:
        bpts = np.array([[216,74],[915,168],[940,669],[56,538]], dtype="float32")#103
        trains = np.array([2,4,5,9,11,14,16,20,21,26])
        imageN = "trains//IMG_2103.jpg"
        return bpts, trains, (imageN)

# runs tests on a given image of a game board
def checkBoard(n):
    folder = "C://Users//Owen Pantry//project//pictures//Boards//Europe//table//wood//"
    bpts, trains, imagefile = selectBoard(n)
    img = preProccess(folder+imagefile)
    board = getBoard(img, bpts)
    board = resize(board, 1000) 
    outputimg = board.copy()
    tcount = 0
    empty = 0
    missed = 0
    extra = 0
    for i in range(0,26):
        present= checkLoc(i,board)
        
        if ((i+1) in trains) == True:
            if present == True:
                cv.drawContours(outputimg, [locs[i]], 0, (255,255,0), 2)
                tcount = tcount+1
            else:
                missed = missed +1 
        else:
            if present == True:
                cv.drawContours(outputimg, [locs[i]], 0, (255,255,0), 2)
                extra = extra +1
            else:
                empty = empty +1

    #print("Train Count",tcount,"empty locs",empty,"Missed", missed, "Extra",extra, ".")
    #cv.imshow(str(n),outputimg)

#colour ranges
lg = np.asarray([[0,0,40],[360,25,73]])
w =  np.asarray([[0,0,53],[37,24,74]])
g =  np.asarray([[63,30,50],[93,45,75]])
b =  np.asarray([[190,20,55],[208,70,80]])
r =  np.asarray([[0,30,56],[35,58,75]])
y = np.asarray([[38,35,65],[60,55, 78.5]])

#expected colours of tiles
locCols = ['lg','lg','y','y','w','b','lg','lg','r','b','r','r','r','lg','lg','r','r','r','g','g','b','b','b','w','w','lg']

#ROIs for board tiles
locs = np.array([[[212,214],[225,215],[222,248],[210,246]]], np.int32)#1
pts = np.array([[[210,246],[222,248],[218,285],[205,284]]], np.int32)#2
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[272,290],[283,295],[268,327],[257,320]]], np.int32)#3
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[287,258],[298,264],[283,296],[273,290]]], np.int32)#4
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[312,317],[342,300],[347,313],[318,328]]], np.int32)#5
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[227,408],[240,413],[226,447],[214,440]]], np.int32)#6
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[202,496],[213,495],[218,525],[205,526]]], np.int32)#7
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[205,526],[217,526],[222,560],[208,561]]], np.int32)#8
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[558,340],[582,357],[576,368],[549,352]]], np.int32)#9
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[574,311],[598,290],[605,299],[580,322]]], np.int32)#10
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[650,215],[663,185],[674,193],[658,222]]], np.int32)#11
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[672,178],[704,172],[706,184],[674,189]]], np.int32)#12
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[716,172],[740,192],[733,202],[709,184]]], np.int32)#13
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[762,199],[790,217],[782,227],[756,210]]], np.int32)#14
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[783,230],[794,228],[793,262],[782,262]]], np.int32)#15
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[798,266],[831,269],[830,283],[797,283]]], np.int32)#16
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[831,269],[853,244],[862,254],[841,278]]], np.int32)#17
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[843,211],[857,209],[865,240],[854,243]]], np.int32)#18
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[926,320],[961,320],[960,332],[924,331]]], np.int32)#19
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[948,332],[959,332],[960,366],[948,366]]], np.int32)#20
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[686,496],[714,513],[706,523],[679,505]]], np.int32)#21
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[715,512],[745,530],[738,539 ],[709,522]]], np.int32)#22
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[745,527],[775,545],[768,555],[741,540]]], np.int32)#23
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[278,329],[311,320],[318,331],[282,341]]], np.int32)#24
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[341,299],[369,282],[376,291],[348,311]]], np.int32)#25
locs = np.append(locs, pts, axis = 0)
pts = np.array([[[268,397],[276,388],[295,417],[284,424]]], np.int32)#26
locs = np.append(locs, pts, axis = 0)


#Running tests on all test images

meanTime = 0
for x in range(90, 104):
    start = timeit.default_timer()
    print("Board :",x, end=" ")
    checkBoard(x)
    stop = timeit.default_timer()
    meanTime = meanTime + (stop - start)
meanTime = meanTime / 14


cv.waitKey(0)
cv.destroyAllWindows()