import cv2,math
import numpy as np

##resize imgage
def resize(img, ratio):
    nwidth = int(img.shape[1] * ratio)
    nheight = int(img.shape[0] * ratio)
    resized = cv2.resize(img, (nwidth, nheight), interpolation=cv2.INTER_AREA)
    return resized

##create w*h PNG file
def createPng(h,w):
    img_height, img_width = h,w
    n_channels = 4
    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
    return transparent_img

#flag = using external mask. If no ex-mask, put fg in mask instead becuz some stupid bug in numpy
def mergePng(bg,fg,mask,flag=False):
    if flag == False:
        fgMask = fg[:, :, 3:]
        img = fg[:, :, :-1]
    else:
        fgMask = mask
        img = fg
    bgMask = 255 - fgMask
    imgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
    bgMask = cv2.cvtColor(bgMask, cv2.COLOR_GRAY2BGR)
    bgNew =  (bg * (1 / 255.0)) * (bgMask * (1 / 255.0))
    imgNew = (img * (1 / 255.0)) * (imgMask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(bgNew, 255.0, imgNew, 255.0, 0.0))

##crop img, x y and width height
def cropImg(img,x,y,w,h):
    return img[y:y+h, x:x+w]

##p0,nose joint...p1,neck joint..
##pix pixel size of fg scaling distance
def getRatio(bg,p0,p1,pix):
    """
    :type p0: jointObject
    :type p1: jointObject
    :rtype: float
    """
    w = bg.shape[1]
    h = bg.shape[0]
    x1 = w*p0.x
    x2 = w*p1.x
    y1 = h*p0.y
    y2 = h*p1.y
    dist = math.hypot(x2 - x1, y2 - y1)
    ratio = dist/pix
    return ratio

##get translation x and y
def getTran(ratio,bg,p0,p1,fgCenterPoints):
    arY = bg.shape[0]*p1.y-(fgCenterPoints[0]*ratio)
    arX = bg.shape[1]*p1.x-(fgCenterPoints[1]*ratio)
    return (int(arX),int(arY))


def mergeWithAnchor(fg, bg, arY, arX,mask):
    tx = ty = 0
    foreground, background = fg.copy(), bg.copy()
    bgH = background.shape[0]
    bgW = background.shape[1]
    fgH = foreground.shape[0]
    fgW = foreground.shape[1]
    ##crop out the fg image if the fg is out bound the bg.
    if fgH+arY > bgH:
        foreground = cropImg(foreground,0,0,fgW,fgH-(fgH+arY-bgH))
        mask = cropImg(mask,0,0,fgW,fgH-(fgH+arY-bgH))
    if fgW+arX > bgW:
        foreground = cropImg(foreground,0,0,fgW-(fgW+arX-bgW),fgH)
        mask = cropImg(mask,0,0,fgW-(fgW+arX-bgW),fgH)
    if arY <0 :
        foreground = cropImg(foreground,0,abs(arY),fgW,fgH)
        mask = cropImg(mask,0,abs(arY),fgW,fgH)
        ty = arY
        arY = 0
    if arX <0 :
        foreground = cropImg(foreground,abs(arX),0,fgW,fgH)
        mask = cropImg(mask,abs(arX),0,fgW,fgH)
        tx = arX
        arX = 0
    ##image indexing
    start_y = arY
    start_x = arX
    end_y = arY+fgH+ty
    end_x = arX+fgW+tx
    cv2.waitKey(0)
    cv2.waitKey(0)
    blended_portion = mergePng(background[start_y:end_y, start_x:end_x,:],foreground,mask,flag=True)
    """
    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    """
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    return background

##feature out the mask, with n pixels, add gaussian blur at the end
def feather(image,n):
    def check(i,j):
        res = []
        for k in range(4):
            ni = i + dirX[k]
            nj = j + dirY[k]
            if (ni,nj) not in visited and ni>= 0 and nj >= 0 and ni < len(image) and nj < len(image[0]) and image[ni][nj][0] == 0:
                res.append((ni,nj))
        return res
    subted = 255/n
    color = 255 - subted
    dirX = [0,0,1,-1]
    dirY = [1,-1,0,0]
    queue = []
    visited = set()
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j][0] == 255:
                queue += check(i,j)
    while color > 0:
        q2 = []
        for item in queue:
            image[item[0]][item[1]] = np.uint8(color)
            res = check(item[0],item[1])
            q2 += res
            visited |= set(res)
        color -= subted
        queue = q2
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur