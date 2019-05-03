import cv2,math
import numpy as np
bg = cv2.imread('demo/2.jpg',-1)
fg = cv2.imread('demo/kqq.png',-1)
st = cv2.imread('demo/kq.jpg')
stp = cv2.imread('demo/kqq.png',-1)

def createPng(h,w):
    img_height, img_width = h,w
    n_channels = 4
    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
    # Save the image for visualization
    return transparent_img

def mergePng(bg,img):
    imgMask = img[:,:,3:]
    img = img[:,:,:-1]
    bgMask = 255 - imgMask
    imgMask = cv2.cvtColor(imgMask, cv2.COLOR_GRAY2BGR)
    bgMask = cv2.cvtColor(bgMask, cv2.COLOR_GRAY2BGR)
    bgNew =  (bg * (1 / 255.0)) * (bgMask * (1 / 255.0))
    imgNew = (img * (1 / 255.0)) * (imgMask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(bgNew, 255.0, imgNew, 255.0, 0.0))

def cropImg(img,x,y,w,h):
    return img[y:y+h, x:x+w]


def cropImg2(w,h,p0,p1):
    x1 = w*p0[0]
    x2 = w*p1[0]
    y1 = h*p0[1]
    y2 = h*p1[1]
    print (x1,y1,x2,y2)
    dist = math.hypot(x2 - x1, y2 - y1)
    ratio = dist/100
    return ratio


def combine_two_color_images_with_anchor(img1, img2, arY, arX):
    tx = ty = 0
    foreground, background = img1.copy(), img2.copy()
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    bgH = background.shape[0]
    bgW = background.shape[1]
    fgH = foreground.shape[0]
    fgW = foreground.shape[1]
    if fgH+arY > bgH:
        foreground = cropImg(foreground,0,0,fgW,fgH-(fgH+arY-bgH))
    if fgW+arX > bgW:
        foreground = cropImg(foreground,0,0,fgW-(fgW+arX-bgW),fgH)
    if arY <0 :
        foreground = cropImg(foreground,0,abs(arY),fgW,fgH)
        ty = arY
        arY = 0
    if arX <0 :
        foreground = cropImg(foreground,abs(arX),0,fgW,fgH)
        tx = arX
        arX = 0
    cv2.imshow('composited image', foreground)
    cv2.waitKey(0)
    # do composite at specified location
    start_y = arY
    start_x = arX
    end_y = arY+fgH+ty
    end_x = arX+fgW+tx
    cv2.waitKey(0)
    cv2.waitKey(0)
    blended_portion = mergePng(background[start_y:end_y, start_x:end_x,:],foreground)
    cv2.imshow('composited image', background)
    """
    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    """
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    cv2.imshow('composited image', background)
    cv2.waitKey(0)

combine_two_color_images_with_anchor(fg,bg,100,300)




