import cv2,math
import numpy as np

#Y,500 X,200
class frame():
    def __init__(self,p0,p1,fgCenterPoints,bg,fg):
        self.p1 = p1
        self.p0 = p0
        self.bg = bg
        self.fg = fg
        self.fgCenterPoints = fgCenterPoints
    def main(self):
        ratio = self.getRatio()
        ar = self.getTran(ratio)
        img2 = self.resize(ratio)
        arX = ar[1]
        arY = ar[0]
        self.combine_two_color_images_with_anchor(img2, self.bg, arX, arY)


    def resize(self,ratio):
        nwidth = int(self.fg.shape[1]*ratio)
        nheight = int(self.fg.shape[0]*ratio)
        resized = cv2.resize(self.fg, (nwidth,nheight), interpolation=cv2.INTER_AREA)
        return resized

    def createPng(self,h,w):
        img_height, img_width = h,w
        n_channels = 4
        transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
        # Save the image for visualization
        return transparent_img

    def mergePng(self,bg,img):
        imgMask = img[:,:,3:]
        img = img[:,:,:-1]
        bgMask = 255 - imgMask
        imgMask = cv2.cvtColor(imgMask, cv2.COLOR_GRAY2BGR)
        bgMask = cv2.cvtColor(bgMask, cv2.COLOR_GRAY2BGR)
        bgNew =  (bg * (1 / 255.0)) * (bgMask * (1 / 255.0))
        imgNew = (img * (1 / 255.0)) * (imgMask * (1 / 255.0))
        return np.uint8(cv2.addWeighted(bgNew, 255.0, imgNew, 255.0, 0.0))

    def cropImg(slef,img,x,y,w,h):
        return img[y:y+h, x:x+w]

    def getRatio(self):
        w = self.bg.shape[1]
        h = self.bg.shape[0]
        x1 = w*self.p0.x
        x2 = w*self.p1.x
        y1 = h*self.p0.y
        y2 = h*self.p1.y
        dist = math.hypot(x2 - x1, y2 - y1)
        ratio = dist/100
        return ratio

    def getTran(self,ratio):
        arY = self.bg.shape[0]*self.p1.y-(self.fgCenterPoints[0]*ratio)
        arX = self.bg.shape[1]*self.p1.x-(self.fgCenterPoints[1]*ratio)
        return (int(arX),int(arY))

    def combine_two_color_images_with_anchor(self,img1, img2, arY, arX):
        tx = ty = 0
        foreground, background = img1.copy(), img2.copy()
        # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
        bgH = background.shape[0]
        bgW = background.shape[1]
        fgH = foreground.shape[0]
        fgW = foreground.shape[1]
        if fgH+arY > bgH:
            foreground = self.cropImg(foreground,0,0,fgW,fgH-(fgH+arY-bgH))
        if fgW+arX > bgW:
            foreground = self.cropImg(foreground,0,0,fgW-(fgW+arX-bgW),fgH)
        if arY <0 :
            foreground = self.cropImg(foreground,0,abs(arY),fgW,fgH)
            ty = arY
            arY = 0
        if arX <0 :
            foreground = self.cropImg(foreground,abs(arX),0,fgW,fgH)
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
        blended_portion = self.mergePng(background[start_y:end_y, start_x:end_x,:],foreground)
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

