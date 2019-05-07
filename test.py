import cv2
from standGen import imageProcess

original_image = 'demo/2.jpg'
bg = cv2.imread(original_image,-1)
##will replace to frameobject
fg = cv2.imread('demo/sp.jpeg',-1)
fgmask = cv2.imread('demo/spMask.jpeg',-1)[:,:,:1]

print (fgmask)
newI = imageProcess.mergePng(fg,fg,fgmask,flag=True)

cv2.imshow('test',newI)
cv2.waitKey(0)