import cv2
from standGen import frameObject,imageProcess,mergeOver,pointObject

input_video = 'video_1.mp4'
capture = cv2.VideoCapture(input_video)
#go = frameObject.frame('stands/explotion/',0.1,1)
ora = frameObject.frame('stands/ora/',0.1,1)
sp = frameObject.frame('stands/sp/',0.1,1)
frameNum = 0
p0 = pointObject.point(0.5,0.4)
p1 = pointObject.point(0.5,0.5)

while True:
    ret, frame = capture.read()
    mergeGo = mergeOver.addOverTracking(ora,frame,p0,p1,60,frameNum,frameRange=(0,100))
    mergeSp = mergeOver.mergeOverTracking(sp,mergeGo,p0,p1,60,frameNum,frameRange=(0,100))
    cv2.imwrite('out/test01.' + str(frameNum).zfill(4) + '.jpeg', mergeSp)
    frameNum += 1