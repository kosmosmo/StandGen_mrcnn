import cv2
from standGen import frameObject,imageProcess,mergeOver,pointObject

input_video = 'video_1.mp4'
capture = cv2.VideoCapture(input_video)
go = frameObject.frame('stands/explotion/',0.1,1)
print (go.getMaster())
frameNum = 0
p0 = pointObject.point(0.5,0.4)
p1 = pointObject.point(0.5,0.5)
while True:
    ret, frame = capture.read()
    mergeGo = mergeOver.mergeOverTracking(go,frame,p0,p1,60)
    cv2.imwrite('out/test01.' + str(frameNum).zfill(4) + '.jpeg', mergeGo)
    frameNum += 1