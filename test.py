import cv2
from standGen import frameObject,imageProcess,mergeOver

input_video = 'video_1.mp4'
capture = cv2.VideoCapture(input_video)
go = frameObject.frame('stands/go/',0,0)
print (go.getMaster())
frameNum = 0
while True:
    ret, frame = capture.read()
    mergeGo = mergeOver.mergeOverCenter(go,frame)
    cv2.imwrite('out/test01.' + str(frameNum).zfill(4) + '.jpeg', mergeGo)
    frameNum += 1