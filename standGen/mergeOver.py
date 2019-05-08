from standGen import imageProcess
import cv2
def getMask(frameObject):
    if frameObject.getPng():
        return cv2.imread(frameObject.getMaster(),-1)[:,:,3:]
    else:
        return cv2.imread(frameObject.getMask(),-1)


def mergeOverTracking(frameObject,bg,p0,p1,dis):
    ratio = imageProcess.getRatio(bg,p0,p1,dis)
    if frameObject.getRatio() == None:
        frameObject.setInitRatio(ratio)
        anchor = imageProcess.getTran(frameObject.getRatio(), bg, p0, p1, frameObject.getAnchor())
        frameObject.setInitTran(anchor)
    else:
        anchor = imageProcess.getTran(frameObject.getRatio(), bg, p0, p1, frameObject.getAnchor())
        frameObject.nextFrame(ratio=ratio,tran=anchor)
    fg = cv2.imread(frameObject.getMaster(),-1)
    fgmask = getMask(frameObject)
    resizeFg = imageProcess.resize(fg,frameObject.getRatio())
    resizeFgmask = imageProcess.resize(fgmask,frameObject.getRatio())
    mergeFg2Bg = imageProcess.mergeWithAnchor(resizeFg,bg,frameObject.getTran()[1],frameObject.getTran()[0],resizeFgmask[:,:,:1])
    return mergeFg2Bg

def mergeOverCenter(frameObject,bg):
    Img = cv2.imread(frameObject.getMaster(),-1)[:,:,:-1]
    Mask = getMask(frameObject)
    goratio = imageProcess.getRatioMatchHeight(bg,Img)
    goAnchor = imageProcess.getTranMatchCenter(goratio,bg,frameObject.getAnchor())
    resizeImg = imageProcess.resize(Img,goratio)
    resizeMask = imageProcess.resize(Mask,goratio)
    mergeOver = imageProcess.mergeWithAnchor(resizeImg,bg,goAnchor[1],goAnchor[0],resizeMask)
    frameObject.nextFrame()
    return mergeOver
