from standGen import imageProcess
import cv2,sys


def getImg(frameObject):
    if frameObject.getPng():
        master = cv2.imread(frameObject.getMaster(),-1)[:,:,:-1]
        mask = cv2.imread(frameObject.getMaster(),-1)[:,:,3:]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return (master,mask)
    else:
        master = cv2.imread(frameObject.getMaster(),-1)
        mask = cv2.imread(frameObject.getMask(),-1)
        return (master,mask)


def mergeOverTracking(frameObject,bg,p0,p1,dis,curFrame,frameRange = (0,sys.maxsize)):
    if curFrame < frameRange[0] or curFrame >= frameRange[1]: return bg
    if frameObject.getEnd() == True: return bg
    ratio = imageProcess.getRatio(bg,p0,p1,dis)
    if frameObject.getRatio() == None:
        frameObject.setInitRatio(ratio)
        anchor = imageProcess.getTran(frameObject.getRatio(), bg, p0, p1, frameObject.getAnchor())
        frameObject.setInitTran(anchor)
    else:
        anchor = imageProcess.getTran(frameObject.getRatio(), bg, p0, p1, frameObject.getAnchor())
        frameObject.nextFrame(ratio=ratio,tran=anchor)
    ##during set init value, 1 frame has been offset, need to check again before excute
    if frameObject.getEnd() == True: return bg
    img = getImg(frameObject)
    fg = img[0]
    fgmask = img[1]
    resizeFg = imageProcess.resize(fg,frameObject.getRatio())[:,:,:3]
    resizeFgmask = imageProcess.resize(fgmask,frameObject.getRatio())
    mergeFg2Bg = imageProcess.mergeWithAnchor(resizeFg,
                                              bg,
                                              frameObject.getTran()[1],
                                              frameObject.getTran()[0],
                                              resizeFgmask[:,:,:1])
    return mergeFg2Bg

def mergeOverCenter(frameObject,bg,curFrame,frameRange=(0,sys.maxsize)):
    if curFrame < frameRange[0] or curFrame >= frameRange[1]: return bg
    if frameObject.getEnd() == True: return bg
    img = getImg(frameObject)
    fg = img[0]
    fgmask = img[1]
    goratio = imageProcess.getRatioMatchHeight(bg,fg)
    goAnchor = imageProcess.getTranMatchCenter(goratio,bg,frameObject.getAnchor())
    resizeImg = imageProcess.resize(fg,goratio)
    resizeMask = imageProcess.resize(fgmask,goratio)
    mergeOver = imageProcess.mergeWithAnchor(resizeImg,bg,goAnchor[1],goAnchor[0],resizeMask[:,:,:1])
    frameObject.nextFrame()
    return mergeOver
