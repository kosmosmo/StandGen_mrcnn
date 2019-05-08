from standGen import standObject
import os,time
class frame(object):
    def __init__(self,path,tolerance,tranPixel):
        """
        :type stands: List[standObject] #
        """
        self.stands = []
        for stand in os.listdir(path):
            self.stands.append(standObject.stand(path+stand+'/'))
        self.standIndex = 0
        self.ratio = None
        self.tran = None
        self.targetRatio = None
        self.targetTran = None
        self.tolerance = tolerance
        self.tranPixel = tranPixel

    def setRatio(self):
        print (self.targetRatio)
        if self.targetRatio == None:
            return
        if self.targetRatio > self.ratio+(self.ratio*self.tolerance):
            self.ratio = self.ratio+(self.ratio*self.tolerance)
        elif self.targetRatio < self.ratio-(self.ratio*self.tolerance):
            self.ratio = self.ratio-(self.ratio*self.tolerance)
        else:
            self.ratio = self.targetRatio

    def setTran(self):
        tran = [self.tran[0],self.tran[1]]
        if self.targetTran[0] > self.tran[0]+self.tranPixel:
            tran[0] = self.tran[0]+self.tranPixel
        elif self.targetTran[0] < self.tran[0]-self.tranPixel:
            tran[0] = self.tran[0] - self.tranPixel
        if self.targetTran[1] > self.tran[1]+self.tranPixel:
            tran[1] = self.tran[1]+self.tranPixel
        elif self.targetTran[1] < self.tran[1]-self.tranPixel:
            tran[1] = self.tran[1] - self.tranPixel
        self.tran = tran

    def nextFrame(self,ratio=None,tran=None):
        self.stands[self.standIndex].nextFrame()
        if ratio:
            self.targetRatio = ratio
            self.setRatio()
        if tran:
            self.targetTran = tran
            self.setTran()
        return self.stands[self.standIndex].curFrame

    def getCur(self):
        return self.stands[self.standIndex].curFrame

    def changeStand(self,index):
        self.standIndex = index
        self.stands[self.standIndex].resetStand()

    def getMaster(self):
        return self.stands[self.standIndex].getMaster()

    def getMask(self):
        return self.stands[self.standIndex].getMask()

    def getAnchor(self):
        return self.stands[self.standIndex].getAnchor()

    def getPng(self):
        return self.stands[self.standIndex].getPng()

    def getEnd(self):
        return self.stands[self.standIndex].getEnd()

    def setInitRatio(self,ratio):
        self.ratio = ratio
        self.targetRatio = ratio

    def setInitTran(self,tran):
        self.tran = tran
        self.targetRatio = tran

    def getRatio(self):
        return self.ratio

    def getTran(self):
        return self.tran











