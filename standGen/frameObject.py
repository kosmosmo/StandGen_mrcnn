from standGen import standObject
import os,time
class frame(object):
    def __init__(self,path,ratio,tolerance):
        """
        :type stands: List[standObject] #
        """
        self.stands = []
        for stand in os.listdir(path):
            self.stands.append(standObject.stand(path+stand+'/'))
        self.standIndex = 0
        self.ratio = ratio
        self.targetRatio = ratio
        self.tolerance = tolerance

    def setRatio(self):
        if self.targetRatio > self.ratio+(self.ratio*self.tolerance):
            self.ratio = self.ratio+(self.ratio*self.tolerance)
        elif self.targetRatio < self.ratio-(self.ratio*self.tolerance):
            self.ratio = self.ratio-(self.ratio*self.tolerance)
        else:
            self.ratio = self.targetRatio

    def nextFrame(self,ratio):
        self.stands[self.standIndex].nextFrame()
        self.targetRatio = ratio
        self.setRatio()
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

    def getRatio(self):
        return self.ratio








a = frame('../stands/sp/',1,0.05)
while True:
    print (a.nextFrame(0.99))
    print (a.getMaster())
    print (a.getRatio())
    time.sleep(1)