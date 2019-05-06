class frame(object):
    def __init__(self,stands):
        """
        :type stands: List[standObject] #
        """
        self.stands = stands
        self.standIndex = 0

    def nextFrame(self):
        self.stands[self.standIndex].nextFrame()

    def changeStand(self,index):
        self.standIndex = index
        self.stands[self.standIndex].resetStand()
