class stand(object):
    def __init__(self,path,state,anchor):
        """
        :type path: str
        :type state:list[int]
        :type anchor:(int,int)
        """
        self.path = path
        self.state = state
        self.anchor = anchor
        self.curFrame = 0

    def resetStand(self):
        self.curFrame = 0

    def getCurFrame(self):
        return self.curFrame

    def nextFrame(self):
        self.curFrame+=1
        return self.curFrame

    def getFilePath(self):
        return