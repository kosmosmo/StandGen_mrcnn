import json,time
class stand(object):
    def __init__(self,path):
        """
        :type path: str
        :type state:list[int]
        :type anchor:(int,int)
        """
        self.end = False
        with open(path+'info.json') as json_file:
            info = json.load(json_file)
            json_file.close()
        self.path = path
        self.states = info['endFrame']
        self.anchor = info['anchor']
        self.png = info['png']
        self.curFrame = 0
        self.curState = 0
        self.curStart = 0
        if self.png: self.ext = '.png'
        else: self.ext = '.jpg'

    def getPng(self):
        return self.png

    def resetStand(self):
        self.curFrame = 0
        self.end = False

    def getCurFrame(self):
        return self.curFrame

    def nextFrame(self):
        self.curFrame+=1
        if self.curFrame > self.states[self.curState][0]:
            if self.states[self.curState][1] == False:
                self.curState = (self.curState+1)
                if self.curState >= len(self.states):
                    self.end= True
                    return
                self.curStart = self.curFrame
            else:
                self.curFrame = self.curStart
        return self.curFrame

    def getMaster(self):
        return self.path+'master/master.'+str(self.curFrame).zfill(5)+self.ext

    def getMask(self):
        return self.path+'mask/master.'+str(self.curFrame).zfill(5)+self.ext

    def getAnchor(self):
        return self.anchor

    def getEnd(self):
        return self.end


