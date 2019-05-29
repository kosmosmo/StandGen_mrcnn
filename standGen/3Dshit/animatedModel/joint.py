import numpy as np
class Joint():
    def __init__(self,index,name,bindLocalTransform):
        """
        :type bindLocalTransform: Matrix4f
        :type animatedTransform: Matrix4f
        :type inverseBindTransform: Matrix4f
        :type children: List<>
        """
        self.index = index
        self.name = name
        self.localBindTransform = bindLocalTransform
        self.animatedTransform = np.zeros((4,4), dtype=float)
        self.inverseBindTransform = np.zeros((4,4), dtype=float)
        self.children = []

    def addChold(self,child):
        self.children.append(child)

    def getAnimatedTransform(self):
        return self.animatedTransform

    def setAnimatedTransform(self,animationTransform):
        self.animatedTransform = animationTransform

    def getInverseBindTransform(self):
        return self.inverseBindTransform

    def calcInverseBindTransform(self, parentBindTransform):
        bindTransform = np.matmul(parentBindTransform,self.localBindTransform)
        self.inverseBindTransform = np.linalg.inv(bindTransform)
        for child in self.children:
            child.calcInverseBindTransform(bindTransform)


