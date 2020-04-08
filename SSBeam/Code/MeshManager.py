import numpy as np
# import cupy as np
from enum import Enum

class ElementType(Enum):
    EulerBeam = 0

class IDSet:
    def __init__(self):
        self.IDList = []
    
    def AddID(self, id):
        self.IDList.append(id)

class Mesh:
    def __init__(self):
        self.nodeCoords = []
        self.elementNodes = []
        self.nodeSet = dict()
        self.elementSet = dict()
        self.nodeIndex = dict()
        self.elemIndex = dict()
        self.nodeCount = 0
        self.elemCount = 0

    def AppendNode(self, nn, coord):
        if len(nn)+len(coord) is not 0:
            self.nodeIndex.update([(int(nn), self.nodeCount)])
            self.nodeCoords.append(coord)
            self.nodeCount += 1
        else:
            print("Input Format Error: Nodal Coordinate (X-coord, Y-coord, Z-coord)")

    def AppendElement(self, en, nn):
        if len(en)+len(nn) is not 0:
            self.elemIndex.update([(int(en), self.elemCount)])
            self.elementNodes.append(nn)
            self.elemCount += 1
        else:
            print("Input Format Error: Element Number, 1st Node, 2nd Node, ...")

    def CreateNodeSet(self, name, nodeIDs):
        self.nodeSet.update([(name, np.array(nodeIDs, dtype = np.uint32))])

    def CreateElementSet(self, name, elementIDs):
        self.elementSet.update([(name, np.array(elementIDs, dtype = np.uint32))])

    def ConvertToNumpyNodes(self):
        self.nodeCoords = np.array(self.nodeCoords, dtype = np.float64)

    def ConvertToNumpyElements(self):
        self.elementNodes = np.array(self.elementNodes, dtype = np.uint32)

    def IsElset(self, key):
        for name in self.elementSet.keys():
            if name == key:
                return True
        return False
