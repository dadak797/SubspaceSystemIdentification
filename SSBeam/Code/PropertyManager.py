from MeshManager import *
from enum import Enum


class DampingType(Enum):
    Rayleigh = 0
    Modal = 1

class ElementProperty:
    def __init__(self, props):
        self.matName = props[0]
        if props[1] == "EulerBeam":
            self.elementType = ElementType.EulerBeam
            self.secArea = np.float64(props[2])
            self.secondMoment = np.float64(props[3])

class Property:
    def __init__(self):
        self.elemProps = dict()

    def AddProperty(self, element, prop):
        self.elemProps[element] = prop

class CMass:
    def __init__(self):
        self.CMs = []

    def AddCMass(self, nodes, mass):
        self.CM = {"nodes": np.uint(nodes), "mass": np.float64(mass)}
        self.CMs.append(self.CM)        

class Damping:
    def __init__(self, dmpType):
        self.dmpRatio = dict()
        self.dmpType = dmpType
    
    def AddDampingRatio(self, mode, dRatio):
        self.dmpRatio[np.uint32(mode)] = np.float64(dRatio)

    def AddRayleighCoeffs(self, a0, a1):
        self.a0 = np.float64(a0)
        self.a1 = np.float64(a1)

class Assign:
    def __init__(self):
        self.properties = dict()
        self.cMasses = dict()
        self.damping = dict()

    def AddProperty(self, name, prop):
        self.properties[name] = prop

    def AddCMass(self, name, cMass):
        self.cMasses.update([(name, cMass)])

    def AddDamping(self, name, damping):
        self.damping.update([(name, damping)])

    def IsCMass(self, name):
        for bcName in self.cMasses.keys():
            if bcName == name:
                return True
        return False

    def IsDamping(self, name):
        for bcName in self.damping.keys():
            if bcName == name:
                return True
        return False