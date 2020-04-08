import numpy as np
# import cupy as np
from enum import Enum

class DispBC:
    def __init__(self):
        self.BCs = []

    def AddDispBC(self, nodes, DOF, disp):
        if disp == "Fixed":
            dispNum = 0.0
        else:
            dispNum = disp
        self.BC = {"nodes": np.uint32(nodes), "DOF": np.uint32(DOF), "disp": np.float64(dispNum)}
        self.BCs.append(self.BC)

class NodalForce:
    def __init__(self):
        self.NFs = []

    def AddNodalForce(self, nodes, DOF, force):
        self.NF = {"nodes": np.uint32(nodes), "DOF": np.uint32(DOF), "force": np.float64(force)}
        self.NFs.append(self.NF)

class HarmonicForce:
    def __init__(self):
        self.HFs = []

    def AddHarmonicForce(self, mode, amplitude, phase):
        self.HF = {"mode": np.uint32(mode), "amplitude": np.float64(amplitude), "phase": np.float64(phase)}
        self.HFs.append(self.HF)

class DispModal:
    def __init__(self):
        self.DMs = []

    def AddDispModal(self, mode, amplitude):
        self.DM = {"mode": int(mode), "amplitude": np.float64(amplitude)}
        self.DMs.append(self.DM)

class Boundary:
    def __init__(self):
        self.dispBCs = dict()
        self.nodalForces = dict()
        self.harmonicForces = dict()
        self.dispModals = dict()

    def AddDispBC(self, name, dispBC):
        self.dispBCs.update([(name, dispBC)])

    def AddNodalForce(self, name, nodalForce):
        self.nodalForces.update([(name, nodalForce)])

    def AddHarmonicForce(self, name, harmonicForce):
        self.harmonicForces[name] = harmonicForce

    def AddDispModal(self, name, dispModal):
        self.dispModals[name] = dispModal

    def IsDispBC(self, name):
        for bcName in self.dispBCs.keys():
            if bcName == name:
                return True
        return False

    def IsNodalForce(self, name):
        for bcName in self.nodalForces.keys():
            if bcName == name:
                return True
        return False

    def GetHarmonicForceFromBCList(self, BCList):
        hfCount = 0
        for bcName in BCList:
            if bcName in self.harmonicForces.keys():
                hfCount += 1
                hfName = bcName
        if hfCount == 0:
            print("There is no harmonic force in BC list.")
            return None
        elif hfCount > 1:
            print("There should be only one harmonic force in BC list.")
            return None
        elif hfCount == 1:
            return self.harmonicForces[hfName]

    def GetDispModalFromBCList(self, BCList):
        dmCount = 0
        for bcName in BCList:
            if bcName in self.dispModals.keys():
                dmCount += 1
                dmName = bcName
        if dmCount == 0:
            # print("There is no displacement by mode in BC list.")
            return None
        elif dmCount > 1:
            print("There should be only one displacement by mode in BC list.")
            return None
        elif dmCount == 1:
            return self.dispModals[dmName]

    def GetNodalForcesFromBCList(self, BCList):
        nfCount = 0
        for bcName in BCList:
            if bcName in self.nodalForces.keys():
                nfCount += 1
                nfName = bcName
        if nfCount == 0:
            # print("There is no nodal force in BC list.")
            return None
        elif nfCount > 1:
            print("There should be only one nodal force in BC list.")
            return None
        elif nfCount == 1:
            return self.nodalForces[nfName]