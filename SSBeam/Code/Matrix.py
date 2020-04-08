# import numpy as np
import cupy as np
from PropertyManager import *
import Functions as fn
import AnalysisManager as am


class Matrix:
    def __init__(self, mesh, material, assign, boundary, analysis):
        self.mesh = mesh
        self.material = material
        self.assign = assign
        self.boundary = boundary
        self.analysis = analysis

    # Create incidence vector between node DOFs and global DOFs
    def CreateIncidence(self):
        self.totalDOF = self.mesh.nodeCount * self.analysis.nodeDOF

        if self.mesh.nodeCount == 0:
            print("Fail to create incidence vector between node DOFs and global DOFs (No nodes are defined at all.)")
        else:
            # incidence matrix: incid(node number, node DOF) = global DOF
            self.descDOFs = np.zeros((self.mesh.nodeCount, self.analysis.nodeDOF), dtype = np.uint32)
            self.fixedDOF = 0
            for bcName in self.analysis.BCList:
                if self.boundary.IsDispBC(bcName):
                    for BC in self.boundary.dispBCs[bcName].BCs:
                        self.descDOFs[self.mesh.nodeIndex[BC["nodes"]]][self.mesh.nodeIndex[BC["DOF"]]] = 1
                        self.fixedDOF += 1

            self.incid = np.zeros((self.mesh.nodeCount, self.analysis.nodeDOF), dtype = np.uint32)
            self.freeDOF = 0
            self.fixed = self.totalDOF - self.fixedDOF
            for node in range(self.mesh.nodeCount):
                for DOF in range(self.analysis.nodeDOF):
                    if self.descDOFs[node][DOF] == 0:
                        self.incid[node][DOF] = self.freeDOF
                        self.freeDOF += 1
                    else:
                        self.incid[node][DOF] = self.fixed
                        self.fixed += 1

    # Create Local Stiffness & Mass Matrix:
    def CreateLocals(self, ei, prop, isDerivative = False):
        if prop.elementType == ElementType.EulerBeam:
            # Calculate Length of Element and Get Young's Modulus(E)
            eleNodes = self.mesh.elementNodes[self.mesh.elemIndex[ei]]
            n1coord = self.mesh.nodeCoords[self.mesh.nodeIndex[eleNodes[0]]]
            n2coord = self.mesh.nodeCoords[self.mesh.nodeIndex[eleNodes[1]]]
            young = self.material.elasticMats[prop.matName].property[0]
            
            # Calculate EI, L, L^2, L^3
            if isDerivative == False:
                EI = young * prop.secondMoment
            else:
                EI = young
            L1 = np.linalg.norm(n1coord - n2coord, ord = 2)
            L2 = L1 * L1
            L3 = L1 * L1 * L1

            # Calculate Local Stiffness Matrix
            stiff_e = [[ 12*EI/L3,  6*EI/L2, -12*EI/L3,  6*EI/L2],
                       [  6*EI/L2,  4*EI/L1,  -6*EI/L2,  2*EI/L1],
                       [-12*EI/L3, -6*EI/L2,  12*EI/L3, -6*EI/L2],
                       [  6*EI/L2,  2*EI/L1,  -6*EI/L2,  4*EI/L1]]
            stiff_e = np.array(stiff_e, dtype = np.float64)

            # Calculate rho, A
            rho = self.material.elasticMats[prop.matName].property[2]
            area = prop.secArea

            # Calculate Local Mass Matrix
            mass_e = [[   156, 22*L1,     54, -13*L1],
                      [ 22*L1,  4*L2,  13*L1,  -3*L2],
                      [    54, 13*L1,    156, -22*L1],
                      [-13*L1, -3*L2, -22*L1,   4*L2]]
            mass_e = np.array(mass_e, dtype = np.float64)
            mass_e = rho*area*L1/420*mass_e

            return stiff_e, mass_e

    def AddConMass(self):
        for bcName in self.analysis.BCList:
            if self.assign.IsCMass(bcName):
                for key, value in self.assign.cMasses.items():
                    for cm in value.CMs:
                        if self.descDOFs[self.mesh.nodeIndex[cm["nodes"]]][0] == 0:
                            gDOF_i = self.incid[self.mesh.nodeIndex[cm["nodes"]]][0]
                            self.M_ff[gDOF_i][gDOF_i] = self.M_ff[gDOF_i][gDOF_i] + cm["mass"]

    def CreateDamping(self):
        for bcName in self.analysis.BCList:
            if self.assign.IsDamping(bcName):
                for value in self.assign.damping.values():
                    analysis = am.Analysis()
                    self.natFreq, self.modeShapes = analysis.RunModalAnalysis(self)
                    if value.dmpType == DampingType.Rayleigh:
                        self.C_ff = value.a0 * self.M_ff + value.a1 * self.K_ff
                    elif value.dmpType == DampingType.Modal:
                        self.C_ff = np.zeros((self.freeDOF, self.freeDOF), dtype = np.float64)
                        for i in np.arange(self.freeDOF):
                            self.C_ff = self.C_ff + 2.0 * value.dmpRatio[i+1] * (2.0 * np.pi * self.natFreq[i]) \
                                      * np.dot(self.modeShapes[:,[i]], self.modeShapes[:,[i]].transpose())
                        self.C_ff = np.dot(np.dot(self.M_ff, self.C_ff), self.M_ff)

    # Create Global Stiffness Matrix
    def CreateGlobals(self):
        # Stiffness Matrices
        self.K_ff = np.zeros((self.freeDOF, self.freeDOF), dtype = np.float64)
        self.K_fs = np.zeros((self.freeDOF, self.fixedDOF), dtype = np.float64)
        self.K_sf = np.zeros((self.fixedDOF, self.freeDOF), dtype = np.float64)
        self.K_ss = np.zeros((self.fixedDOF, self.fixedDOF), dtype = np.float64)

        # Mass Matrices
        self.M_ff = np.zeros((self.freeDOF, self.freeDOF), dtype = np.float64)
        self.M_fs = np.zeros((self.freeDOF, self.fixedDOF), dtype = np.float64)
        self.M_sf = np.zeros((self.fixedDOF, self.freeDOF), dtype = np.float64)
        self.M_ss = np.zeros((self.fixedDOF, self.fixedDOF), dtype = np.float64)

        props = self.assign.properties[self.analysis.assignName]
        for targetElement, elemProp in props.elemProps.items():
            if self.mesh.IsElset(targetElement):
                elements = self.mesh.elementSet[targetElement]

                for ei in elements:
                    stiff_e, mass_e = self.CreateLocals(ei, elemProp)

                    eleNodes = self.mesh.elementNodes[self.mesh.elemIndex[ei]]
                    eleNodeCount = len(eleNodes)
                    # nodeDOF = prop.nodeDOF
                    nodeDOF = self.analysis.nodeDOF
            
                    for iNode in np.arange(eleNodeCount):
                        for jNode in np.arange(eleNodeCount):
                            for iDOF in np.arange(nodeDOF):
                                for jDOF in np.arange(nodeDOF):
                                    gDOF_i = self.incid[self.mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                                    gDOF_j = self.incid[self.mesh.nodeIndex[eleNodes[jNode]]][jDOF]
                                    if gDOF_i < self.freeDOF and gDOF_j < self.freeDOF: 
                                        self.K_ff[gDOF_i][gDOF_j] = self.K_ff[gDOF_i][gDOF_j] \
                                                                + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                        self.M_ff[gDOF_i][gDOF_j] = self.M_ff[gDOF_i][gDOF_j] \
                                                                + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    elif gDOF_i < self.freeDOF and gDOF_j >= self.freeDOF:
                                        self.K_fs[gDOF_i][gDOF_j - self.freeDOF] = self.K_fs[gDOF_i][gDOF_j - self.freeDOF] \
                                                                + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                        self.M_fs[gDOF_i][gDOF_j - self.freeDOF] = self.M_fs[gDOF_i][gDOF_j - self.freeDOF] \
                                                                + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    elif gDOF_i >= self.freeDOF and gDOF_j < self.freeDOF:
                                        self.K_sf[gDOF_i - self.freeDOF][gDOF_j] = self.K_sf[gDOF_i - self.freeDOF][gDOF_j] \
                                                                + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                        self.M_sf[gDOF_i - self.freeDOF][gDOF_j] = self.M_sf[gDOF_i - self.freeDOF][gDOF_j] \
                                                                + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    else:
                                        self.K_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] = self.K_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] \
                                                                + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                        self.M_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] = self.M_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] \
                                                                + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
            else:
                ei = np.uint32(targetElement)
                stiff_e, mass_e = self.CreateLocals(ei, elemProp)

                eleNodes = self.mesh.elementNodes[self.mesh.elemIndex[ei]]
                eleNodeCount = len(eleNodes)
                # nodeDOF = prop.nodeDOF
                nodeDOF = self.analysis.nodeDOF
        
                for iNode in np.arange(eleNodeCount):
                    for jNode in np.arange(eleNodeCount):
                        for iDOF in np.arange(nodeDOF):
                            for jDOF in np.arange(nodeDOF):
                                gDOF_i = self.incid[self.mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                                gDOF_j = self.incid[self.mesh.nodeIndex[eleNodes[jNode]]][jDOF]
                                if gDOF_i < self.freeDOF and gDOF_j < self.freeDOF: 
                                    self.K_ff[gDOF_i][gDOF_j] = self.K_ff[gDOF_i][gDOF_j] \
                                                            + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    self.M_ff[gDOF_i][gDOF_j] = self.M_ff[gDOF_i][gDOF_j] \
                                                            + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                elif gDOF_i < self.freeDOF and gDOF_j >= self.freeDOF:
                                    self.K_fs[gDOF_i][gDOF_j - self.freeDOF] = self.K_fs[gDOF_i][gDOF_j - self.freeDOF] \
                                                            + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    self.M_fs[gDOF_i][gDOF_j - self.freeDOF] = self.M_fs[gDOF_i][gDOF_j - self.freeDOF] \
                                                            + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                elif gDOF_i >= self.freeDOF and gDOF_j < self.freeDOF:
                                    self.K_sf[gDOF_i - self.freeDOF][gDOF_j] = self.K_sf[gDOF_i - self.freeDOF][gDOF_j] \
                                                            + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    self.M_sf[gDOF_i - self.freeDOF][gDOF_j] = self.M_sf[gDOF_i - self.freeDOF][gDOF_j] \
                                                            + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                else:
                                    self.K_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] = self.K_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] \
                                                            + stiff_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]
                                    self.M_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] = self.M_ss[gDOF_i - self.freeDOF][gDOF_j - self.freeDOF] \
                                                            + mass_e[iDOF + iNode * nodeDOF][jDOF + jNode * nodeDOF]

        # Add Concentrated Mass on Global Mass Matrix
        self.AddConMass()

        # Construct Damping Matrix
        self.CreateDamping()

    # Create Force Vector
    def CreateForceVector(self):
        self.F_f = np.zeros((self.freeDOF,), dtype = np.float64)
        nodalForces = self.boundary.GetNodalForcesFromBCList(self.analysis.BCList)
        if nodalForces is not None:
            for nodalForce in nodalForces.NFs:
                fixed = self.descDOFs[self.mesh.nodeIndex[nodalForce["nodes"]]][self.mesh.nodeIndex[nodalForce["DOF"]]]
                iDOF = self.incid[self.mesh.nodeIndex[nodalForce["nodes"]]][self.mesh.nodeIndex[nodalForce["DOF"]]]
                if fixed == 0:
                    self.F_f[iDOF] = self.F_f[iDOF] + nodalForce["force"]
                elif fixed == 1:
                    print("Force cannot be applied with displacement boundary condition at the same time.")
            return True
        else:
            return False
            
        # for bcName in self.analysis.BCList:
        #     if self.boundary.IsNodalForce(bcName):
        #         for nodalForce in self.boundary.nodalForces[bcName].NFs:
        #             fixed = self.descDOFs[self.mesh.nodeIndex[nodalForce["nodes"]]][self.mesh.nodeIndex[nodalForce["DOF"]]]
        #             iDOF = self.incid[self.mesh.nodeIndex[nodalForce["nodes"]]][self.mesh.nodeIndex[nodalForce["DOF"]]]
        #             if fixed == 0:
        #                 self.F_f[iDOF] = self.F_f[iDOF] + nodalForce["force"]
        #             elif fixed == 1:
        #                 print("Force cannot be applied with displacement boundary condition at the same time.")


    def MakeCompatibilityMatrix(self, ei):
        eleNodes = self.mesh.elementNodes[self.mesh.elemIndex[ei]]
        eleNodeCount = len(eleNodes)
        nodeDOF = self.analysis.nodeDOF

        C_i = np.zeros((eleNodeCount*nodeDOF, self.freeDOF), dtype=np.uint8)

        for iNode in np.arange(eleNodeCount):
            for iDOF in np.arange(nodeDOF):
                gDOF_i = self.incid[self.mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                if gDOF_i < self.freeDOF:
                    C_i[iDOF+iNode*nodeDOF, gDOF_i] = 1

        return C_i