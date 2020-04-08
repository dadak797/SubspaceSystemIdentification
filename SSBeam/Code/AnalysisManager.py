from enum import Enum
import numpy as np
# import cupy as np
from Matrix import *
from scipy.linalg import eig
from scipy import signal
import Functions as fn
from ResultManager import *
# from FileIO import *
import FileIO as fIO


class AnalysisType(Enum):
    LinearStatic = 0
    Modal = 1
    Dynamic = 2

class LoadType(Enum):
    FreeVibration = 0
    RandomVibration = 1
    HarmonicVibration = 2

class Analysis:
    nodeDOF = 2  # 2D Beam has 2 DOFs per node

    def SetAnalysisType(self, anaType):
        if anaType == "LinearStatic":
            self.anaType = AnalysisType.LinearStatic
        elif anaType == "Modal":
            self.anaType = AnalysisType.Modal
        elif anaType == "Dynamic":
            self.anaType = AnalysisType.Dynamic
        else:
            print(anaType, "is a wrong analysis type.")

    def SetAssigns(self, assignName, params):
        self.assignName = assignName
        self.BCList = params

    def SetLoadType(self, loadType):
        if loadType == "FreeVibration":
            self.loadType = LoadType.FreeVibration
        elif loadType == "RandomVibration":
            self.loadType = LoadType.RandomVibration
        elif loadType == "HarmonicVibration":
            self.loadType = LoadType.HarmonicVibration
        else:
            print("There is no such load type.")

    def SetTimeStep(self, time):
        self.Ti = np.float64(time[0])
        self.Tf = np.float64(time[1])
        self.dT = np.float64(time[2])

    def RunAnalysis(self, matrix, boundary):
        result = Result(self.anaType)
        timeManager = fIO.TimeManager()
        timeManager.StartTimer("Analysis")
        if self.anaType == AnalysisType.LinearStatic:
            disp = self.RunLinearStaticAnalysis(matrix)
            result.SetResults(disp)
        elif self.anaType == AnalysisType.Modal:
            natFreq, modeShapes = self.RunModalAnalysis(matrix)
            result.SetResults(natFreq, modeShapes)
        elif self.anaType == AnalysisType.Dynamic:
            disp, vel, acc = self.RunDynamicAnalysis(matrix, boundary)
            result.SetResults(self.Ts, disp, vel, acc)
        timeManager.EndTimer("Analysis")
        return result

    def RunModalAnalysis(self, matrix):
        w2, x = eig(matrix.K_ff, matrix.M_ff)  # Get Eigenvalues & Eigenvectors : K x = w2 M x
        w = np.sqrt(w2.real)
        idx = np.argsort(w)  # Get Index for Sorting with Descending Order
        w = w[idx]  # Angular Frequencies (rad/sec)
        f = w / (2 * np.pi)  # Eigenfrequencies (Hz=1/sec)
        x = x[:,idx]  # Relocation of Eigenvectors

        # Get Mass Normalized Eigenvectors
        Mdiag = np.diagonal(np.dot(np.dot(x.transpose(), matrix.M_ff), x))  # x^T * M * X
        x = x / np.sqrt(Mdiag)

        return f, x

    def RunLinearStaticAnalysis(self, matrix):
        IsForceVector = matrix.CreateForceVector()  # Create Force Vector
        if IsForceVector:
            return np.linalg.solve(matrix.K_ff, matrix.F_f)
        else:
            return np.zeros((matrix.F_f.shape), dtype=np.float64)

    def CreateTimeArray(self, Ti, Tf, dT):
        TCount = int((Tf - Ti) / dT) + 1
        Ts = np.linspace(Ti, Tf, num = TCount)
        return Ts

    def GetInitialConditions(self, matrix, boundary):
        # Get Initial Displacement by Nodal Force
        hasNodalForce = matrix.CreateForceVector()
        if hasNodalForce:
            u0_NF = np.linalg.solve(matrix.K_ff, matrix.F_f)  # u0 = K^-1*F
        else:
            u0_NF = np.zeros((matrix.F_f.shape), dtype=np.float64)

        # Get Initial Displacement by Mode Shapes
        u0_DM = self.GetInitDispByMode(matrix, boundary)

        v0 = np.zeros((matrix.freeDOF,), dtype = np.float64)  # v = 0 for all DOFs
        return u0_NF + u0_DM, v0

    def GetInitDispByMode(self, matrix, boundary):
        dispModals = boundary.GetDispModalFromBCList(self.BCList)
        u0_DM = np.zeros((matrix.F_f.shape), dtype=np.float64)
        if dispModals is not None:
            for dispModal in dispModals.DMs:
                u0_DM = u0_DM + matrix.modeShapes[:,dispModal['mode']-1] * dispModal['amplitude']
        return u0_DM

    # def Newmark(self, matrix, disp, vel, acc, beta = 0.25, gamma = 0.5):

    def SolveStateSpace(self, matrix, u0, v0):
        DOFs = matrix.freeDOF  # Number of free DOF
        Minv = np.linalg.inv(matrix.M_ff)  # Inverse M

        matO = np.zeros((DOFs, DOFs), dtype = np.float64)  # Zero matrix
        matI = np.eye(DOFs, dtype = np.float64)  # Identity matrix
        MinvK = np.dot(Minv, matrix.K_ff)  # Inverse M * K
        MinvC = np.dot(Minv, matrix.C_ff)  # Inverse C * K

        Ac = np.vstack((np.hstack((matO, matI)), np.hstack((-MinvK, -MinvC))))
        Bc = np.vstack((matO, Minv))
        Cc = np.hstack((matI, matO))
        Dc = matO

        # Write Eigenvalue of A_exact
        eigval, _ = eig(Ac)
        eigval = np.array([eigval], dtype=np.complex128)
        fn.WriteMatrix("Res_EigenvalueAc.out", eigval)

        system = signal.lti(Ac, Bc, Cc, Dc)  # Continuous Linear Time Invariant System with Ac, Bc, Cc, Dc

        X0 = np.hstack((u0, v0))  # Initial Condition: [u0, v0]

        _, Y, X = signal.lsim(system, self.U, self.Ts, X0 = X0)  # Tout: Time, Output Y, state X = [disp, vel]
        # Calculate Acceleration (a = M^-1*(-Ku-Cv+f))
        acc = np.dot(X[:,0:DOFs], -MinvK.transpose()) + np.dot(X[:,DOFs:2*DOFs], -MinvC.transpose()) + np.dot(self.U, Minv.transpose())

        return Y, X[:,DOFs:2*DOFs], acc


    def RunDynamicAnalysis(self, matrix, boundary):
        self.Ts = self.CreateTimeArray(self.Ti, self.Tf, self.dT)
        u0, v0 = self.GetInitialConditions(matrix, boundary)
        self.GetExternalForce(matrix, boundary)
        disp, vel, acc = self.SolveStateSpace(matrix, u0, v0)

        return disp, vel, acc


    def GetExternalForce(self, matrix, boundary):
        DOFs = matrix.freeDOF
        # External Force
        if self.loadType == LoadType.FreeVibration:
            self.U = np.zeros((len(self.Ts), DOFs), dtype = np.float64)
        elif self.loadType == LoadType.RandomVibration:
            np.random.seed(1234)
            amplitude = 1000.
            self.U = amplitude * (np.random.rand(len(self.Ts), DOFs) - 0.5)  # random.rand() returns random number of 0 to 1
        elif self.loadType == LoadType.HarmonicVibration:
            self.U = np.zeros((len(self.Ts), DOFs), dtype = np.float64)
            harmonic = boundary.GetHarmonicForceFromBCList(self.BCList)
            self.GenerateHarmonicForce(harmonic, matrix)
        else:
            pass


    def GenerateHarmonicForce(self, harmonic, matrix):
        for hf in harmonic.HFs:
            modeID = hf["mode"] - 1
            amplitude = hf["amplitude"]
            phaseAngle = np.deg2rad(hf["phase"])

            modeShape = matrix.modeShapes[:,modeID]
            natFreq = matrix.natFreq[modeID]
            # U = np.zeros((len(self.Ts), matrix.freeDOF), dtype = np.float64)
            Ts = self.Ts.reshape((self.Ts.shape[0], 1))
            self.U = self.U + amplitude*modeShape*np.sin(2*np.pi*natFreq*Ts-phaseAngle)


        # plt.figure(figsize = (10, 8))
        # plt.title("Harmonic Load", loc='center')

        # for iDOF in np.arange(matrix.freeDOF):
        #     label = str(iDOF+1) + 'th DOF'
        #     plt.plot(self.Ts, self.U[:, iDOF], label=label)
        # plt.legend(loc='lower right', framealpha=0.5)
        # plt.xlabel("Time (sec)")
        # plt.ylabel("Force (N)")

        # plt.show()
