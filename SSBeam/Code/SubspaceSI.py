import numpy as np
# import cupy as np
import Functions as fn
from enum import Enum
from scipy.linalg import logm
from AnalysisManager import LoadType
from scipy.linalg import eig
from AnalysisManager import *
from Matrix import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy as cp
import FileIO as fIO


class OutputType(Enum):
    Disp = 0
    Vel = 1
    Acc = 2

class SubspaceSI:
    def __init__(self):
        self.usedInputDOFs = []
        self.usedOutputDOFs = []

    def SetLoadType(self, loadType):
        if loadType == "FreeVibration":
            self.loadType = LoadType.FreeVibration
        elif loadType == "RandomVibration":
            self.loadType = LoadType.RandomVibration
        elif loadType == "HarmonicVibration":
            self.loadType = LoadType.HarmonicVibration
        else:
            print("Load type is not set for Subspace SI.")

    def SetOutputType(self, outputType):
        if outputType == "Disp":
            self.outputType = OutputType.Disp
        elif outputType == "Vel":
            self.outputType = OutputType.Vel
        elif outputType == "Acc":
            self.outputType = OutputType.Acc

    def SetMeasurementNoiseInfo(self, seedNumber, noiseMagnitude):
        self.seedNumber = int(seedNumber)
        self.noiseMagnitude = np.float64(noiseMagnitude)

    def SetHankelMatrixParams(self, kk, dT, Tw):
        self.kk = np.uint32(kk)
        self.dT = np.float64(dT)
        self.Tw = np.float64(Tw)  # Time Window Size (sec)

    def SetTimeParams(self, Ti, Tf, Tstride, update):
        self.Ti = np.float64(Ti)
        self.Tf = np.float64(Tf)
        self.Tstride = np.float64(Tstride)
        if update == "Update":
            self.PriorUpdate = True
        elif update == "NoUpdate":
            self.PriorUpdate = False
        else:
            print("Prior information update keyword should be Update or NoUpdate.")

    def SetPriorInfo(self, assignName, params):
        self.priorAssignName = assignName
        self.priorBCList = params

    def AddUsedInputDOF(self, node, DOF):
        self.usedInputDOFs.append([int(node), int(DOF)])

    def AddUsedOutputDOF(self, node, DOF):
        self.usedOutputDOFs.append([int(node), int(DOF)])

    def ConstructHankel(self, mesh, matrix, analysis, result):
        self.timeManager.StartTimer("Hankel Matrix Construction")
        self.NN = np.uint32(int(self.Tw/self.dT) + 1)  # Total Steps
        self.nn = matrix.freeDOF * 2  # System Dimension

        # Get Measured Location Matrix (C_loc) (C_d, C_v and C_a are same, if the measured location is same)
        self.C_loc = np.zeros((self.pp, matrix.freeDOF), dtype = np.uint8)
        for ip in np.arange(self.pp):
            self.C_loc[ip, self.yMask[ip]] = 1

        # Check the Final Timestep Data
        if self.NN + self.kk - 2 > self.y_p.shape[0]:
            print("The data is not exist for the given time index. Check whether a data of the final timestep(k+N-2) exists.")
            return

        # Check Size of the Hankel Matrix
        if self.kk <= self.nn:
            print("The value k should be larger than the system matrix dimension n.")
            print("The value k is", self.kk, "and the system dimension n is", self.nn)
            return
        elif self.kk > np.uint32(self.Tw / (np.float64(self.mm + self.pp) * self.dT)):
            print("The value k should not be larger than the value Tw/((m+p)dT).")
            print("The value k is", self.kk, "and the value Tw/((m+p)dT) is", np.uint32(self.Tw / (np.float64(self.mm + self.pp) * self.dT)))
            print("Tw:", self.Tw, ", m:", self.mm, ", p:", self.pp, ", dT:", self.dT)
            return

        # Construct Input Hankel Matrix (Uk)
        # Not to transpose twice, Uk_T is constructed. (u_m is transposed and Wk is transposed for LQ Decomposition)
        Uk_T = np.zeros((self.NN, self.kk * self.mm), dtype = np.float64)
        for i_k in np.arange(self.kk):
            Uk_T[:, i_k*self.mm:(i_k+1)*self.mm] = self.u_m[self.idxTi+i_k:self.idxTi+self.NN+i_k,:]
        print("Uk construction has finished with size", (Uk_T.shape[1], Uk_T.shape[0]))

        # Construct Output Hankel Matrix (Yk)
        # Not to transpose twice, Yk_T is constructed. (y_p is transposed and Wk is transposed for LQ Decomposition)
        Yk_T = np.zeros((self.NN, self.kk * self.pp), dtype = np.float64)
        for i_k in np.arange(self.kk):
            Yk_T[:, i_k*self.pp:(i_k+1)*self.pp] = self.y_p[self.idxTi+i_k:self.idxTi+self.NN+i_k,:]
        print("Yk construction has finished with size", (Yk_T.shape[1], Yk_T.shape[0]))

        # Construct Hankel Matrix
        self.Wk_T = np.hstack((Uk_T, Yk_T))

        self.timeManager.EndTimer("Hankel Matrix Construction")


    def LQFactorization(self):
        # LQ Factorization is performed by using QR Factorization:
        # A = LQ^T
        # Q, R = QR(A.transpose())
        # L = R.transpose()
        
        Q, L_T = np.linalg.qr(self.Wk_T)
        L = L_T.transpose()
        
        km = self.kk * self.mm
        kp = self.kk * self.pp

        self.L11 = L[:km, :km]
        self.L21 = L[km:km+kp, 0:km]
        self.L22 = L[km:km+kp, km:km+kp]

        self.Q1 = Q[:, :km]
        self.Q2 = Q[:, km:km+kp]


    def SVDecomposition(self):
        # Singular Value Decomposition
        U, S, V = np.linalg.svd(self.L22)

        # Get system dimension of the estimated systems.
        # If dimension decision method is required, code here
        # self.nDim = self.nn  # For All Modes Measured
        # if self.nDim % 2 != 0:
        #     print("Dimension of the estimated system should be an even number.")
        self.nDim = 8  # Temporary

        km = self.kk * self.mm
        kp = self.kk * self.pp

        self.U1 = U[:, :self.nDim]
        self.S1 = S[:self.nDim]
        self.V1 = V[:, :self.nDim]

        self.SS = S[:self.nDim*2]  # To write singular values on file
        # U2 = U[:, self.nDim:kp]
        # S2 = S[self.nDim:kp]
        # v2 = V[:, :self.nDim]


    def DiscreteSystemMatrices(self):
        Gk = self.U1 * np.sqrt(self.S1)

        Gk1 = Gk[:self.pp*(self.kk-1), :]
        Gk2 = Gk[self.pp:self.kk*self.pp, :]

        self.Ad_hat = np.dot(np.linalg.pinv(Gk1), Gk2)
        self.Cd_hat = Gk[:self.pp, :self.nn]

        # Calculate Bd_hat and Dd_hat
        if self.loadType != LoadType.FreeVibration:
            pass
        else:
            pass


    def DiscreteToContinuous(self):
        # Convert Discrete System Matrices to Continuous System Matrices
        self.Ac_hat = logm(self.Ad_hat) / self.dT
        self.Cc_hat = self.Cd_hat

        # # Write Eigenvalue of Ac_hat
        # eigval, _ = eig(self.Ac_hat)
        # idx = np.argsort(-np.abs(eigval.imag))
        # eigval = eigval[idx]
        # self.eigvalAc.append(eigval)

        # Calculate Bc_hat and Dc_hat
        if self.loadType != LoadType.FreeVibration:
            # self.Bc_hat = np.linalg(self.Ad_hat - np.eye(n_est), np.dot(self.Ac_hat, self.Bd_hat))
            # self.Dc_hat = self.Dd_hat
            pass
        else:
            pass


    def EstimatedSystemMatrices(self):
        self.timeManager.StartTimer("LQ Factorization")
        self.LQFactorization()
        self.timeManager.EndTimer("LQ Factorization")

        self.timeManager.StartTimer("Singular Value Decomposition")
        self.SVDecomposition()
        self.timeManager.EndTimer("Singular Value Decomposition")

        self.DiscreteSystemMatrices()
        self.DiscreteToContinuous()


    def EstimateElementProperty(self, K_est, matrix, mesh, material, assign):
        matPI = np.zeros((mesh.elemCount, mesh.elemCount), dtype=np.float64)
        vecK = np.zeros((mesh.elemCount,), dtype=np.float64)

        props = assign.properties[self.priorAssignName]
        for elem_j, elemProp_j in props.elemProps.items():
            ej = np.uint32(elem_j)
            eleNodes_j = mesh.elementNodes[mesh.elemIndex[ej]]

            eleNodeCount = len(eleNodes_j)
            analysis = Analysis()
            nodeDOF = analysis.nodeDOF

            C_j = matrix.MakeCompatibilityMatrix(ej)
            kd_j, _ = matrix.CreateLocals(ej, elemProp_j, isDerivative=True)

            CkdC_j = np.dot(np.dot(C_j.transpose(), kd_j), C_j)

            for elem_i, elemProp_i in props.elemProps.items():
                ei = np.uint32(elem_i)
                eleNodes_i = mesh.elementNodes[mesh.elemIndex[ei]]

                C_i = matrix.MakeCompatibilityMatrix(ei)
                kd_i, _ = matrix.CreateLocals(ei, elemProp_i, isDerivative=True)

                CkdC_i = np.dot(np.dot(C_i.transpose(), kd_i), C_i)

                matPI[mesh.elemIndex[ej], mesh.elemIndex[ei]] = np.sum(CkdC_j*CkdC_i)

            vecK[mesh.elemIndex[ej]] = np.sum(CkdC_j*K_est)

        return np.linalg.solve(matPI, vecK)


    def ArbitraryToPhysical(self, matrix, mesh, material, assign, boundary, analysis):
        self.timeManager.StartTimer("Similarity Transformation")

        nq = int(self.nDim / 2) 
        freeDOF = int(self.nn / 2)

        Minv = np.linalg.inv(matrix.M_ff)

        MinvKp = np.dot(Minv, self.Kprior)
        MinvCp = np.dot(Minv, self.Cprior)

        # Partial Modes Measured
        if nq < freeDOF:
            nUnmeasured = freeDOF - nq

            matO_bar = np.zeros((nUnmeasured, nUnmeasured), dtype = np.float64)  # Zero matrix
            matI_bar = np.eye(nUnmeasured, dtype = np.float64)  # Identity matrix

            modeShape_bar = self.modeShapesKp[:,nq:freeDOF]

            Kp_bar = np.dot(np.dot(modeShape_bar.transpose(), self.Kprior), modeShape_bar)
            Cp_bar = np.dot(np.dot(modeShape_bar.transpose(), self.Cprior), modeShape_bar)

            Aprior_bar = np.vstack((np.hstack((matO_bar, matI_bar)), np.hstack((-Kp_bar, -Cp_bar))))

            # About Bc_hat
            if self.loadType != LoadType.FreeVibration:
                pass
            else:
                pass

            matO_mu = np.zeros((self.nDim, self.nn-self.nDim), dtype=np.float64)
            matO_um = np.zeros((self.nn-self.nDim, self.nDim), dtype=np.float64)

            self.Ac_hat = np.vstack((np.hstack((self.Ac_hat, matO_mu)), np.hstack((matO_um, Aprior_bar))))

            Cc_bar = np.hstack((-np.dot(MinvKp, modeShape_bar), -np.dot(MinvCp, modeShape_bar)))
            self.Cc_hat = np.hstack((self.Cc_hat, np.dot(self.C_loc, Cc_bar)))

        # Write Eigenvalue of Ac_hat
        eigval, _ = eig(self.Ac_hat)
        idx = np.argsort(-np.abs(eigval.imag))
        eigval = eigval[idx]
        self.eigvalAc.append(eigval)

        # Ac_hat = [[A_11, A_12]
        #           [A_21, A_22]]
        # A_11_T = self.Ac_hat[:nq, :nq].transpose()
        # A_12_T = self.Ac_hat[:nq, nq:2*nq].transpose()
        # A_21_T = self.Ac_hat[nq:2*nq, :nq].transpose()
        # A_22_T = self.Ac_hat[nq:2*nq, nq:2*nq].transpose()
        A_11_T = self.Ac_hat[:freeDOF, :freeDOF].transpose()
        A_12_T = self.Ac_hat[:freeDOF, freeDOF:self.nn].transpose()
        A_21_T = self.Ac_hat[freeDOF:self.nn, :freeDOF].transpose()
        A_22_T = self.Ac_hat[freeDOF:self.nn, freeDOF:self.nn].transpose()

        G_11 = np.kron(A_11_T, np.eye(freeDOF))
        G_12 = np.kron(A_12_T, np.eye(freeDOF))
        G_21 = np.kron(A_21_T, np.eye(freeDOF))
        G_22 = np.kron(A_22_T, np.eye(freeDOF))

        if self.outputType == OutputType.Disp:
            # G_d = np.kron(np.eye(nq), self.C_loc)
            G_d = np.kron(np.eye(freeDOF), self.C_loc)
            G_cd = np.vstack((np.hstack((G_d, np.zeros(G_d.shape), np.zeros(G_d.shape), np.zeros(G_d.shape))),\
                              np.hstack((np.zeros(G_d.shape), G_d, np.zeros(G_d.shape), np.zeros(G_d.shape))),\
                              np.hstack((G_11, G_21, -np.eye(G_11.shape[0]), np.zeros(G_21.shape))),\
                              np.hstack((G_12, G_22, np.zeros(G_12.shape), -np.eye(G_22.shape[0])))))
            vec_Cc_hat = self.Cc_hat.reshape(self.Cc_hat.shape[0]*self.Cc_hat.shape[1], order='F')
            d_d = np.hstack((vec_Cc_hat, np.zeros((G_11.shape[0]*2))))
            G_c = G_cd
            dd = d_d

        elif self.outputType == OutputType.Vel:
            # G_v = np.kron(np.eye(nq), self.C_loc)
            G_v = np.kron(np.eye(freeDOF), self.C_loc)
            G_cv = np.vstack((np.hstack((np.zeros(G_v.shape), np.zeros(G_v.shape), G_v, np.zeros(G_v.shape))),\
                              np.hstack((np.zeros(G_v.shape), np.zeros(G_v.shape), np.zeros(G_v.shape), G_v)),\
                              np.hstack((G_11, G_21, -np.eye(G_11.shape[0]), np.zeros(G_21.shape))),\
                              np.hstack((G_12, G_22, np.zeros(G_12.shape), -np.eye(G_22.shape[0])))))
            vec_Cc_hat = self.Cc_hat.reshape(self.Cc_hat.shape[0]*self.Cc_hat.shape[1], order='F')
            d_v = np.hstack((vec_Cc_hat, np.zeros((G_11.shape[0]*2))))
            G_c = G_cv
            dd = d_v

        elif self.outputType == OutputType.Acc:
            Ga_11 = np.kron(A_11_T, self.C_loc)
            Ga_12 = np.kron(A_12_T, self.C_loc)
            Ga_21 = np.kron(A_21_T, self.C_loc)
            Ga_22 = np.kron(A_22_T, self.C_loc)
            G_ca = np.vstack((np.hstack((np.zeros(Ga_11.shape), np.zeros(Ga_21.shape), Ga_11, Ga_21)),\
                              np.hstack((np.zeros(Ga_12.shape), np.zeros(Ga_22.shape), Ga_12, Ga_22)),\
                              np.hstack((G_11, G_21, -np.eye(G_11.shape[0]), np.zeros(G_21.shape))),\
                              np.hstack((G_12, G_22, np.zeros(G_12.shape), -np.eye(G_22.shape[0])))))
            vec_Cc_hat = self.Cc_hat.reshape(self.Cc_hat.shape[0]*self.Cc_hat.shape[1], order='F')
            d_a = np.hstack((vec_Cc_hat, np.zeros((G_11.shape[0]*2))))
            G_c = G_ca
            dd = d_a

        if self.pp == freeDOF:
            print("All DOFs are measured.")

            Tau = np.linalg.solve(G_c, dd)
            Ts_tmp = Tau.reshape(4, freeDOF, freeDOF)
            Ts = np.vstack((np.hstack((Ts_tmp[0].transpose(), Ts_tmp[1].transpose())),\
                            np.hstack((Ts_tmp[2].transpose(), Ts_tmp[3].transpose()))))

            Aest = np.dot(np.dot(Ts, self.Ac_hat), np.linalg.inv(Ts))
            Cest = np.dot(self.Cc_hat, np.linalg.inv(Ts))
            H1 = -Aest[freeDOF:2*freeDOF,:freeDOF]
            H2 = -Aest[freeDOF:2*freeDOF,freeDOF:2*freeDOF]
            MH1 = np.dot(matrix.M_ff, H1)  # Estimated Stiffness Matrix (not symmetric)
            MH2 = np.dot(matrix.M_ff, H2)  # Estimated Damping Matrix (not symmetric)

            # Estimate Second Moment of Inertia(I) from K_est
            Ie_est = self.EstimateElementProperty(MH1, matrix, mesh, material, assign)

            # Reconstruct Stiffness and Damping Matrices with Ie_est, which are symmetric.
            matrix_est = self.GetEstimatedStiffness(mesh, material, assign, boundary, analysis, Ie_est)
            K_est = matrix_est.K_ff

            # Estimate Damping Ratios with the Estimated Stiffness Matrix
            dmpRatio_est, C_est = self.GetEstimatedDamping(matrix_est, MH2, analysis)

        elif self.pp < freeDOF:
            print("Partial DOFs are measured.")

            G_mk = np.kron(np.eye(freeDOF), MinvKp)
            G_mc = np.kron(np.eye(freeDOF), MinvCp)

            G_e = np.vstack((np.hstack((G_mk, np.zeros(G_mk.shape), G_mc + G_11, G_21)),\
                             np.hstack((np.zeros(G_mk.shape), G_mk, G_12, G_mc + G_22))))

            Gprior = np.vstack((np.hstack((np.dot(G_e.transpose(), G_e), G_c.transpose())),\
                                np.hstack((G_c, np.zeros((G_c.shape[0], G_c.shape[0]))))))
            Dprior = np.hstack((np.zeros((G_e.shape[1])), dd))

            Tau_prior = np.linalg.solve(Gprior, Dprior)
            Ts_tmp = Tau_prior[:G_e.shape[1]].reshape(4, freeDOF, freeDOF)
            Ts = np.vstack((np.hstack((Ts_tmp[0].transpose(), Ts_tmp[1].transpose())),\
                            np.hstack((Ts_tmp[2].transpose(), Ts_tmp[3].transpose()))))

            Aest = np.dot(np.dot(Ts, self.Ac_hat), np.linalg.inv(Ts))  # Estimated System Matrix A
            Cest = np.dot(self.Cc_hat, np.linalg.inv(Ts))  # Estimated System Matrix C
            H1 = -Aest[freeDOF:2*freeDOF,:freeDOF]
            H2 = -Aest[freeDOF:2*freeDOF,freeDOF:2*freeDOF]
            MH1 = np.dot(matrix.M_ff, H1)  # Estimated Stiffness Matrix (not symmetric)
            MH2 = np.dot(matrix.M_ff, H2)  # Estimated Damping Matrix (not symmetric)

            # Estimate Second Moment of Inertia(I) from K_est
            Ie_est = self.EstimateElementProperty(MH1, matrix, mesh, material, assign)

            # Reconstruct Stiffness and Damping Matrices with Ie_est, which are symmetric.
            matrix_est = self.GetEstimatedStiffness(mesh, material, assign, boundary, analysis, Ie_est)
            K_est = matrix_est.K_ff
            modeShape_est = matrix_est.modeShapes

            # Estimate Damping Ratios with the Estimated Stiffness Matrix
            dmpRatio_est, C_est = self.GetEstimatedDamping(matrix_est, MH2, analysis)

            if self.PriorUpdate:
                self.Kprior = K_est
                self.Cprior = C_est
                # self.Cprior = MH2
                self.modeShapesKp = modeShape_est

        self.timeManager.EndTimer("Similarity Transformation")

        return Ie_est, dmpRatio_est
    

    def GetEstimatedStiffness(self, mesh, material, assign, boundary, analysis, Ie_est):
        # Add New Property with the Estimated Properties
        prop_est = cp.deepcopy(assign.properties[self.priorAssignName])
        for ei, elemProp in prop_est.elemProps.items():
            elemProp.secondMoment = Ie_est[mesh.elemIndex[np.uint32(ei)]]
        assign.AddProperty("Prop_est", prop_est)

        analysis_est = Analysis()
        analysis_est.SetAssigns("Prop_est", self.priorBCList)

        matrix_est = Matrix(mesh, material, assign, boundary, analysis_est)
        matrix_est.CreateIncidence()  # Create Incidence Vector
        matrix_est.CreateGlobals()  # Create Global Stiffness & Mass Matrices

        return matrix_est

    def GetPriorMatrices(self, mesh, material, assign, boundary):
        # Construct Stiffness and Damping Matrices with Prior Information
        analysisPrior = Analysis()
        analysisPrior.SetAssigns(self.priorAssignName, self.priorBCList)

        matrix_p = Matrix(mesh, material, assign, boundary, analysisPrior)
        matrix_p.CreateIncidence()  # Create Incidence Vector
        matrix_p.CreateGlobals()  # Create Global Stiffness & Mass Matrices

        self.Kprior = matrix_p.K_ff
        self.Cprior = matrix_p.C_ff

        self.modeShapesKp = matrix_p.modeShapes


    def GetEstimatedDamping(self, matrix_est, MH2, analysis):
        matPI = np.zeros((matrix_est.freeDOF, matrix_est.freeDOF), dtype=np.float64)
        vecC = np.zeros((matrix_est.freeDOF,), dtype=np.float64)

        natFreq, modeShapes = analysis.RunModalAnalysis(matrix_est)  # Calculate Natural Frequencies and Mode Shapes with Estimate Stiffness Matrix and Mass Matrix

        for jMode in np.arange(matrix_est.freeDOF):
            c_j = 2.0*(2.0*np.pi*natFreq[jMode])*np.dot(modeShapes[:,[jMode]], modeShapes[:,[jMode]].transpose())
            c_j = np.dot(np.dot(matrix_est.M_ff, c_j), matrix_est.M_ff)

            for iMode in np.arange(matrix_est.freeDOF):
                c_i = 2.0*(2.0*np.pi*natFreq[iMode])*np.dot(modeShapes[:,[iMode]], modeShapes[:,[iMode]].transpose())
                c_i = np.dot(np.dot(matrix_est.M_ff, c_i), matrix_est.M_ff)

                matPI[jMode, iMode] = np.sum(c_j*c_i)

            vecC[jMode] = np.sum(c_j*MH2)

        dmpRatio_est = np.linalg.solve(matPI, vecC)

        C_est = np.zeros((matrix_est.freeDOF, matrix_est.freeDOF), dtype = np.float64)
        for i in np.arange(matrix_est.freeDOF):
            C_est = C_est + 2.0*dmpRatio_est[i]*(2.0*np.pi*natFreq[i])*np.dot(modeShapes[:,[i]], modeShapes[:,[i]].transpose())
        C_est = np.dot(np.dot(matrix_est.M_ff, C_est), matrix_est.M_ff)

        return dmpRatio_est, C_est


    def GetInputOutput(self, matrix, mesh, analysis, result):
        # Get Input Mask
        self.uMask = []
        for inputDOF in self.usedInputDOFs:
            gDOF_i = matrix.incid[mesh.nodeIndex[inputDOF[0]]][inputDOF[1]-1]
            self.uMask.append(gDOF_i)
        self.uMask = np.array(self.uMask, dtype = np.uint32)
        self.u_m = analysis.U[:,self.uMask]
        self.mm = len(self.uMask)

        # Get Output Mask
        self.yMask = []
        for outputDOF in self.usedOutputDOFs:
            gDOF_j = matrix.incid[mesh.nodeIndex[outputDOF[0]]][outputDOF[1]-1]
            self.yMask.append(gDOF_j)
        self.yMask = np.array(self.yMask, dtype = np.uint32)
        
        if self.outputType == OutputType.Disp:
            self.y_p = result.dynamic.disp[:,self.yMask]
        elif self.outputType == OutputType.Vel:
            self.y_p = result.dynamic.vel[:,self.yMask]
        elif self.outputType == OutputType.Acc:
            self.y_p = result.dynamic.acc[:,self.yMask]
        self.pp = len(self.yMask)

        # Add Artificial Measurement Noise
        np.random.seed(self.seedNumber)
        noiseMat = 2.0 * (np.random.rand(self.y_p.shape[0], self.y_p.shape[1]) - 0.5)
        self.y_p = self.y_p + noiseMat * self.noiseMagnitude * self.y_p

        # If the time step of Subspace SI is different from one of the analysis, inter/extrapolation is required here
        # u_m -> return u_m, y_p -> return y_p


    def RunSubspaceSI(self, mesh, material, assign, boundary, analysis, matrix, result):
        self.timeManager = fIO.TimeManager()
        self.timeManager.StartTimer("SubspaceSI")

        # Get Prior Stiffness and Damping Matrices
        self.GetPriorMatrices(mesh, material, assign, boundary)  

        # Get Input and Output Vectors
        self.GetInputOutput(matrix, mesh, analysis, result)

        self.N_SI = np.uint32((self.Tf - self.Ti) / self.Tstride) + 1
        stepSI = np.uint32(self.Tstride / self.dT)

        # Declare variables to save the estimated values
        self.secondMoment = np.zeros((self.N_SI, mesh.elemCount), dtype=np.float64)
        self.dmpRatio = np.zeros((self.N_SI, matrix.freeDOF), dtype=np.float64)
        self.singularValue = []
        self.eigvalAc = []

        for i_SI in np.arange(self.N_SI):
            self.timeManager.WriteSIStep(i_SI, self.N_SI)  # Print & Write SI Step
            self.idxTi = np.uint32(self.Ti / self.dT) + i_SI * stepSI
            self.ConstructHankel(mesh, matrix, analysis, result)
            self.EstimatedSystemMatrices()
            Ie_est, dmpRatio_est = self.ArbitraryToPhysical(matrix, mesh, material, assign, boundary, analysis)

            # Save Data
            self.secondMoment[i_SI,:] = Ie_est
            self.dmpRatio[i_SI,:] = dmpRatio_est
            self.singularValue.append(self.SS)

        # Convert to Numpy array
        self.singularValue = np.array(self.singularValue, dtype=np.float64)  
        self.eigvalAc = np.array(self.eigvalAc, dtype=np.complex128)

        self.timeManager.EndTimer("SubspaceSI")

 
    def PlotEstimatedResult(self):
        self.PlotSingularValue()
        self.PlotEstimatedHistory()


    def PlotEstimatedHistory(self):
        # Estimated Second Moments of Inertia
        plt.figure(figsize = (10, 8))
        plt.title("Estimated Parameters (Second Moment of Inertia)", loc='center')

        minSecondMoment = np.amin(self.secondMoment)
        maxSecondMoment = np.amax(self.secondMoment)
        if np.isinf(minSecondMoment) or np.isnan(minSecondMoment) or np.isinf(maxSecondMoment) or np.isnan(maxSecondMoment):
            print("Estimated second moment has NaN or Inf value.")
        else:
            for iElem in range(self.secondMoment.shape[1]):
                label = 'Element ' + str(iElem+1)
                plt.plot(np.arange(1, self.N_SI+1), self.secondMoment[:, iElem], label=label)
            plt.legend(loc='lower right', framealpha=0.5)
            plt.xlim(1, self.N_SI)
            plt.ylim(minSecondMoment - 0.1*(maxSecondMoment-minSecondMoment), maxSecondMoment + 0.1*(maxSecondMoment-minSecondMoment))
            plt.xlabel("SI Step")
            plt.ylabel("Estimated Second Moment (m^4)")
            plt.show()

        # Estimated Damping Ratios
        plt.figure(figsize = (10, 8))
        plt.title("Estimated Parameters (Damping Ratio)", loc='center')

        minDmpRatio = np.amin(self.dmpRatio)
        maxDmpRatio = np.amax(self.dmpRatio)
        if np.isinf(minDmpRatio) or np.isnan(minDmpRatio) or np.isinf(maxDmpRatio) or np.isnan(maxDmpRatio):
            print("Estimated damping ratio has NaN or Inf value.")
        else:
            for iMode in range(self.dmpRatio.shape[1]):
                label = 'Mode ' + str(iMode+1)
                plt.plot(np.arange(1, self.N_SI+1), self.dmpRatio[:, iMode], label=label)
            plt.legend(loc='lower right', framealpha=0.5)
            plt.xlim(1, self.N_SI)
            plt.ylim(minDmpRatio - 0.1*(maxDmpRatio-minDmpRatio), maxDmpRatio + 0.1*(maxDmpRatio-minDmpRatio))
            plt.xlabel("SI Step")
            plt.ylabel("Estimated Damping Ratio")
            plt.show()


    def PlotSingularValue(self, withSIStep=False):
        # Singular Values with SI Step
        if withSIStep:
            plt.figure(figsize = (10, 8))
            plt.title("Singular Values of L22", loc='center')

            for i_SV in range(self.singularValue.shape[1]):
                label = str(i_SV+1) + 'th Singular Value'
                plt.plot(np.arange(1, self.N_SI+1), self.singularValue[:, i_SV], label=label)
            plt.legend(loc='lower right', framealpha=0.5)
            plt.xlim(0, self.N_SI)
            plt.ylim(0.5*np.amin(self.singularValue[:, self.singularValue.shape[1]-1]), 2.*np.amax(self.singularValue[:, 0]))
            plt.yscale('log')
            plt.ylabel("Singular Value")

            plt.show()

        nRow = 3
        nCol = 3
        for i_SI in np.arange(0, self.singularValue.shape[0], nRow*nCol):
            plt.figure(figsize = (14, 9))
            plt.title("Singular Values of L22", loc='center')
            for iRow in np.arange(nRow):
                for iCol in np.arange(nCol):
                    index = iRow*nCol+iCol+i_SI
                    if index < self.singularValue.shape[0]:
                        plt.subplot(nRow, nCol, iRow*nCol+iCol+1)
                        plt.plot(np.arange(1, self.singularValue.shape[1]+1), self.singularValue[index,:], marker='o')
                        plt.title("SI Step:" + str(index+1))
                        plt.xlim(1, self.singularValue.shape[1])
                        plt.yscale('log')
                        plt.ylabel("Singular Value")
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.show()


    def WriteEstimatedResult(self):
        fn.WriteMatrix("Res_EigenvalueAc.out", self.eigvalAc, isNew=False)
        fn.WriteMatrix("Res_SecondMoment.out", self.secondMoment)
        fn.WriteMatrix("Res_DampingRatio.out", self.dmpRatio)
        fn.WriteMatrix("Res_SingularValue.out", self.singularValue)

