import Functions as fn
import AnalysisManager as am
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rcParams


class ModalResult:
    def __init__(self, modeCount):
        self.modeCount = modeCount

    def SetResults(self, natFreq, modeShapes):
        self.natFreq = natFreq[:self.modeCount]
        self.modeShapes = modeShapes[:,:self.modeCount]

    def PrintResults(self):
        print("--- Natural Frequencies(Hz) ---")
        fn.ShowMatrix(self.natFreq)
        print("--- Mode Shapes ---")
        fn.ShowMatrix(self.modeShapes)

    def GetModeCoords(self, mesh, matrix, sFactor, nDivide):
        self.modeCoords = np.zeros((self.modeCount, mesh.nodeCoords.shape[0], mesh.nodeCoords.shape[1]), dtype = np.float64)
        for mode in range(self.modeCount):
            for node in range(mesh.nodeCount):
                fixed = matrix.descDOFs[node][0]
                iDOF = matrix.incid[node][0]
                if fixed == 0:
                    self.modeCoords[mode][node][0] = mesh.nodeCoords[node][0]
                    self.modeCoords[mode][node][1] = mesh.nodeCoords[node][1] + sFactor * self.modeShapes[iDOF][mode]
                    self.modeCoords[mode][node][2] = mesh.nodeCoords[node][2]
                elif fixed == 1:
                    self.modeCoords[mode][node][0] = mesh.nodeCoords[node][0]
                    self.modeCoords[mode][node][1] = mesh.nodeCoords[node][1]
                    self.modeCoords[mode][node][2] = mesh.nodeCoords[node][2]

    def PlotModeCoords(self):
        for mode in range(self.modeCount):
            plt.plot(self.modeCoords[mode,:,:1], self.modeCoords[mode,:,1:2])
        plt.show()


class StaticResult:
    def SetResults(self, disp):
        self.disp = disp
    
    def PrintResults(self):
        print("--- Displacement(m) ---")
        fn.ShowMatrix(self.disp)

    def GetDeformedCoords(self, mesh, matrix, sFactor, nDivide):
        self.deformedCoords = np.zeros(\
            (mesh.nodeCoords.shape[0] + mesh.elemCount*(nDivide-1), mesh.nodeCoords.shape[1]), dtype = np.float64)

        # Get Deformed Coordinates at Nodes
        for node in range(mesh.nodeCount):
            fixed = matrix.descDOFs[node][0]
            iDOF = matrix.incid[node][0]
            if fixed == 0:
                self.deformedCoords[node*nDivide][0] = mesh.nodeCoords[node][0]
                self.deformedCoords[node*nDivide][1] = mesh.nodeCoords[node][1] + sFactor * self.disp[iDOF]
                self.deformedCoords[node*nDivide][2] = mesh.nodeCoords[node][2]
            elif fixed == 1:
                self.deformedCoords[node*nDivide][0] = mesh.nodeCoords[node][0]
                self.deformedCoords[node*nDivide][1] = mesh.nodeCoords[node][1]
                self.deformedCoords[node*nDivide][2] = mesh.nodeCoords[node][2]

        # Get Deformed Coordinates along Element
        for ei in range(mesh.elemCount):
            eleNodes = mesh.elementNodes[ei]
            eleNodeCount = len(eleNodes)
            nodeDOF = 2

            eleDisp = np.zeros((eleNodeCount*nodeDOF,), dtype = np.float64)
            for iNode in np.arange(eleNodeCount):
                for iDOF in np.arange(nodeDOF):
                    fixed = matrix.descDOFs[mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                    dof = matrix.incid[mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                    if fixed == 0:
                        eleDisp[iNode*nodeDOF+iDOF] = self.disp[dof]
                    elif fixed == 1:
                        eleDisp[iNode*nodeDOF+iDOF] = 0.

            shapeFn = np.zeros((nDivide-1, eleNodeCount*nodeDOF), dtype = np.float64)
            ncoord = np.zeros((nDivide-1, mesh.nodeCoords.shape[1]), dtype = np.float64)
            n1coord = mesh.nodeCoords[mesh.nodeIndex[eleNodes[0]]]
            n2coord = mesh.nodeCoords[mesh.nodeIndex[eleNodes[1]]]
            L1 = np.linalg.norm(n1coord - n2coord, ord = 2)
            for iDivide in np.arange(nDivide-1):
                xi = np.float64((iDivide+1)/nDivide)
                ncoord[iDivide] = n1coord + xi * (n2coord - n1coord)
                shapeFn[iDivide][0] = 1 - 3*xi**2 + 2*xi**3
                shapeFn[iDivide][1] = L1 * (xi - 2*xi**2 + xi**3)
                shapeFn[iDivide][2] = 3*xi**2 - 2*xi**3
                shapeFn[iDivide][3] = L1 * (-xi**2 + xi**3)
            disp = np.dot(shapeFn, eleDisp)

            for iDivide in np.arange(nDivide-1):
                self.deformedCoords[1+iDivide+ei*nDivide][0] = ncoord[iDivide][0]
                self.deformedCoords[1+iDivide+ei*nDivide][1] = ncoord[iDivide][1] + sFactor * disp[iDivide]

    def PlotDeformedCoords(self):
        plt.plot(self.deformedCoords[:,:1], self.deformedCoords[:,1:2])
        plt.show()


class DynamicResult:
    def SetResults(self, Ti, disp, vel, acc):
        self.Ti = Ti
        self.disp = disp
        self.vel = vel
        self.acc = acc

    def PrintResults(self):
        print("--- Displacement(m) ---")
        fn.ShowMatrix(self.disp)
        print("--- Velocity(m/sec) ---")
        fn.ShowMatrix(self.vel)
        print("--- Acceleration(m/sec^2) ---")
        fn.ShowMatrix(self.acc)

    def PlotTimeHistory(self, initDOF, finalDOF):
        for iDOF in range(initDOF-1, finalDOF):
            plt.figure(figsize = (10, 8))

            plt.subplot(3, 1, 1)
            plt.plot(self.Ti, self.disp[:,iDOF])
            plt.title("DOF:" + str(iDOF+1))
            plt.gca().set_yticklabels(['{:6.2e}'.format(x) for x in plt.gca().get_yticks()]) 
            plt.ylabel("Displacement(m)")

            plt.subplot(3, 1, 2)
            plt.plot(self.Ti, self.vel[:,iDOF])
            plt.gca().set_yticklabels(['{:6.2e}'.format(x) for x in plt.gca().get_yticks()]) 
            plt.ylabel("Velocity(m/sec)")

            plt.subplot(3, 1, 3)
            plt.plot(self.Ti, self.acc[:,iDOF])
            plt.gca().set_yticklabels(['{:6.2e}'.format(x) for x in plt.gca().get_yticks()]) 
            plt.xlabel("Time(sec)")
            plt.ylabel("Acceleration(m/sec^2)")

            plt.show()

    def GetDeformedCoords(self, mesh, matrix, sFactor, nDivide):
        self.deformedCoords = np.zeros(\
            (self.Ti.shape[0], mesh.nodeCoords.shape[0] + mesh.elemCount*(nDivide-1), mesh.nodeCoords.shape[1]), dtype = np.float64)

        for tStep in np.arange(self.Ti.shape[0]):
            # Print Current Step
            if (tStep+1)%1000 == 0:
                print("Current Step for Deformed Coordinates:", tStep+1, "/", self.Ti.shape[0])

            # Get Deformed Coordinates at Nodes
            for node in range(mesh.nodeCount):
                fixed = matrix.descDOFs[node][0]
                iDOF = matrix.incid[node][0]
                if fixed == 0:
                    self.deformedCoords[tStep][node*nDivide][0] = mesh.nodeCoords[node][0]
                    self.deformedCoords[tStep][node*nDivide][1] = mesh.nodeCoords[node][1] + sFactor * self.disp[tStep][iDOF]
                    self.deformedCoords[tStep][node*nDivide][2] = mesh.nodeCoords[node][2]
                elif fixed == 1:
                    self.deformedCoords[tStep][node*nDivide][0] = mesh.nodeCoords[node][0]
                    self.deformedCoords[tStep][node*nDivide][1] = mesh.nodeCoords[node][1]
                    self.deformedCoords[tStep][node*nDivide][2] = mesh.nodeCoords[node][2]

            # Get Deformed Coordinates along Element
            for ei in range(mesh.elemCount):
                eleNodes = mesh.elementNodes[ei]
                eleNodeCount = len(eleNodes)
                nodeDOF = 2

                eleDisp = np.zeros((eleNodeCount*nodeDOF,), dtype = np.float64)
                for iNode in np.arange(eleNodeCount):
                    for iDOF in np.arange(nodeDOF):
                        fixed = matrix.descDOFs[mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                        dof = matrix.incid[mesh.nodeIndex[eleNodes[iNode]]][iDOF]
                        if fixed == 0:
                            eleDisp[iNode*nodeDOF+iDOF] = self.disp[tStep][dof]
                        elif fixed == 1:
                            eleDisp[iNode*nodeDOF+iDOF] = 0.

                shapeFn = np.zeros((nDivide-1, eleNodeCount*nodeDOF), dtype = np.float64)
                ncoord = np.zeros((nDivide-1, mesh.nodeCoords.shape[1]), dtype = np.float64)
                n1coord = mesh.nodeCoords[mesh.nodeIndex[eleNodes[0]]]
                n2coord = mesh.nodeCoords[mesh.nodeIndex[eleNodes[1]]]
                L1 = np.linalg.norm(n1coord - n2coord, ord = 2)
                for iDivide in np.arange(nDivide-1):
                    xi = np.float64((iDivide+1)/nDivide)
                    ncoord[iDivide] = n1coord + xi * (n2coord - n1coord)
                    shapeFn[iDivide][0] = 1 - 3*xi**2 + 2*xi**3
                    shapeFn[iDivide][1] = L1 * (xi - 2*xi**2 + xi**3)
                    shapeFn[iDivide][2] = 3*xi**2 - 2*xi**3
                    shapeFn[iDivide][3] = L1 * (-xi**2 + xi**3)
                disp = np.dot(shapeFn, eleDisp)

                for iDivide in np.arange(nDivide-1):
                    self.deformedCoords[tStep][1+iDivide+ei*nDivide][0] = ncoord[iDivide][0]
                    self.deformedCoords[tStep][1+iDivide+ei*nDivide][1] = ncoord[iDivide][1] + sFactor * disp[iDivide]

    def PlotDeformedCoords(self):
        # for tStep in np.arange(self.Ti.shape[0]):
            # plt.plot(self.deformedCoords[tStep][:,:1], self.deformedCoords[tStep][:,1:2])
        plt.plot(self.deformedCoords[0][:,:1], self.deformedCoords[0][:,1:2])
        plt.show()

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        x = self.deformedCoords[i][:,:1]
        y = self.deformedCoords[i][:,1:2]
        self.line.set_data(x, y)
        self.time.set_text("Time = " + str("{:6.2f} sec").format(self.Ti[i]))

        return self.line, self.time,
    
    def AnimateDeformedShape(self, save, repeat):
        # Install Path of FFmpeg
        rcParams['animation.ffmpeg_path'] = r'D:\Utility\FFmpeg\FFmpeg\bin\ffmpeg.exe'

        fig, ax = plt.subplots()
        ax.set_xlim(( 0., 20.))
        ax.set_ylim((-1.E-3, 1.E-3))
        self.line, = ax.plot([], [], lw=2)

        # Display Current Time
        axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])
        axtext.axis("off")
        self.time = axtext.text(0.5,0.5, str(0), ha="left", va="top")

        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init, frames=self.Ti.shape[0], interval=10, blit=False, repeat=repeat)
        if save:
            anim.save('Vibration.gif', writer='imagemagick', fps=30)

        plt.show()


class Result:
    def __init__(self, anaType):
        self.anaType = anaType

    def SetResults(self, *results):
        if self.anaType == am.AnalysisType.LinearStatic:
            self.static = StaticResult()
            self.static.SetResults(results[0])
        elif self.anaType == am.AnalysisType.Modal:
            self.modal = ModalResult(len(results[0]))
            self.modal.SetResults(results[0], results[1])
        elif self.anaType == am.AnalysisType.Dynamic:
            self.dynamic = DynamicResult()
            self.dynamic.SetResults(results[0], results[1], results[2], results[3])

    def PrintResults(self):
        if self.anaType == am.AnalysisType.LinearStatic:
            self.static.PrintResults()
        elif self.anaType == am.AnalysisType.Modal:
            self.modal.PrintResults()
        elif self.anaType == am.AnalysisType.Dynamic:
            self.dynamic.PrintResults()

    def GetDeformedShape(self, mesh, matrix, sFactor = 1, nDivide = 10):
        if self.anaType == am.AnalysisType.LinearStatic:
            self.static.GetDeformedCoords(mesh, matrix, sFactor, nDivide)
        elif self.anaType == am.AnalysisType.Modal:
            self.modal.GetModeCoords(mesh, matrix, sFactor, nDivide)
        elif self.anaType == am.AnalysisType.Dynamic:
            self.dynamic.GetDeformedCoords(mesh, matrix, sFactor, nDivide)

    def PlotDeformedShape(self):
        if self.anaType == am.AnalysisType.LinearStatic:
            self.static.PlotDeformedCoords()
        elif self.anaType == am.AnalysisType.Modal:
            self.modal.PlotModeCoords()
        elif self.anaType == am.AnalysisType.Dynamic:
            self.dynamic.PlotDeformedCoords()

    def PlotTimeHistory(self, initDOF, finalDOF):
        if self.anaType == am.AnalysisType.Dynamic:
            self.dynamic.PlotTimeHistory(initDOF, finalDOF)

    def AnimateDeformedShape(self, save = False, repeat = False):
        if self.anaType == am.AnalysisType.Dynamic:
            self.dynamic.AnimateDeformedShape(save, repeat)