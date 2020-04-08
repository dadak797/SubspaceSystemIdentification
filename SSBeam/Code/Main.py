from FileIO import *
from Matrix import *
import Functions as fn
from ResultManager import *
from SubspaceSI import *


## Read Input File (Return: Mesh, Material, Assign, Boundary, Analysis) ##
mesh, material, assign, boundary, analysis, subspaceSI = ReadInputFile()

## Create Stiffness & Mass Matrices ##
matrix = Matrix(mesh, material, assign, boundary, analysis)
matrix.CreateIncidence()  # Create Incidence Vector
matrix.CreateGlobals()  # Create Global Stiffness & Mass Matrices

## Analysis ##
result = analysis.RunAnalysis(matrix, boundary)
if analysis.anaType == AnalysisType.Modal:
    result.PrintResults()
else:
    pass
# result.PlotTimeHistory(5, 5) # result.PlotTimeHistory(10, 10)
# result.GetDeformedShape(mesh, matrix, nDivide = 6)
# result.PlotDeformedShape()
# result.AnimateDeformedShape(repeat = True)

## Subspace System Identification ##
subspaceSI.RunSubspaceSI(mesh, material, assign, boundary, analysis, matrix, result)
subspaceSI.WriteEstimatedResult()
subspaceSI.PlotEstimatedResult()






