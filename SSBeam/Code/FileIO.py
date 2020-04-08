import sys
from MeshManager import *
import numpy as np
# import cupy as np
from enum import Enum
from MaterialManager import *
from PropertyManager import *
from BoundaryManager import *
from AnalysisManager import *
from SubspaceSI import *
import time


class Keyword(Enum):
    NODE = "#NODE"
    ELEMENT = "#ELEMENT"
    MATERIAL = "#MATERIAL"
    ASSIGN = "#ASSIGN"
    BOUNDARY = "#BOUNDARY"
    CLOAD = "#CLOAD"
    CMASS = "#CMASS"
    INITIAL = "#INITIAL"
    TIMESTEP = "#TIMESTEP"
    DAMPING = "#DAMPING"
    HARMONIC = "#HARMONIC"
    ANALYSIS = "#ANALYSIS"
    SUBSPACESI = "#SUBSPACESI"
    SUBSPACESI_ITERATIVE = "#SUBSPACESI_ITERATIVE"
    SUBSPACESI_USEDINPUT = "#SUBSPACESI_USEDINPUT"
    SUBSPACESI_USEDOUTPUT = "#SUBSPACESI_USEDOUTPUT"
    SUBSPACESI_PRIOR_INFO = "#SUBSPACESI_PRIOR_INFO"
    SUBSPACESI_ARTIFICIAL_NOISE = "#SUBSPACESI_ARTIFICIAL_NOISE"

    @classmethod
    def IsKeyword(self, word):
        if word == self.NODE.value or \
            word == self.ELEMENT.value or \
            word == self.MATERIAL.value or \
            word == self.ASSIGN.value or \
            word == self.BOUNDARY.value or \
            word == self.CLOAD.value or \
            word == self.CMASS.value or \
            word == self.INITIAL.value or \
            word == self.TIMESTEP.value or \
            word == self.DAMPING.value or \
            word == self.HARMONIC.value or \
            word == self.ANALYSIS.value or \
            word == self.SUBSPACESI.value or \
            word == self.SUBSPACESI_USEDINPUT.value or \
            word == self.SUBSPACESI_USEDOUTPUT.value or \
            word == self.SUBSPACESI_PRIOR_INFO.value or \
            word == self.SUBSPACESI_ITERATIVE.value or \
            word == self.SUBSPACESI_ARTIFICIAL_NOISE.value:
            return True
        else:
            return False


def GetOptions(opts):
    options = dict()
    for option in opts:
        tmp = option.strip().split('=')
        if len(tmp) != 2:
            print("Option Format Error")
            sys.exit()
        options.update([(tmp[0].strip(), tmp[1].strip())])
    return options


def ReadNode(f, mesh, options):
    f.readline()

    # Read Node Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            mesh.AppendNode(word[0].strip(), [x.strip() for x in word[1:]])
        else:
            break

    # Create Node Set
    if options is not None:
        nodeIDs = IDSet()
        for nodeID in mesh.nodeIndex.keys():
            nodeIDs.AddID(nodeID)
        mesh.CreateNodeSet(options["NSET"], nodeIDs.IDList)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadElement(f, mesh, options):
    f.readline()

    # Read Element Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            mesh.AppendElement(word[0].strip(), [x.strip() for x in word[1:]])
        else:
            break
        
    if options is not None:
        elementIDs = IDSet()
        for elementID in mesh.elemIndex.keys():
            elementIDs.AddID(elementID)
        mesh.CreateElementSet(options["ELSET"], elementIDs.IDList)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadMaterial(f, material, options):
    f.readline()

    # Read Material Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            elastic = Elastic()
            for prop in word:
                elastic.AddProperty(prop.strip())
            material.AddElastic(options["NAME"], elastic)
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadProperty(f, assign, options):
    f.readline()

    # Read Assign Information'
    prop = Property()
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            prop.AddProperty(word[0].strip(), ElementProperty([x.strip() for x in word[1:]]))
        else:
            break
    assign.AddProperty(options["NAME"], prop)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadBoundary(f, boundary, options):
    f.readline()

    # Read Boundary Information
    dispBC = DispBC()
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            dispBC.AddDispBC(word[0].strip(), word[1].strip(), word[2].strip())
        else:
            break
    boundary.AddDispBC(options["NAME"], dispBC)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadNodalForce(f, boundary, options):
    f.readline()

    # Read Concentrated Load Information
    nodalForce = NodalForce()
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            nodalForce.AddNodalForce(word[0].strip(), word[1].strip(), word[2].strip())
        else:
            break
    boundary.AddNodalForce(options["NAME"], nodalForce)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadConcentratedMass(f, assign, options):
    f.readline()

    # Read Concentrated Load Information
    cMass = CMass()
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            cMass.AddCMass(word[0].strip(), word[1].strip())
        else:
            break
    assign.AddCMass(options["NAME"], cMass)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadInitialCondition(f, boundary, options):
    f.readline()

    # Read Initial Condition Information
    if options["TYPE"] == "ByForce":
        nodalForce = NodalForce()
    elif options["TYPE"] == "ByDisp":
        pass
    elif options["TYPE"] == "ByMode":
        dispModal = DispModal()

    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            if options["TYPE"] == "ByForce":
                nodalForce.AddNodalForce(word[0].strip(), word[1].strip(), word[2].strip())
            elif options["TYPE"] == "ByDisp":
                pass
            elif options["TYPE"] == "ByMode":
                dispModal.AddDispModal(word[0].strip(), word[1].strip())
        else:
            break

    if options["TYPE"] == "ByForce":            
        boundary.AddNodalForce(options["NAME"], nodalForce)
    elif options["TYPE"] == "ByDisp":
        pass
    elif options["TYPE"] == "ByMode":
        boundary.AddDispModal(options["NAME"], dispModal)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options    


def ReadTimeStep(f, analysis, options):
    f.readline()

    # Read Analysis Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            analysis.SetTimeStep([x.strip() for x in word])
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options   


def ReadDampingRatio(f, assign, options):
    f.readline()

    # Read Damping Information
    if options["TYPE"] == "Rayleigh":
        damping = Damping(DampingType.Rayleigh)
    elif options["TYPE"] == "Modal":
        damping = Damping(DampingType.Modal)

    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            if options["TYPE"] == "Rayleigh":
                damping.AddRayleighCoeffs(word[0].strip(), word[1].strip())
            elif options["TYPE"] == "Modal":
                damping.AddDampingRatio(word[0].strip(), word[1].strip())
        else:
            break
    assign.AddDamping(options["NAME"], damping)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options  


def ReadHarmonic(f, boundary, options):
    f.readline()

    # Read Harmonic Information
    harmonic = HarmonicForce()

    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            harmonic.AddHarmonicForce(word[0].strip(), word[1].strip(), word[2].strip())
        else:
            break

    # Add Harmonic Force to Boundary
    boundary.AddHarmonicForce(options["NAME"], harmonic)

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options 


def ReadAnalysisSetting(f, analysis, options):
    f.readline()

    # Read Analysis Information
    analysis.SetLoadType(options["LOADTYPE"])

    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            analysis.SetAnalysisType(word[0].strip())
            analysis.SetAssigns(word[1].strip(), [x.strip() for x in word[2:]])
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadSubspaceSI(f, subspaceSI, options):
    f.readline()

    # Read Subspace System Identification Information
    subspaceSI.SetLoadType(options["LOADTYPE"])
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.SetHankelMatrixParams(word[0].strip(), word[1].strip(), word[2].strip())
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadSubspaceSIIterative(f, subspaceSI, options):
    f.readline()

    # Read Prior Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.SetTimeParams(word[0].strip(), word[1].strip(), word[2].strip(), word[3].strip())
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None
    return keyword, options        


def ReadSubspaceSIUsedInput(f, subspaceSI, options):
    f.readline()

    # Read Used Input Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.AddUsedInputDOF(word[0].strip(), word[1].strip())
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadSubspaceSIUsedOutput(f, subspaceSI, options):
    f.readline()

    # Read Used Output Information
    subspaceSI.SetOutputType(options["TYPE"])

    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.AddUsedOutputDOF(word[0].strip(), word[1].strip())
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None

    return keyword, options


def ReadSubspaceSIPriorInfo(f, subspaceSI, options):
    f.readline()

    # Read Prior Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.SetPriorInfo(word[0].strip(), [x.strip() for x in word[1:]])
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None
    return keyword, options


def ReadSubspaceSIArtificialNoise(f, subspaceSI, options):
    f.readline()

    # Read Prior Information
    while True:
        line = f.readline()
        if not line: 
            word = [] 
            break

        word = line.split(',')
        keyword = word[0].strip()

        if not Keyword.IsKeyword(keyword):
            subspaceSI.SetMeasurementNoiseInfo(word[0].strip(), word[1].strip())
        else:
            break

    # Get Options after Keyword
    if len(word) > 1:
        options = GetOptions(word[1:])
    else:
        options = None
    return keyword, options


def ReadInputFile():
    # Check Invalid Command (python *.py *.inp)
    if len(sys.argv) == 1:
        print("You need a input file name! Please enter the name:")
        fileName = input().strip()
    elif len(sys.argv) == 2:
        fileName = sys.argv[1]

    # Check Invalid Input File Name
    try:
        f = open(fileName, 'r')
    except FileNotFoundError as e:
        print(e)

    timeManager = TimeManager(isNew=True)
    timeManager.StartTimer("Reading Input")

    # Declare Mesh Model, Material, Assign, Boundary Conditions, Analysis Setting, Subspace System Identification
    mesh = Mesh()
    material = Material()
    assign = Assign()
    boundary = Boundary()
    analysis = Analysis()
    subspaceSI = SubspaceSI()

    while True:
        line = f.readline()
        if not line: break
        word = line.split(',')
        if len(word) > 1:
            options = GetOptions(word[1:])
        else:
            options = None
        keyword = word[0].strip()

        while keyword != None:
            if keyword == Keyword.NODE.value:
                keyword, options = ReadNode(f, mesh, options)
                print(" - Reading node information has done. %d nodes are defined." %mesh.nodeCount)
            elif keyword == Keyword.ELEMENT.value:
                keyword, options = ReadElement(f, mesh, options)
                print(" - Reading element information has done. %d elements are defined." %mesh.elemCount)
            elif keyword == Keyword.MATERIAL.value:
                keyword, options = ReadMaterial(f, material, options)
                print(" - Reading material information has done.")
            elif keyword == Keyword.ASSIGN.value:
                keyword, options = ReadProperty(f, assign, options)
                print(" - Reading assign information has done.")
            elif keyword == Keyword.BOUNDARY.value:
                keyword, options = ReadBoundary(f, boundary, options)
                print(" - Reading boundary information has done.")
            elif keyword == Keyword.CLOAD.value:
                keyword, options = ReadNodalForce(f, boundary, options)
                print(" - Reading nodal force information has done.")
            elif keyword == Keyword.CMASS.value:
                keyword, options = ReadConcentratedMass(f, assign, options)
                print(" - Reading concentrated mass information has done.")
            elif keyword == Keyword.INITIAL.value:
                keyword, options = ReadInitialCondition(f, boundary, options)
                print(" - Reading initial condition information has done.")
            elif keyword == Keyword.TIMESTEP.value:
                keyword, options = ReadTimeStep(f, analysis, options)
                print(" - Reading time-step information has done.")
            elif keyword == Keyword.DAMPING.value:
                keyword, options = ReadDampingRatio(f, assign, options)
                print(" - Reading damping information has done.")
            elif keyword == Keyword.HARMONIC.value:
                keyword, options = ReadHarmonic(f, boundary, options)
                print(" - Reading harmonic information has done.")
            elif keyword == Keyword.ANALYSIS.value:
                keyword, options = ReadAnalysisSetting(f, analysis, options)
                print(" - Reading analysis information has done.")
            elif keyword == Keyword.SUBSPACESI.value:
                keyword, options = ReadSubspaceSI(f, subspaceSI, options)
                print(" - Reading subspace SI information has done.")
            elif keyword == Keyword.SUBSPACESI_ITERATIVE.value:
                keyword, options = ReadSubspaceSIIterative(f, subspaceSI, options)
                print(" - Reading subspace SI information has done.")
            elif keyword == Keyword.SUBSPACESI_USEDINPUT.value:
                keyword, options = ReadSubspaceSIUsedInput(f, subspaceSI, options)
                print(" - Reading subspace SI input information has done.")
            elif keyword == Keyword.SUBSPACESI_USEDOUTPUT.value:
                keyword, options = ReadSubspaceSIUsedOutput(f, subspaceSI, options)
                print(" - Reading subspace SI output information has done.")
            elif keyword == Keyword.SUBSPACESI_PRIOR_INFO.value:
                keyword, options = ReadSubspaceSIPriorInfo(f, subspaceSI, options)
                print(" - Reading subspace SI prior information has done.")
            elif keyword == Keyword.SUBSPACESI_ARTIFICIAL_NOISE.value:
                keyword, options = ReadSubspaceSIArtificialNoise(f, subspaceSI, options)
                print(" - Reading subspace SI measurement noise information has done.")
            else:
                keyword = None
                break

    f.close()
    timeManager.EndTimer("Reading Input")

    # Convert List to Numpy Array
    mesh.ConvertToNumpyNodes()
    mesh.ConvertToNumpyElements()

    # Return Mesh Info., Material, BC Info., Analysis Setting
    return mesh, material, assign, boundary, analysis, subspaceSI


class TimeManager:
    def __init__(self, isNew=False):
        self.timer = dict()
        if isNew:
            self.fileTimeManager = open("Res_ElasedTime.out", 'w')
        else:
            self.fileTimeManager = open("Res_ElasedTime.out", 'a')  # 'a' is addition mode

    def StartTimer(self, timerName):
        self.timer[timerName] = time.time()
        print("-", timerName, "has started.")

    def EndTimer(self, timerName):
        print("-", timerName, "has done.")
        elaspedTime = time.time() - self.timer[timerName]
        strTime = "----- Time for " + timerName + ": " + '{:.4f}'.format(elaspedTime) + " sec -----"
        print(strTime)
        self.fileTimeManager.write(strTime+"\n")

    def WriteSIStep(self, currentStep, finalStep):
        strStep = "Current SI Step = " + str(currentStep+1) + " / " + str(finalStep)
        print(strStep)
        self.fileTimeManager.write("\n"+strStep+"\n")