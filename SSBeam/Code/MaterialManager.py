import numpy as np
# import cupy as np

class Elastic:
    def __init__(self):
        self.property = []
        # property[0] : Young's Modulus
        # property[1] : Poisson's Ratio
        # property[2] : Density

    def AddProperty(self, prop):
        self.property.append(np.array(prop, dtype = np.float64))

class Material:
    def __init__(self):
        self.elasticMats = dict()

    def AddElastic(self, name, elastic):
        self.elasticMats.update([(name, elastic)])

    