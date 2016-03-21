import numpy as np

class Neuron:

	def __init__(self, nInputs):
		self.nInputs = nInputs + 1
		#Generate random weights including bias neuron
		self.weight = np.random.rand(self.nInputs,)