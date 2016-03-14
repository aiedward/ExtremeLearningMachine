import numpy as np

class Neuron:

	def __init__(self, nInputs):
		#Generate random weights including bias neuron
		self.weight = np.random.rand(nInputs + 1,)