import numpy as np
from Neuron import Neuron

class NeuronLayer:

	def __init__(self, nNeurons, nInputEachNeuron):
		self.nNeurons = nNeurons
		self.neurons = np.array([Neuron(nInputEachNeuron)] for _ in range(nNeurons))