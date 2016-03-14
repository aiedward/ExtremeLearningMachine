import random as rand
import activation_functions as af
import numpy as np
from NeuronLayer import NeuronLayer

class FeedforwardSingleNeuralNetwork:

	def __init__(self, nInNeurons, nHiddenNeurons, nOutNeurons):

		self.nInNeurons = nInNeurons + 1 #Plus 1 for including bias neuron
		self.nHiddenNeurons = nHiddenNeurons
		self.nOutNeurons = nOutNeurons
		
		self.hiddenLayer = NeuronLayer(self.nHiddenNeurons, nInNeurons)
		self.outLayer = NeuronLayer(self.nOutNeurons, self.hiddenLayer.nNeurons)

		
	def feedForward(self, dataset):
		
		
		hiddenNeuronActivation = np.array(len(dataset), self.nHiddenNeurons)
		
		for sample in range(len(dataset)):
			for hNeuron in range(self.nHiddenNeurons):
				activation = 0.0
				for inNeuron in range(self.nInNeurons):
					activation += (dataset[sample][inNeuron] * self.hiddenLayer.neurons[hNeuron].weight[inNeuron])
								
				hiddenNeuronActivation[sample][hNeuron] = self.activationFunction(activation)

		return hiddenNeuronActivation		

	def train(self, trainSet, labelSet):

		hiddenNeuronActivation = self.feedForward(trainSet)	
		
		self.outWeights = np.dot(np.linalg.pinv(hiddenNeuronActivation), labelSet)

		

	def activationFunction(self, data):
		return af.sigmoid(data)		

	def predict(self, testSet):
		
		print self.outWeights
		print np.shape(self.outWeights)

		hiddenNeuronActivation = self.feedForward(testSet)
		return np.dot(self.outWeights, hiddenNeuronActivation)
#		outNeuronActivation = [1]*self.nOutNeurons
		
#		for outNeuron in range(self.nOutNeurons):
#			sumTotal = 0.0
#			for hNeuron in range(self.nHiddenNeurons):
#				sumTotal += (hiddenNeuronActivation[outNeuron][hNeuron] * self.outWeights[hNeuron])
			
#		outNeuronActivation[outNeuron] = self.activationFunction(sumTotal)

#		return outNeuronActivation	
