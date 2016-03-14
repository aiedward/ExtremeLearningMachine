import random as rand
import activation_functions as af
import numpy as np


class FeedforwardSingleNeuralNetwork:

	def __init__(self, nInNeurons, nHiddenNeurons, nOutNeurons):

		self.nInNeurons = nInNeurons + 1 #Plus 1 for including bias neuron
		self.nHiddenNeurons = nHiddenNeurons
		self.nOutNeurons = nOutNeurons
		

		#List of input weights randomly initialised
		self.inWeights = [[rand.uniform(0,1) for _ in range(self.nHiddenNeurons)] for _ in range(self.nInNeurons)]
		
	def feedForward(self, dataset):
		
		inNeuronActivation = [[1]*self.nInNeurons for _ in range(len(dataset))]	
		hiddenNeuronActivation = [[1]*self.nHiddenNeurons for _ in range(len(dataset))]

		for sample in range(len(dataset)):
			#Except bias neuron
			for i in range(self.nInNeurons - 1):
				inNeuronActivation[sample][i] = dataset[sample][i]


		for sample in range(len(dataset)):
			for hNeuron in range(self.nHiddenNeurons):
				sumTotal = 0.0
				for inNeuron in range(self.nInNeurons):
					sumTotal += (inNeuronActivation[sample][inNeuron] * self.inWeights[inNeuron][hNeuron])
								
				hiddenNeuronActivation[sample][hNeuron] = self.activationFunction(sumTotal)

		return hiddenNeuronActivation		

	def train(self, trainSet, labelSet):

		hiddenNeuronActivation = self.feedForward(trainSet)	
		
		self.outWeights = np.transpose(np.dot(np.linalg.pinv(hiddenNeuronActivation), np.transpose(labelSet)))

		

	def activationFunction(self, data):
		return af.sigmoid(data)		

	def predict(self, testSet):
		
		hiddenNeuronActivation = self.feedForward(testSet)

		outNeuronActivation = [1]*self.nOutNeurons
		
		for outNeuron in range(self.nOutNeurons):
			sumTotal = 0.0
			for hNeuron in range(self.nHiddenNeurons):
				sumTotal += (hiddenNeuronActivation[outNeuron][hNeuron] * self.outWeights[hNeuron])
			
		outNeuronActivation[outNeuron] = self.activationFunction(sumTotal)

		return outNeuronActivation	
