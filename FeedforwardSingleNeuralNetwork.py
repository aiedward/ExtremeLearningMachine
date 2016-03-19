import random as rand
import activation_functions as af
import numpy as np
from NeuronLayer import NeuronLayer

class FeedforwardSingleNeuralNetwork:

	def __init__(self, nHiddenNeurons):
		
		self.nHiddenNeurons = nHiddenNeurons
				
	def feedForward(self, dataset):
		
		
		hiddenNeuronActivation = np.zeros((len(dataset), self.nHiddenNeurons))
		
		for sample in range(len(dataset)):
			for hNeuron in range(self.hiddenLayer.nNeurons):
				activation = 0.0
				for inNeuron in range(self.nInNeurons):
					activation += (dataset[sample][inNeuron] * self.hiddenLayer.neurons[hNeuron].weight[inNeuron])
				#Add bias weight ( w1x1 + w2x2 + (-1)bWeight)
				activation -= self.hiddenLayer.neurons[hNeuron].weight[self.nInNeurons]				
				hiddenNeuronActivation[sample][hNeuron] = self.activationFunction(activation)

		return hiddenNeuronActivation		

	def train(self, trainSet, labelSet):

		self.nInNeurons = len(trainSet[0])
		self.hiddenLayer = NeuronLayer(self.nHiddenNeurons, self.nInNeurons) 
		self.nOutNeurons = len(labelSet[0])
		self.outLayer = NeuronLayer(self.nOutNeurons, self.nHiddenNeurons)

		hiddenNeuronActivation = self.feedForward(trainSet)	
		
		outWeights = np.dot(np.linalg.pinv(hiddenNeuronActivation), np.array(labelSet))
		
		#Setar os pesos de saida na camada dos neurônios de saida 		

	def activationFunction(self, data):
		return af.sigmoid(data)		

	def predict(self, testSet):
		

		outNeuronActivation = np.zeros((len(testSet), self.nOutNeurons))	

		hiddenNeuronActivation = self.feedForward(testSet)

		for sample in range(len(testSet)):
			for outNeuron in range(self.nOutNeurons):
				activation = 0.0
				for hNeuron in range(self.nHiddenNeurons):
					activation += (hiddenNeuronActivation[sample][hNeuron] * self.outLayer.neurons[outNeuron].weight[hNeuron])
				
				activation -= self.outLayer.neurons[outNeuron].weight[self.nHiddenNeurons]	
				outNeuronActivation[sample][outNeuron] = self.activationFunction(activation)

		
		return outNeuronActivation	
