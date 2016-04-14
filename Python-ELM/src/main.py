#!/usr/bin/env python
from elm import ELMClassifier
import numpy as np
import random

class Main:

	def findMaxNumberOfApps(self, file):
		file.seek(0)
		apps = []

		for line in file:
			tuples = line.split(';')
		
			for t in range(len(tuples)):
				tup = tuples[t]
				vals = tup.split(',')
				apps.append(vals[0])
		
		return max(apps)
	
	# 1. Battery Level => 0-6
	# 2. Brightness Level => 0-4
	# 3. Screen Orientarion => 0-2
	# 4. Power => 0-2
	# 5. Headset => 0-2
	# 6. App => 0-X
	# 7. Last Period of the Day => 0-4
	# 8. Last Time between apps => 0-5
	# 9. Las GPS => 0-3
	def createNeuronInput(self, activatedValue, maxValue):
		
		inputs = []

		for pos in range(int(maxValue) + 1):
			if(pos != int(activatedValue)):
				inputs.append(0)
			else:
				inputs.append(1)
		
		return inputs



	def run(self):

		rawDataset = open('../../selectedUserMore', 'r')

		# Value 0 represents max number of apps => It must be set
		maxValuesByFeature = [6,4,2,2,2,0,4,5,3]

		datasetFeatures = []
		datasetLabels = []

		maxApps = self.findMaxNumberOfApps(rawDataset)

		# Setting max number of apps
		maxValuesByFeature[5] = maxApps

		rawDataset.seek(0)
		for line in rawDataset:
			tuples = line.split(';')


			for t in range(len(tuples)):
				
				tup = tuples[t]
				vals = tup.split(',')
				label = vals[0]
				features = vals[1].split('_')
				
				datasetFeatures.append(features)
				datasetLabels.append(label)


		rawDataset.close()		

		nTuples = len(datasetLabels)
		
		nTrain = int(0.7 * nTuples)
				
		nMaxHidden = 3005
		totalAccuracy = 0.0

		trainSetFeatures = np.array(datasetFeatures[:nTrain], dtype=np.int)
		trainSetLabels = np.array(datasetLabels[:nTrain], dtype=np.int)

		testSetFeatures = np.array(datasetFeatures[nTrain:], dtype=np.int)
		testSetLabels = np.array(datasetLabels[nTrain:], dtype=np.int)


		# for x in range(200,nMaxHidden):
		
		ffsn = ELMClassifier(n_hidden=1000, activation_func='sine')
		ffsn.fit(trainSetFeatures, trainSetLabels)
			
		results = ffsn.predict(testSetFeatures)
			
		nPredictedCorrectly = 0	

		
		for test in range(len(testSetLabels)):
			realValue = testSetLabels[test]
			predictedValue = results[test]	
			
			if(int(realValue) == int(predictedValue)):
				nPredictedCorrectly += 1

				
			# print "Real: " + str(realValue) + " / Predicted: " + str(predictedValue)	

		totalTests = nTuples - nTrain
		accuracy = float(nPredictedCorrectly) / totalTests	
		# print "N Hidden: " + str(x) + " / Accuracy: " + str(accuracy)
		print "Accuracy: " + str(accuracy)
		totalAccuracy += accuracy


		# meanTotalAccuracy = totalAccuracy / nIterations	
		# print "Total Accuracy: " + str(meanTotalAccuracy)	

main = Main()
main.run()







