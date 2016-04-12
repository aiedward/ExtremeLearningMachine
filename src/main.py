#!/usr/bin/env python
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

		rawDataset = open('../selectedUserMore', 'r')

		# Value 0 represents max number of apps => It must be set
		maxValuesByFeature = [6,4,2,2,2,0,4,5,3]

		dataset = []

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
								
				line = label + " " + " ".join(features)
				dataset.append(line)

		rawDataset.close()		

		nLines = len(dataset)

		nLinesToTrain = int(0.7 * nLines)

		datasetTrain = dataset[:nLinesToTrain]
		datasetTest = dataset[nLinesToTrain:]

		fileTrain = open('../datasetTrain', 'w')
		fileTrain.write(str(len(datasetTrain)) + " 10" + " " + str(maxApps))
		fileTrain.write("\n")

		for line in datasetTrain:
			fileTrain.write(line)
			fileTrain.write("\n")

		fileTrain.close()	

		fileTest = open('../datasetTest', 'w')
		fileTest.write(str(len(datasetTest)) + " 10" + " " + str(maxApps))
		fileTest.write("\n")

		for line in datasetTest:
			fileTest.write(line)
			fileTest.write("\n")

		fileTest.close()


main = Main()
main.run()







