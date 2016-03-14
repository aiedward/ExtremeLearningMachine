#!/usr/bin/env python
from FeedforwardSingleNeuralNetwork import FeedforwardSingleNeuralNetwork
import numpy as np



ffsn = FeedforwardSingleNeuralNetwork(3,5,2)

ffsn.train([[1,2,3],[1,2,1]], [3,1])

print ffsn.predict([[1,2,3],[1,2,1],[1,2,2]])



