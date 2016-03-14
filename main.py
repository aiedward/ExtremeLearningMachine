#!/usr/bin/env python
from FeedforwardSingleNeuralNetwork import FeedforwardSingleNeuralNetwork
import numpy as np



ffsn = FeedforwardSingleNeuralNetwork(4,3,2)

ffsn.train([[0,1,0,1],[0,1,1,0],[1,0,1,0]], [[0,1], [1,0], [0,0]])

print ffsn.predict([[0,1,0,1], [0,0,0,1]])




