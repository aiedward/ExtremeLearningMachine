#!/usr/bin/env python
from FeedforwardSingleNeuralNetwork import FeedforwardSingleNeuralNetwork
import numpy as np



ffsn = FeedforwardSingleNeuralNetwork(3)
ffsn.train([[0,1,0,1], [1,0,1,0], [0,1,1,0]], [[0,1],[1,0],[0,1]])





