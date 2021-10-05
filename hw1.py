import numpy as np
from numpy import genfromtxt
import sklearn

dataSet = genfromtxt('diabetes.csv', delimiter=',')
features = []
labels = []
for i in range(len(dataSet[0])):
    if i < len(dataSet[0])-1:
        features.append(dataSet[:,i])
    else:
        labels.append(dataSet[:,i])

print(labels)

