import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
from sklearn import tree

def classifierSVM():
    clf = svm.SVC(kernel = 'linear', C = 1, gamma='auto')
    return clf

def classifierDT():
    clf = tree.DecisionTreeClassifier()
    return clf

def classifierLDA():
    clf = LinearDiscriminantAnalysis()
    return clf

rawSet = genfromtxt('diabetes.csv', delimiter=',')
features = []
labels = []
dataSet = np.delete(rawSet,0,0)

for i in range(len(dataSet)):
    features.append(dataSet[i][0:len(dataSet[0])-1])
    labels.append(dataSet[i][len(dataSet[0])-1])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)



#tree.plot_tree(clf)
#plt.savefig('tree_visualization.png') 



#scores = cross_val_score(clf, training_data, testing_data, cv=4)
#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

prediction = []
for i in range(len(X_test)):
    prediction.append(clf.predict([X_test[i]]))

f1_score = f1_score(prediction, y_test, average=None)
for i, _ in enumerate(f1_score):
    print(f"F1 score of class {i}: {_}")
print("Average F1 score:", np.mean(f1_score))

