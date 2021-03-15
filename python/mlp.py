from numpy import genfromtxt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

csv = genfromtxt('data.csv', delimiter=',')
data = csv[:,:2]
categories = csv[:,2]

#enc = OneHotEncoder()

inputs = [ [3.5,3], [1.5,3], [1.8,1.9] ]

clf = MLPClassifier(hidden_layer_sizes=(4,4),
                    activation = "relu")
clf.fit(data, categories)

predictions = clf.predict(inputs)
print(predictions)

