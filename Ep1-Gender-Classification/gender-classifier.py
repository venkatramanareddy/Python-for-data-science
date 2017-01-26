import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import naive_bayes

#Inputs ([height,weight,shoe-size])
Inputs = [	[181,80,44], [177,70,43], [160,60,38], [154, 54, 37], [166, 65, 40],
     		[190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]] 

#Outputs (gender)
Output = ['male','male','female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#input_height = input("Enter height : ")
#input_weight = input("Enter weight : ")
#input_shoe_size = input("Enter shoe-size : ")

input_height = 175
input_weight = 64
input_shoe_size = 39

#Using decision tree classifier
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(Inputs,Output)
prediction = classifier.predict([[input_height,input_weight,input_shoe_size]])

print "Predicted gender by decision tree : " + str(prediction)

#Using support vector machines
classifier = svm.SVC()
classifier = classifier.fit(Inputs,Output)
prediction = classifier.predict([[input_height,input_weight,input_shoe_size]])

print "Predicted gender by SVM : " + str(prediction)

#Using K-Nearest Neighbours 
classifier = neighbors.KNeighborsClassifier()
classifier = classifier.fit(Inputs,Output)
prediction = classifier.predict([[input_height,input_weight,input_shoe_size]])

print "Predicted gender by K-Nearest Neighbours : " + str(prediction)

#Using Adaboost algorithm
classifier = ensemble.AdaBoostClassifier()
classifier = classifier.fit(Inputs,Output)
prediction = classifier.predict([[input_height,input_weight,input_shoe_size]])

print "Predicted gender by Adaboost : " + str(prediction)

#Using Gaussian Naive Bayes
classifier = naive_bayes.GaussianNB()
classifier = classifier.fit(Inputs,Output)
prediction = classifier.predict([[input_height,input_weight,input_shoe_size]])

print "Predicted gender by GaussianNB : " + str(prediction)


