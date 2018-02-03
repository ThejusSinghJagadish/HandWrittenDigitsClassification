#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:39:58 2018

@author: thejussinghj
"""

from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn import datasets

# load the MNIST dataset and split it into training and testing data

mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(mnist.data, mnist.target,
	test_size=0.25, random_state=42)


#Train the perceptron

model = Perceptron(n_iter=30, eta0=1.0, random_state=84)
model.fit(trainData, trainLabels)

#Evaluate the perceptron
prediction = model.predict(testData)
print(classification_report(prediction, testLabels))