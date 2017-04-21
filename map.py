#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First [docId trueRelevance]

Second [docId queryId]

Third   [documentId   maxValueFromNeuralNet  probabilityFromNeuralNet]
"""
import numpy as np
from numpy import genfromtxt

def apk(actual, predicted):     #, k=10
    #if len(predicted)>k:
    #    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    return score / len(actual) #, k)

def mapk(actual, predicted):        #, k=10
    return np.mean([apk(a,p) for a,p in zip(actual, predicted)])


truerel = genfromtxt('first.csv', delimiter=',')
idd = genfromtxt('second.csv', delimiter=',')
model = genfromtxt('third.csv', delimiter=',')
truerel = truerel[1:900000]
idd = idd[1:900000]
model = model[1:900000]


aa = list(zip(truerel[:,0], np.argmax(truerel[:,1:5], axis=1)))
pp = list(zip(model[:,0], model[:,2]))

values = set(map(lambda x:x[1], idd))
lists = [[int(y[0]) for y in idd if y[1]==x] for x in values]
#numbers = [ int(x) for x in numbers ]actual = []

def getKey(item):
     return item[1]

actual = []
predicted = []
for i in range(len(lists)):
    a1 = [aa[idx] for idx in lists[i]]
    actual.append(a1)
    a2 = [pp[idx] for idx in lists[i]]  
    a2.sort(key=getKey,reverse=True)
    predicted.append(a2)
print(mapk(actual, predicted))