#!/usr/bin/env python
# coding: utf-8

# Logistic Regression Problem

import numpy as np
import csv
from sklearn import preprocessing

#### File reading
dat_file = np.load('student.npz')
A = dat_file['A_learn']
final_grades = dat_file['b_learn']
m = final_grades.size
b = np.zeros(m)
for i in range(m):
    if final_grades[i]>11:
        b[i] = 1.0
    else:
        b[i] = -1.0

A_test = dat_file['A_test']
final_grades_test = dat_file['b_test']
m_test = final_grades_test.size
b_test = np.zeros(m_test)
for i in range(m_test):
    if final_grades_test[i]>11:
        b_test[i] = 1.0
    else:
        b_test[i] = -1.0


d = 27      # features
n = d+1     # with the intercept

lam2 = 0.1      # for the 2-norm regularization best:0.1
lam1 = 0.03     # for the 1-norm regularization best:0.03

L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2


## Oracles
### Related to function f
def f(x, lambda2=lam2):
    l = 0.0
    for i in range(A.shape[0]):
        if b[i] > 0 :
            l += np.log( 1 + np.exp(-np.dot( A[i] , x ) ) )
        else:
            l += np.log( 1 + np.exp(np.dot( A[i] , x ) ) )
    return l/m + lambda2/2.0*np.dot(x,x)

def f_grad(x, lambda2=lam2):
    g = np.zeros(n)
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) )
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) )
    return g/m + lambda2*x

## Related to function f_i (one example)
def f_ex(x, i, lambda2=lam2):

    #
    #### TODO: task 1
    #
    return 0.0

def f_grad_ex(x, i, lambda2=lam2):
    g = np.zeros(n)

    #
    #### TODO: task 1
    #

    return g


### Related to function g
def g(x, lambda1=lam1):
    return lam1*np.linalg.norm(x,1)

def g_prox(x, gamma, lambda1=lam1):
    p = np.zeros(n)
    for i in range(n):
        if x[i] < - lambda1*gamma:
            p[i] = x[i] + lambda1*gamma
        if x[i] > lambda1*gamma:
            p[i] = x[i] - lambda1*gamma
    return p


### Related to function F
def F(x, lambda1=lam1, lambda2=lam2):
    return f(x, lambda2=lambda2) + g(x, lambda1=lambda1)



## Prediction Function
def prediction_train(w,PRINT):
    pred = np.zeros(A.shape[0])
    perf = 0
    for i in range(A.shape[0]):
        p = 1.0/( 1 + np.exp(-np.dot( A[i] , w ) ) )
        if p>0.5:
            pred[i] = 1.0
            if b[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A.shape[0]

def prediction_test(w,PRINT):
    pred = np.zeros(A_test.shape[0])
    perf = 0
    for i in range(A_test.shape[0]):
        p = 1.0/( 1 + np.exp(-np.dot( A_test[i] , w ) ) )
        if p>0.5:
            pred[i] = 1.0
            if b_test[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b_test[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A_test.shape[0]

