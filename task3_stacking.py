#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/26 23:26
# @Author  : Junru_Lu
# @Site    : 
# @File    : task3_stacking.py
# @Software: PyCharm

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from numpy import genfromtxt
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVR
reload(sys)
sys.setdefaultencoding('utf-8')
import time

start = time.clock()

def stacking(base_models,function,X, Y, T, wv_X, wv_T):
    models = base_models
    folds = list(KFold(len(Y), n_folds=10, random_state=0))
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))
    for i, bm in enumerate(models):
        clf = bm[1]
        S_test_i = np.zeros((T.shape[0], len(folds)))
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_holdout = X[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = clf.predict(T)[:]
        S_test[:, i] = S_test_i.mean(1)
    S_train = np.column_stack((S_train, wv_X))
    S_test = np.column_stack((S_test, wv_T))
    nuss=function()
    nuss.fit(S_train, Y)
    yp = nuss.predict(S_test)[:]
    return yp
def stacking_easy(base_models,function,X, Y, T):
    models = base_models
    folds = list(KFold(len(Y), n_folds=10, random_state=0))
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))
    for i, bm in enumerate(models):
        clf = bm[1]
        S_test_i = np.zeros((T.shape[0], len(folds)))
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_holdout = X[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = clf.predict(T)[:]
        S_test[:, i] = S_test_i.mean(1)
    nuss=function()
    nuss.fit(S_train, Y)
    yp = nuss.predict(S_test)[:]
    return yp
def score(L,y_te):
    i = 0
    final_score = 0.0
    y_te_list = y_te.tolist()
    while i < len(y_te_list):
        L[i]=abs(L[i])
        if L[i] <= 0 and y_te_list[i][0] == 0:
            score = 0.0
        else:
            try:
                score = abs(L[i] - y_te_list[i][0]) / max(L[i], y_te_list[i][0])
            except:
                score = 0.0
        final_score += score
        i += 1
    print 1.0 - final_score / len(y_te_list)
def save(L,filename):
    Lname = [s.strip().split(',')[0] for s in list(open('data/task3_test_final.csv', 'r').readlines())[1:]]
    i = 0
    w = open(filename, 'w')
    while i < 300000:
        w.write(Lname[i] + "	" + str(abs(L[i])) + "\n")
        i += 1
    w.close()

S=1000000

datapath_trainxconfer = r'res/author_confer_train.csv' #改成去重后会议个数(完成)
deliverydata_trainxconfer = genfromtxt(datapath_trainxconfer, delimiter=',', skip_header=False)
train_xconfer = deliverydata_trainxconfer[:, 1:][:S,:]
datapath_trainxgrowth = r'res/author_refer_train_allyearsgrowth.csv' #每年被引数量的增长值(完成)
deliverydata_trainxgrowth = genfromtxt(datapath_trainxgrowth, delimiter=',', skip_header=False)
train_xgrowth = deliverydata_trainxgrowth[:, 1:][:S,:]
datapath_trainx = r'res/author_refer_train_allyears.csv' #每年被引用文章数量(完成)
deliverydata_trainx = genfromtxt(datapath_trainx, delimiter=',', skip_header=False)
train_xpre = deliverydata_trainx[:, 1:][:S,:]
train_xstd = np.std(train_xpre,axis=1) #文章引用的标准差(完成)
datapath_trainx2016 = r'res/author_refer_train_upto2016.csv' #截止2016的文章篇数(完成)
deliverydata_trainx2016 = genfromtxt(datapath_trainx2016, delimiter=',', skip_header=False)
train_x2016 = deliverydata_trainx2016[:, 1:][:S,:]
datapath_trainxconferwithweight = r'res/author_conferwithweight_train.csv' #带权重的会议数量(完成)
deliverydata_trainxconferwithweight = genfromtxt(datapath_trainxconferwithweight, delimiter=',', skip_header=False)
train_xconferwithweight = deliverydata_trainxconferwithweight[:, 1:][:S,:]

datapath_trainy = r'data/train.csv'
deliverydata_trainy = genfromtxt(datapath_trainy, delimiter=',', skip_header=True)
train_x = np.column_stack((train_xpre,train_xgrowth,train_xstd,train_xconfer))
train_x1 = np.column_stack((train_xpre,train_xgrowth,train_xstd,train_xconferwithweight))
train_y = train_x2016 + deliverydata_trainy[:, 1:][:S,:]

datapath_testconfer = r'res/author_confer_test.csv'
deliverydata_testconfer = genfromtxt(datapath_testconfer, delimiter=',', skip_header=False)
testconfer = deliverydata_testconfer[:, 1:]
datapath_testgrowth = r'res/author_refer_test_allyearsgrowth.csv'
deliverydata_testgrowth = genfromtxt(datapath_testgrowth, delimiter=',', skip_header=False)
testgrowth = deliverydata_testgrowth[:, 1:]
datapath_test = r'res/author_refer_test_allyears.csv'
deliverydata_test = genfromtxt(datapath_test, delimiter=',', skip_header=False)
testpre = deliverydata_test[:, 1:]
teststd = np.std(testpre,axis=1)
datapath_testconferwithweight = r'res/author_conferwithweight_test.csv'
deliverydata_testconferwithweight = genfromtxt(datapath_testconferwithweight, delimiter=',', skip_header=False)
testconferwithweight = deliverydata_testconferwithweight[:, 1:]
test = np.column_stack((testpre,testgrowth,teststd,testconfer))
test1 = np.column_stack((testpre,testgrowth,teststd,testconferwithweight))

x_tr, x_te, y_tr, y_te,x_a,x_b= train_test_split(train_x, train_y,train_xconferwithweight,test_size=0.3, random_state=0)
x_tr1, x_te1, y_tr1, y_te1,x_a1,x_b1= train_test_split(train_x1, train_y,train_xconfer,test_size=0.3, random_state=0)

hub=HuberRegressor(max_iter=300,alpha=0.63,tol=1e-07)

L1 = [int(round(p, 0)) for p in stacking_easy([['fm1',hub]], LinearSVR, x_tr,y_tr,x_te).tolist()]
print "huber_confer:",
score(L1,y_te)
L2 = [int(round(q, 0)) for q in stacking_easy([['fm1',hub]],LinearSVR,train_x,train_y,test).tolist()]
save(L2,'result/task3_huber_confer.txt')

L3 = [int(round(p, 0)) for p in stacking([['fm1',hub]], LinearSVR, x_tr,y_tr,x_te,x_a,x_b).tolist()]
print "huber_weightasstacking:",
score(L3,y_te)
L4 = [int(round(q, 0)) for q in stacking([['fm1',hub]],LinearSVR,train_x,train_y,test,train_xconferwithweight,testconferwithweight).tolist()]
save(L4,'result/task3_huber+weightasstacking.txt')

L5 = [int(round(p, 0)) for p in stacking_easy([['fm1',hub]], LinearSVR, x_tr1,y_tr1,x_te1).tolist()]
print "huber_weight:",
score(L5,y_te1)
L6 = [int(round(q, 0)) for q in stacking_easy([['fm1',hub]],LinearSVR,train_x1,train_y,test1).tolist()]
save(L6,'result/task3_huber_weight.txt')

L7 = [int(round(p, 0)) for p in stacking([['fm1',hub]], LinearSVR, x_tr1,y_tr1,x_te1,x_a1,x_b1).tolist()]
print "huber_conferasstacking:",
score(L7,y_te1)
L8 = [int(round(q, 0)) for q in stacking([['fm1',hub]],LinearSVR,train_x1,train_y,test1,train_xconfer,testconfer).tolist()]
save(L8,'result/task3_huber+conferasstacking.txt')

end = time.clock()
print '\n'
print "read: %f s" % (end - start)