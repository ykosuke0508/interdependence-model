# -*- coding:utf-8 -*-
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.svm as svm
import pandas as pd
import copy
import time
import random
from operator import itemgetter
import sklearn.ensemble as ensemble
import scipy.optimize as optimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sympy import *
from collections import Counter

IDD_PRED_METHODS = {"GIBBS", "FPI", "ESwP", "ESwPfLR"}

class inner_dummy_binary_classifier(object):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.output_label = None

    def fit(X,Y):
        self.output_label = int(Y[0])

class problem_transformation_methods_template(object):
    def __init__(self):
        self._clf = None
        self._n_classes = None
        self._name = None
        self._training_time = None
        self._prediction_time = None
        raise ValueError

    @property
    def train_time(self):
        return self._training_time

    @property
    def prediction_time(self):
        return self._prediction_time

    @property
    def name(self):
        return self._name

    @property
    def n_classes(self):
        return self._n_classes

    def print_time(self):
        print("Method Name     : {}".format(self._name))
        if self._training_time is None:
            try:
                raise ValueError
            except:
                print("The model has not trained yet!")
        else:
            print("Training Time   : {0:04.4f} [sec]".format(self._training_time))

        if self._prediction_time is None:
            try:
                raise ValueError
            except:
                print("The model has not predicted yet!")
        else:
            print("Prediction Time : {0:04.4f} [sec]".format(self._prediction_time))

class InterdependenceModel(problem_transformation_methods_template):
    def __init__(self,clf,prediction_method = "GIBBS"):

        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError("There is not the inner classifier.")

        if prediction_method not in IDD_PRED_METHODS:
            raise ValueError("There is not {} as an prediction method for IDD model.".format(prediction_method))

        self._clf = clf
        self._n_classes = None
        self._name = "Interdependence Model"
        self._prediction_method = prediction_method
        self._training_time = None
        self._prediction_time = None
        self._wX = None
        self._wS = None
        self._max_n_labels = None
        # (self._constant) is dict type and recode the case that labels are all zero or all one.
        self._constant = {}

    def _label_check(self, train_S):
        for (i,row) in enumerate(train_S.T):
            if len(set(row)) < 2:
                self._constant[i] = int(row[0])

    def fit(self, train_X, train_S):
        self._training_time = time.time()
        self._label_check(train_S)
        keys = list(self._constant.keys())
        if len(keys) != 0:
            train_S = np.delete(train_S, keys, 1)
        self._n_classes = train_S.shape[1]
        feature_size  = train_X.shape[1]
        self._wX = np.zeros([feature_size + 1, self._n_classes])
        self._wS = np.zeros([self._n_classes, self._n_classes])

        for i in range(self._n_classes):
            train_Y = train_S[:,i]
            if i == 0:
                new_X = np.c_[train_X,train_S[:,i+1:]]
            elif i == self._n_classes - 1:
                new_X = np.c_[train_X,train_S[:,:-1]]
            else:
                new_X = np.c_[train_X,train_S[:,:i]]
                new_X = np.c_[new_X, train_S[:,i+1:]]

            clf = copy.deepcopy(self._clf)
            clf.fit(new_X, train_Y)
            self._wX[1:,i] = clf.coef_[:,:-self._n_classes+1]
            self._wX[0,i] = clf.intercept_
            self._wS[:,i] = np.insert(clf.coef_[:,-self._n_classes+1:], i, 0)
        self._max_n_labels = np.max(train_S.sum(axis = 1))
        self._training_time = time.time() - self._training_time
        inner_info = self._clf
        return self._name, inner_info

    def predict(self, test_X, prior_knowledge=None, tau = 0.5, N = 10000, conv = 0.00001):
        self._prediction_time = time.time()
        if prior_knowledge is not None:
            org_pr_know = prior_knowledge.copy()
        else:
            org_pr_know = None
        keys = list(self._constant.keys())
        if prior_knowledge is not None and len(keys) != 0:
            prior_knowledge = np.delete(prior_knowledge, keys, 1)
        pred_ = None

        def make_bin_vector(n):
            m = self._n_classes
            return np.array(list(map(int,list(format(n, '0{}b'.format(m))))))

        def logistic_function(x):
            return 1 / (1 + np.exp(-x))

        test_X = np.insert(test_X,0,1,axis = 1)
        bias = test_X.dot(self._wX)
        self.y = np.ones_like(bias)

        # GIBBS
        if self._prediction_method == "GIBBS":
            if prior_knowledge is None:
                result = np.zeros([test_X.shape[0], self._n_classes])
                r = np.random.uniform(0, 1, N * self._n_classes)
                for i in range(N):
                    for j in range(self._n_classes):
                        p = logistic_function(bias + self.y.dot(self._wS))
                        self.y[:,j] = np.where(p[:,j] > r[i * self._n_classes + j], 1, 0)
                    if i > N * 0.01 - 1:
                        result += self.y
                if tau < 0:
                    pred_ = result / (N * 0.99)
                else:
                    pred_ = np.where((result / (N * 0.99)) > tau, 1, 0)
            else: # with prior knowledge
                result = np.zeros([test_X.shape[0], self._n_classes])
                r = np.random.uniform(0, 1, N * self._n_classes)
                for i in range(N):
                    for j in range(self._n_classes):
                        p = logistic_function(bias + self.y.dot(self._wS))
                        self.y[:,j] = np.where(p[:,j] > r[i * self._n_classes + j], 1, 0)
                        self.y[:,j] = np.where(prior_knowledge[:,j] == prior_knowledge[:,j],
                                               prior_knowledge[:,j],
                                               self.y[:,j])
                    if i > N * 0.01 - 1:
                        result += self.y
                if tau < 0:
                    pred_ = result / (N * 0.99)
                else:
                    pred_ = np.where((result / (N * 0.99)) > tau, 1, 0)
        # FPI
        elif self._prediction_method == "FPI":
            if prior_knowledge is None:
                for i in range(N):
                    old_y = copy.deepcopy(self.y)
                    self.y = logistic_function(bias + self.y.dot(self._wS))
                    if np.linalg.norm(old_y - self.y) < conv:
                        break
                if tau < 0:
                    pred_ = self.y
                else:
                    pred_ = np.where(self.y > tau, 1, 0)
            else: # with prior knowledge
                for i in range(N):
                    self.y = np.where(prior_knowledge == prior_knowledge, prior_knowledge, self.y)
                    old_y = copy.deepcopy(self.y)
                    self.y = logistic_function(bias + self.y.dot(self._wS))
                    if np.linalg.norm(old_y - self.y) < conv:
                        break
                if tau < 0:
                    self.y = np.where(prior_knowledge == prior_knowledge, prior_knowledge, self.y)
                    pred_ = self.y
                else:
                    self.y = np.where(prior_knowledge == prior_knowledge, prior_knowledge, self.y)
                    pred_ = np.where(self.y > tau, 1, 0)
        # ES with Pruning
        elif self._prediction_method == "ESwP":
            inf = float("inf")

            def cross_entropy(b, labels):
                logf = logistic_function(b + labels.dot(self._wS))
                one_minus_logf = 1 - logf
                L_now = np.where(labels == 0, -np.log(one_minus_logf), -np.log(logf)).sum()
                return L_now

            def future_best(b, now):
                labels = np.copy(now)
                buf = np.where(labels == -1, 0, labels)
                b = b + buf.dot(self._wS)
                w_filter = np.tile(labels, (self.n_classes, 1))
                w = np.where(w_filter != -1, 0, self._wS.T)
                wp = np.where(w > 0, w, 0).sum(axis = 1)
                wm = np.where(w < 0, w, 0).sum(axis = 1)
                fmax = b + wp
                fmin = b + wm
                min1 = -np.log(logistic_function(fmax))
                min0 = -np.log(logistic_function(1 - fmin))
                # label = 0, label = 1, label = unknown
                L = np.where(labels == 0, min0, 0).sum() \
                  + np.where(labels == 1, min1, 0).sum() \
                  + np.where(labels == -1, np.minimum(min0, min1), 0).sum()
                return L

            def search_best(b, now, i, best_labels, best_L):
                if i == self.n_classes:
                    now_L = cross_entropy(b, now)
                    if best_L > now_L:
                        best_labels = np.copy(now)
                        best_L = now_L
                    return best_labels, best_L

                updated_now = np.copy(now)
                if now[i] != -1:
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        return best_labels, best_L
                    else:
                        return search_best(b, updated_now, i+1, best_labels, best_L)

                if now[i] == -1:
                    updated_now[i] = 0
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        pass
                    else:
                        best_labels, best_L = search_best(b, updated_now, i+1, best_labels, best_L)

                    updated_now[i] = 1
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        return best_labels, best_L
                    else:
                        return search_best(b, updated_now, i+1, best_labels, best_L)

            if prior_knowledge is None:
                temp = np.ones_like(bias) * -1
            else: # with prior knowledge
                temp = np.where(prior_knowledge == prior_knowledge, prior_knowledge, -1)
            for (j,b) in enumerate(bias):
                now = temp[j]
                init = np.zeros([self.n_classes])
                init_L = cross_entropy(b, init)
                pred, _ = search_best(b, now, 0, init, init_L)
                self.y[j] = pred
            pred_ = self.y.astype(np.int32)
        # ES+ (for Logistic Regression)
        elif self._prediction_method == "ESwPfLR":
            inf = float("inf")

            def first_pruning(now, b):
                for (j,(a, w)) in enumerate(zip(b, self._wS.T)):
                    a = a + np.where(now == 1, w, 0).sum()
                    w = np.where(now == 0, 0, w)
                    w = np.where(now == 1, 0, w)
                    wp = np.where(w > 0, w, 0)
                    wm = np.where(w < 0, w, 0)
                    maxg = a + wp.sum()
                    ming = a + wm.sum()
                    maxf = 1 / (1 + np.exp(-maxg))
                    minf = 1 / (1 + np.exp(-ming))
                    if -np.log(maxf) > -np.log(1 - maxf):
                        now[j] = 0
                        continue
                    if -np.log(1 - minf) > -np.log(minf):
                        now[j] = 1
                        continue
                return now

            def cross_entropy(b, labels):
                logf = logistic_function(b + labels.dot(self._wS))
                one_minus_logf = 1 - logf
                L_now = np.where(labels == 0, -np.log(one_minus_logf), -np.log(logf)).sum()
                return L_now

            def future_best(b, now):
                labels = np.copy(now)
                buf = np.where(labels == -1, 0, labels)
                b = b + buf.dot(self._wS)
                w_filter = np.tile(labels, (self.n_classes, 1))
                w = np.where(w_filter != -1, 0, self._wS.T)
                wp = np.where(w > 0, w, 0).sum(axis = 1)
                wm = np.where(w < 0, w, 0).sum(axis = 1)
                fmax = b + wp
                fmin = b + wm
                min1 = -np.log(logistic_function(fmax))
                min0 = -np.log(logistic_function(1 - fmin))
                # label = 0, label = 1, label = unknown
                L = np.where(labels == 0, min0, 0).sum() \
                  + np.where(labels == 1, min1, 0).sum() \
                  + np.where(labels == -1, np.minimum(min0, min1), 0).sum()
                return L

            def search_best(b, now, i, best_labels, best_L):
                if i == self.n_classes:
                    now_L = cross_entropy(b, now)
                    if best_L > now_L:
                        best_labels = np.copy(now)
                        best_L = now_L
                    return best_labels, best_L

                updated_now = np.copy(now)
                if now[i] != -1:
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        return best_labels, best_L
                    else:
                        return search_best(b, updated_now, i+1, best_labels, best_L)

                if now[i] == -1:
                    updated_now[i] = 0
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        pass
                    else:
                        best_labels, best_L = search_best(b, updated_now, i+1, best_labels, best_L)

                    updated_now[i] = 1
                    # Pruning
                    if best_L < future_best(b, updated_now):
                        return best_labels, best_L
                    else:
                        return search_best(b, updated_now, i+1, best_labels, best_L)

            if prior_knowledge is None:
                temp = np.ones_like(bias) * -1
            else: # with prior knowledge
                temp = np.where(prior_knowledge == prior_knowledge, prior_knowledge, -1)
            for (j,b) in enumerate(bias):
                now = temp[j]
                while(1):
                    last = np.copy(now)
                    now = first_pruning(now, b)
                    if np.all(now != -1) or np.all(last == now):
                        break
                init = np.zeros([self.n_classes])
                init_L = cross_entropy(b, init)
                pred, _ = search_best(b, now, 0, init, init_L)
                self.y[j] = pred
            pred_ = self.y.astype(np.int32)

        if len(self._constant) != 0:
            n_instances = test_X.shape[0]
            sorted(self._constant.items(), key=lambda x: x[0])
            for k, v in self._constant.items():
                add_ = np.array([v for _ in range(n_instances)])
                pred_ = np.insert(pred_,k,add_,axis = 1)
        if org_pr_know is not None:
            pred_ = np.where(org_pr_know == org_pr_know, org_pr_know, pred_)
        self._prediction_time = time.time() - self._prediction_time
        return pred_

def macro_accuracy_score(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError
    land = (np.array(y_true * y_pred)).sum(axis = 1)
    lor = np.where((y_true + y_pred) >= 1, 1, 0).sum(axis = 1)
    score = land / lor
    return score.mean()
