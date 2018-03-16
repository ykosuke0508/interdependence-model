# -*- coding: utf-8 -*-
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.svm as svm
import pandas as pd
import copy
import time
import random
import pickle
from operator import itemgetter
import sklearn.ensemble as ensemble
import scipy.optimize as optimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sympy import *


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


class LabelPowersets(problem_transformation_methods_template):
    def __init__(self,clf):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
        self._n_classes = None
        self._name = "Label Powersets"
        self._training_time = None
        self._prediction_time = None

    def fit(self, train_X, train_S):
        self._training_time = time.time()
        self._n_classes = train_S.shape[1]
        encoder = np.array([2 ** i for i in range(self._n_classes)])
        powerset_S = train_S.dot(encoder)
        inner_info = self._clf.fit(train_X, powerset_S)
        self._training_time = time.time() - self._training_time
        return self._name, inner_info

    def predict(self, test_X, prior_knowledge=None):
        self._prediction_time = time.time()
        if prior_knowledge is None:
            pred_Y = self._clf.predict(test_X)
        else:
            # Format : prior_knowledge
            # missing value -> None
            pred_Y = []
            pred_proba = self._clf.predict_proba(test_X)[:]
            master_classes = np.array(self._clf.classes_)
            master_mask = np.zeros(master_classes.shape[0])
            master_mask[:] = True
            for i in range(test_X.shape[0]):
                classes_ = copy.deepcopy(master_classes)
                answer_mask = copy.deepcopy(master_mask)
                for (j,value) in enumerate(prior_knowledge[i,::-1]):
                    index = self._n_classes - 1 - j
                    if value != value:
                        mask = classes_ - 2 ** index < 0
                        classes_ = np.where(mask, classes_, classes_ - 2 ** index)
                    elif value == 1:
                        mask = classes_ - 2 ** index < 0
                        answer_mask *= np.where(mask, False, True)
                        classes_ = np.where(mask, classes_, classes_ - 2 ** index)
                    else: # value == 0:
                        mask = classes_ - 2 ** index < 0
                        answer_mask *= np.where(mask, True, False)
                        classes_ = np.where(mask, classes_, classes_ - 2 ** index)
                if np.all(pred_proba[i] * answer_mask == 0):
                    pred_Y.append(np.nan)
                else:
                    pred_Y.append(master_classes[np.argmax(pred_proba[i] * answer_mask)])
            pred_Y = np.array(pred_Y)
        pred_S = []
        for (i,decimal) in enumerate(pred_Y):
            if decimal == decimal:
                pred_S.append([int(value) for value in bin(decimal.astype(int) + 2 ** self._n_classes)[:2:-1]])
            else:
                buf = master_classes[np.argmax(pred_proba[i])]
                buf = [int(value) for value in bin(buf + 2 ** self._n_classes)[:2:-1]]
                buf = np.array(buf)
                buf = -np.where(prior_knowledge[i] == prior_knowledge[i], prior_knowledge[i], buf)
                pred_S.append(list(buf))
        self._prediction_time = time.time() - self._prediction_time
        return np.array(pred_S)

class BinaryRelevance(problem_transformation_methods_template):
    def __init__(self,clf):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
        self._n_classes = None
        self._name = "Binary Relevance"
        self._l_models = []
        self._training_time = None
        self._prediction_time = None
        self._constant = {}

    def fit(self, train_X, train_S):
        self._training_time = time.time()
        self._n_classes = train_S.shape[1]
        self._l_models = []
        for i in range(self._n_classes):
            clf = copy.deepcopy(self._clf)
            try:
                clf.fit(train_X, train_S[:,i])
                self._l_models.append(clf)
            except:
                if len(set(train_S[:,i])) != 1:
                    raise ValueError("debag")
                self._l_models.append(None)
                self._constant[i] = int(list(set(train_S[:,i]))[0])

        inner_info = self._l_models[:]
        self._training_time = time.time() - self._training_time
        return self._name, inner_info

    def predict(self, test_X, prior_knowledge=None):
        self._prediction_time = time.time()
        n_instances = test_X.shape[0]
        l_pred_S = []
        for (i,clf) in enumerate(self._l_models):
            if clf is None:
                buf = self._constant[i]
                buf_list = [buf for _ in range(n_instances)]
                l_pred_S.append(buf_list)
            else:
                pred_Y = clf.predict(test_X)[:]
                l_pred_S.append(list(pred_Y))
        if prior_knowledge is None:
            self._prediction_time = time.time() - self._prediction_time
            return np.array(l_pred_S).T
        else:
            pred_S = np.array(l_pred_S).T
            self._prediction_time = time.time() - self._prediction_time
            return np.where(prior_knowledge == prior_knowledge, prior_knowledge, pred_S)

class ClassifierChains(problem_transformation_methods_template):
    def __init__(self,clf):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
    def __init__(self,clf):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
    def __init__(self,clf):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
        self._n_classes = None
        self._name = "Classifier Chains"
        self._l_models = []
        self._training_time = None
        self._prediction_time = None
        self._order = []
        self._constant = {}

    def fit(self, train_X, train_S, b_random = False):
        self._training_time = time.time()
        self._n_classes = train_S.shape[1]
        self._l_models = []
        self._order = [i for i in range(self._n_classes)]

        n_instances = train_X.shape[0]
        if b_random:
            random.shuffle(self._order)

        for i in self._order:
            clf = copy.deepcopy(self._clf)
            try:
                clf.fit(train_X, train_S[:,i])
                self._l_models.append(clf)
                train_X = np.c_[train_X, clf.predict(train_X)]
            except:
                if len(set(train_S[:,i])) != 1:
                    raise ValueError("debag")
                self._l_models.append(None)
                self._constant[i] = int(list(set(train_S[:,i]))[0])
                buf = self._constant[i]
                buf_list = np.array([buf for _ in range(n_instances)])
                train_X = np.c_[train_X, buf_list]
        self._training_time = time.time() - self._training_time
        inner_info = self._l_models[:]
        return self._name, inner_info


    def predict(self, test_X, prior_knowledge=None):
        self._prediction_time = time.time()
        n_instances = test_X.shape[0]
        l_pred_S = []
        for (i, clf) in enumerate(self._l_models):
            if clf is None:
                buf = self._constant[i]
                pred_Y = np.array([buf for _ in range(n_instances)])
            else:
                pred_Y = clf.predict(test_X)[:]

            if prior_knowledge is not None:
                pred_Y = np.where(prior_knowledge[:,i] == prior_knowledge[:,i], prior_knowledge[:,i], pred_Y)
            l_pred_S.append(list(pred_Y))
            pred_Y = np.matrix(pred_Y).T
            test_X = np.c_[test_X, pred_Y]
        pred_S = np.zeros_like(np.array(l_pred_S))
        for (i, pred) in zip(self._order, l_pred_S):
            pred_S[i] = np.array(pred)
        pred_S = pred_S.T
        self._prediction_time = time.time() - self._prediction_time
        return pred_S

    @property
    def order(self):
        return self._order

class MetaStacking(problem_transformation_methods_template):
    """
    Title : Discriminative Methods for Multi-labeled Classification
    Authors : Shantanu Godbole and Sunita Sarawagi
    """
    def __init__(self,clf,f_threshold = 0.5):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
        self._n_classes = None
        self._name = "Meta Stacking"
        self._l_f_models = []
        self._l_s_models = []
        self._training_time = None
        self._prediction_time = None
        self._f_threshold = f_threshold
        self._fconstant = {}
        self._sconstant = {}

    def fit(self, train_X, train_S):
        self._training_time = time.time()
        self._n_classes = train_S.shape[1]

        def first_stage_train(train_X, train_S):
            model_list = []
            for i in range(self._n_classes):
                clf = copy.deepcopy(self._clf)
                try:
                    clf.fit(train_X, train_S[:,i])
                    model_list.append(clf)
                except:
                    if len(set(train_S[:,i])) != 1:
                        raise ValueError("debag")
                    model_list.append(None)
                    self._fconstant[i] = int(list(set(train_S[:,i]))[0])
            return model_list

        def second_stage_train(train_extra_X, train_S):
            model_list = []
            for i in range(self._n_classes):
                clf = copy.deepcopy(self._clf)
                try:
                    clf.fit(train_extra_X, train_S[:,i])
                    model_list.append(clf)
                except:
                    if len(set(train_S[:,i])) != 1:
                        raise ValueError("debag")
                    model_list.append(None)
                    self._sconstant[i] = int(list(set(train_S[:,i]))[0])
            return model_list

        def first_stage_predict(test_X):
            l_pred_S = []
            n_instances = test_X.shape[0]
            for (i, clf) in enumerate(self._l_f_models):
                if clf is None:
                    buf = self._fconstant[i]
                    buf_list = [buf for _ in range(n_instances)]
                    l_pred_S.append(buf_list)
                else:
                    pred_Y = clf.predict(test_X)[:]
                    l_pred_S.append(list(pred_Y))
            pred_S = np.array(l_pred_S)
            pred_S = np.where(pred_S.T > self._f_threshold, 1, 0)
            return pred_S

        self._l_f_models = first_stage_train(train_X, train_S)
        added_feature = first_stage_predict(train_X)
        train_extra_X = np.c_[train_X, added_feature]
        self._l_s_models = second_stage_train(train_extra_X, train_S)
        self._training_time = time.time() - self._training_time
        inner_info = self._l_f_models[:]
        inner_info.extend(self._l_s_models[:])
        return self._name, inner_info

    def predict(self,test_X, prior_knowledge=None):
        self._prediction_time = time.time()

        def first_stage_predict(test_X):
            l_pred_S = []
            n_instances = test_X.shape[0]
            for (i, clf) in enumerate(self._l_f_models):
                if clf is None:
                    buf = self._fconstant[i]
                    buf_list = [buf for _ in range(n_instances)]
                    l_pred_S.append(buf_list)
                else:
                    pred_Y = clf.predict(test_X)[:]
                    l_pred_S.append(list(pred_Y))
            pred_S = np.array(l_pred_S)
            pred_S = np.where(pred_S.T > self._f_threshold, 1, 0)
            return pred_S

        def second_stage_predict(test_extra_X):
            l_pred_S = []
            n_instances = test_extra_X.shape[0]
            for (i,clf) in enumerate(self._l_s_models):
                if clf is None:
                    buf = self._sconstant[i]
                    buf_list = [buf for _ in range(n_instances)]
                    l_pred_S.append(buf_list)
                else:
                    pred_Y = clf.predict(test_extra_X)[:]
                    l_pred_S.append(list(pred_Y))
            pred_S = np.array(l_pred_S)
            pred_S = pred_S.T
            return pred_S

        added_feature = first_stage_predict(test_X)
        if prior_knowledge is not None:
            added_feature = np.where(prior_knowledge == prior_knowledge, prior_knowledge, added_feature)
        test_extra_X = np.c_[test_X, added_feature]
        pred_S = second_stage_predict(test_extra_X)
        if prior_knowledge is not None:
            pred_S = np.where(prior_knowledge == prior_knowledge, prior_knowledge, pred_S)
        self._prediction_time = time.time() - self._prediction_time
        return pred_S

class SubsetMapping(problem_transformation_methods_template):
    def __init__(self,clf, T = 5):
        if ('fit' not in dir(clf)) or ('predict' not in dir(clf)):
            raise ValueError
        self._clf = clf
        self._n_classes = None
        self._name = "Subset Mapping (Subset Matching)"
        self._l_models = []
        self._training_time = None
        self._prediction_time = None
        self._train_data = []
        self._constant = {}

    def fit(self, train_X, train_S):
        self._training_time = time.time()
        self._n_classes = train_S.shape[1]
        self._l_models = []
        for i in range(self._n_classes):
            clf = copy.deepcopy(self._clf)
            try:
                clf.fit(train_X, train_S[:,i])
                self._l_models.append(clf)
            except:
                if len(set(train_S[:,i])) != 1:
                    raise ValueError("debag")
                self._l_models.append(None)
                self._constant[i] = int(list(set(train_S[:,i]))[0])

        inner_info = self._l_models[:]
        self._train_data = [str(row) for row in train_S]
        self._train_data = np.array([list(map(int,row[1:-1].split(" "))) for row in set(self._train_data)])
        self._training_time = time.time() - self._training_time
        return self._name, inner_info

    def predict(self, test_X, prior_knowledge=None):
        self._prediction_time = time.time()
        l_pred_S = []
        n_instances = test_X.shape[0]
        for (i, clf) in enumerate(self._l_models):
            if clf is None:
                buf = self._constant[i]
                buf_list = [buf for _ in range(n_instances)]
                l_pred_S.append(buf_list)
            else:
                pred_Y = clf.predict(test_X)[:]
                l_pred_S.append(list(pred_Y))
        pred_S = np.array(l_pred_S).T
        pred_S = np.where(pred_S == 0, -1, 1)
        pm_one_train_data = np.where(self._train_data == 0, -1, 1)
        if prior_knowledge is not None:
            pred_S = np.where(prior_knowledge == prior_knowledge, prior_knowledge, pred_S)
        pred = [self._train_data[np.argmin(np.sum(np.exp(-row * pm_one_train_data), axis = 1))] for row in pred_S]
        pred = np.array(pred)
        if prior_knowledge is not None:
            pred = np.where(prior_knowledge == prior_knowledge, prior_knowledge, pred)
        self._prediction_time = time.time() - self._prediction_time
        return pred

def generate_prior_knowledge(ground_truth, n_open_labels, same=False, seed = None):
    if not isinstance(n_open_labels, int):
        if (not n_open_labels.isdigit()):
            raise ValueError("n_open_labels must be integer number.")
        else:
            n_open_labels = int(n_open_labels)

    n_candidate_labels = ground_truth.shape[1]

    if n_open_labels <= 0 or n_candidate_labels <= n_open_labels:
        raise ValueError("Now, n_open_labels is {0}.\nn_open_labels must satisfy 0 < {0} < {1}".format(n_open_labels, n_candidate_labels))

    if seed is not None:
        random.seed(seed)

    if same:
        labels = [i for i in range(n_candidate_labels)]
        random.shuffle(labels)
        prior_k = np.empty(ground_truth.shape)
        prior_k[:,:] = np.nan
        for label in labels[:n_open_labels]:
            prior_k[:,label] = ground_truth[:,label]
        return prior_k
    else:
        labels = [i for i in range(n_candidate_labels)]
        prior_k = np.empty(ground_truth.shape)
        prior_k[:,:] = np.nan
        for i in range(ground_truth.shape[0]):
            sampled_labels = random.sample(labels, n_open_labels)
            prior_k[i,sampled_labels] = ground_truth[i,sampled_labels]
        return prior_k

def save_split_index(data, n_groups, seed=None, pickle_name=None, dir_name="."):
    n_instances = data.shape[0]
    indexes = [i for i in range(n_instances)]

    if seed is not None:
        random.seed(seed)

    random.shuffle(indexes)
    groups = []
    for i in range(n_groups):
        groups.append(indexes[i::n_groups])

    if pickle_name:
        with open('{}/{}_{}_split.pickle'.format(dir_name, pickle_name, n_groups), 'wb') as f:
            pickle.dump(groups, f)

    return groups
