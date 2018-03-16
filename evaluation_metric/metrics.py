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

"""
references
A Literature Survey on Algorithms for Multi-label Learning : [Mohammad S Sorower]
"""

def _isndarray(instance):
    return isinstance(instance, np.ndarray)

def _is_zero_one(ndarray):
    return np.all(np.logical_or(ndarray == 1, ndarray == 0))

def _change2int(ndarray):
    return ndarray.astype(np.int32)

def _format_check(instance, name):
    if _isndarray(instance) and _is_zero_one(instance):
        return _change2int(instance)
    else:
        raise ValueError("The format of {0} is incorrect.\n{0} is numpy.ndarray, and {0}'s elements are 0/1 only.")

def _previous_check(y_true, y_pred):
    y_true = _format_check(y_true, "y_true")
    y_pred = _format_check(y_pred, "y_pred")
    if np.any(y_true.sum(axis=1) == 0):
        print("There is no-labeled instance in ground truth.")
    return y_true, y_pred


"""
Example-based metric
"""
# accuracy
def accuracy(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : and
    cap = y_true * y_pred
    # cup : or
    cup = y_true + y_pred - cap
    cap = cap.sum(axis=1)
    cup = cup.sum(axis=1)
    score = cap / cup
    score = np.mean(score)
    return score

# precision
def precision(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : and
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_pred = y_pred.sum(axis=1)
    score = cap / y_pred
    score = np.where(score == score, score, 0)
    score = np.mean(score)
    return score

# recall
def recall(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : and
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_true = y_true.sum(axis=1)
    score = cap / y_true
    score = np.where(score == score, score, 1)
    score = np.mean(score)
    return score

# F1 measure
def F1_measure(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : and
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_true = y_true.sum(axis=1)
    y_pred = y_pred.sum(axis=1)
    score = (2 * cap) / (y_true + y_pred)
    score = np.mean(score)
    return score

# hamming loss
def hamming_loss(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    score = np.where(y_true != y_pred, 1, 0)
    score = np.mean(score)
    return score

# 0/1 loss
def zero_one_loss(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    miss = np.where(y_true != y_pred, 1, 0).sum(axis = 1)
    score = np.where(miss > 0, 1, 0)
    score = np.mean(score)
    return score

# Exact Match Ratio
def exact_match_ratio(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    miss = np.where(y_true != y_pred, 1, 0).sum(axis = 1)
    score = np.where(miss == 0, 1, 0)
    score = np.mean(score)
    return score

"""
Label-based metric
"""

# macro Precision
def macro_precision(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum(axis=0)
    y_pred = y_pred.sum(axis=0)
    score = prod / y_pred
    score = np.mean(score)
    return score

# macro Recall
def macro_recall(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum(axis=0)
    y_true = y_true.sum(axis=0)
    score = prod / y_true
    score = np.mean(score)
    return score

# macro F1 measure
def macro_F1_measure(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum(axis=0)
    y_true = y_true.sum(axis=0)
    y_pred = y_pred.sum(axis=0)
    score = (2 * prod) / (y_true + y_pred)
    score = np.mean(score)
    return score

# micro Precision
def micro_precision(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_pred = y_pred.sum()
    score = prod / y_pred
    return score

# micro Recall
def micro_recall(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_true = y_true.sum()
    score = prod / y_true
    return score

# micro F1 measrue
def micro_F1_measure(y_true, y_pred):
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_true = y_true.sum()
    y_pred = y_pred.sum()
    score = (2 * prod) / (y_true + y_pred)
    return score

def get_all_score(y_true, y_pred):
    score_dict = {}
    score_dict["accuracy"] = accuracy(y_true, y_pred)
    score_dict["precision"] = precision(y_true, y_pred)
    score_dict["recall"] = recall(y_true, y_pred)
    score_dict["zero_one_loss"] = zero_one_loss(y_true, y_pred)
    score_dict["hamming_loss"] = hamming_loss(y_true, y_pred)
    score_dict["F1_measure"] = F1_measure(y_true, y_pred)
    score_dict["exact_match_ratio"] = exact_match_ratio(y_true, y_pred)

    score_dict["macro_precision"] = macro_precision(y_true, y_pred)
    score_dict["macro_recall"] = macro_recall(y_true, y_pred)
    score_dict["macro_F1_measure"] = macro_F1_measure(y_true, y_pred)

    score_dict["micro_precision"] = micro_precision(y_true, y_pred)
    score_dict["micro_recall"] = micro_recall(y_true, y_pred)
    score_dict["micro_F1_measure"] = micro_F1_measure(y_true, y_pred)
    return score_dict
