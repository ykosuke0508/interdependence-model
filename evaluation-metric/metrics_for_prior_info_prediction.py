import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.svm as svm
import pandas as pd
import copy
import time
import random
import warnings
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

# 入力がndarray形式かどうかを確認
def _isndarray(instance):
    return isinstance(instance, np.ndarray)

# 全ての要素が0/1になっているかを確認
def _is_zero_one(ndarray):
    return np.all(np.logical_or(ndarray == 1, ndarray == 0))

# ndarrayの内部の要素の形式をint型に変更
def _change2int(ndarray):
    return ndarray.astype(np.int32)

def _format_check(instance, name):
    if _isndarray(instance) and _is_zero_one(instance):
        return _change2int(instance)
    else:
        raise ValueError("The format of {0} is incorrect.\n{0} is numpy.ndarray, and {0}'s elements are 0/1 only.")

def _previous_check(y_true, y_pred):
    y_true = _format_check(y_true, "y_true")
    y_pred = _check_unable_predict(y_pred, "y_pred")
    if np.any(y_true.sum(axis=1) == 0):
        warnings.warn("There is no-labeled instance in ground truth.")
    return y_true, y_pred

def _is_unable_LP(ndarray):
    if np.any(ndarray < 0) or np.any(ndarray.astype(str) == "-0.0"):
        return True
    else:
        return False

def _check_unable_predict(instance, name):
    if _isndarray(instance):
        # LP methodを厳密に適用できなかった場合には警告を発する。
        if _is_unable_LP(instance):
            warnings.warn("You cannot strictly apply LP method to this data.")
            # 厳密に適用できなかった場合には負の数で答えを保持しているので、正の数に直しておく。
            instance = np.absolute(instance)

        if _is_zero_one(instance):
            return _change2int(instance)
        else:
            raise ValueError("The format of {0} is incorrect.\n{0} is numpy.ndarray, and {0}'s elements are 0/1 only.")
    else:
        raise ValueError("The format of {0} is incorrect.\n{0} is numpy.ndarray, and {0}'s elements are 0/1 only.")


"""
Example-based metric
"""
# accuracy
def accuracy(y_true, y_pred, prior_info):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : andをとった結果
    cap = y_true * y_pred
    cap = np.where(prior_info == prior_info, 0, cap)
    # cup : orをとった結果
    cup = y_true + y_pred - cap
    cup = np.where(prior_info == prior_info, 0, cup)
    # 各インスタンスごとにandとorをとったラベルの数を数える。
    cap = cap.sum(axis=1)
    cup = cup.sum(axis=1)
    score = cap / cup
    score = np.where(cup == 0, 1, score)
    score = np.mean(score)
    return score

# precision
def precision(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : andをとった結果
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_pred = y_pred.sum(axis=1)

    score = cap / y_pred
    # TODO: もし、scoreの要素にnanがある場合にはwarningを出す。
    # nanが出てきたら0とみなして計算する。
    score = np.where(score == score, score, 0)
    score = np.mean(score)
    return score

# recall
def recall(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : andをとった結果
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_true = y_true.sum(axis=1)

    score = cap / y_true
    # TODO: もし、scoreの要素にnanがある場合にはwarningを出す。
    # nanが出てきたら1とみなして計算する。
    score = np.where(score == score, score, 1)
    score = np.mean(score)
    return score

# F1 measure
def F1_measure(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    # cap : andをとった結果
    cap = y_true * y_pred
    cap = cap.sum(axis=1)
    y_true = y_true.sum(axis=1)
    y_pred = y_pred.sum(axis=1)
    score = (2 * cap) / (y_true + y_pred)
    score = np.mean(score)
    return score

# hamming loss
def hamming_loss(y_true, y_pred, prior_info):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    score = np.where(y_true != y_pred, 1, 0)
    score = np.where(prior_info == prior_info, 0, score)
    #print(score)
    n_labels = np.where(prior_info == prior_info, 0, 1)
    #print(n_labels)
    score = score.sum() / n_labels.sum()
    return score

# 0/1 loss
def zero_one_loss(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    miss = np.where(y_true != y_pred, 1, 0).sum(axis = 1)
    score = np.where(miss > 0, 1, 0)
    score = np.mean(score)
    return score

# Exact Match Ratio
def exact_match_ratio(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
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
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum(axis=0)
    y_pred = y_pred.sum(axis=0)
    score = prod / y_pred
    score = np.mean(score)
    return score

# macro Recall
def macro_recall(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum(axis=0)
    y_true = y_true.sum(axis=0)
    score = prod / y_true
    score = np.mean(score)
    return score

# macro F1 measure
def macro_F1_measure(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
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
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_pred = y_pred.sum()
    score = prod / y_pred
    return score

# micro Recall
def micro_recall(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_true = y_true.sum()
    score = prod / y_true
    return score

# micro F1 measrue
def micro_F1_measure(y_true, y_pred):
    # y_trueとy_predがいずれも0/1のみをとるndarray形式であることを確認。
    y_true, y_pred = _previous_check(y_true, y_pred)
    prod = y_true * y_pred
    prod = prod.sum()
    y_true = y_true.sum()
    y_pred = y_pred.sum()
    score = (2 * prod) / (y_true + y_pred)
    return score



# TODO: 未実装
# logloss
# auc

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

def main():
    dname = "scene"
    blk_num = 0
    split_path = "pickle_data/{}_10_split.pickle".format(dname)
    y_path = "/Users/admin/Dropbox/Thesis/Masters_Thesis/experimental_data/{0}/{0}_labels.csv".format(dname)
    split_list = np.load(split_path)
    y = pd.read_csv(y_path)
    y = np.array(y)
    y = y[split_list[blk_num]]
    # blkとnp_instの数字が逆になってしまっている。
    # 修正するのは時間がかかるので、これを間違えないようにする。
    pred_path = "/Users/admin/Dropbox/Thesis/Masters_Thesis/results_for_final/results_prior_info_setting/{0}/not_same/{0}_BR_c1_blk_{1}_{2}p_inst.pickle".format(dname, 1, blk_num)
    pred = np.load(pred_path)
    score_dict = get_all_score(y, pred)
    for k, v in score_dict.items():
        print("{} : {}".format(k,v))

if __name__ == '__main__':
    main()
