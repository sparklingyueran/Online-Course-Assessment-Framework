from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

import os
import json
import numpy as np
import pandas as pd

## 1. Load Data
score_dict = np.load('score_dict.npy', allow_pickle=True).item()
speech_emo_dict = np.load('speech_emo_dict.npy', allow_pickle=True).item()
speech_emo_mean_dict = np.load('speech_emo_mean_dict.npy', allow_pickle=True).item()
emo_class_dict= np.load('emo_class_dict.npy', allow_pickle=True).item()
diari_per_dict = np.load('diari_per_dict.npy',allow_pickle=True).item()
asr_speed_mean_dict = np.load('asr_speed_mean_dict.npy', allow_pickle=True).item()
asr_count_mean_dict =np.load('asr_count_mean_dict.npy', allow_pickle=True).item()
asr_count_sum_dict =np.load('asr_count_sum_dict.npy', allow_pickle=True).item()
asr_count_max5_dict =np.load('asr_count_max5_dict.npy', allow_pickle=True).item()
token_set_dict =np.load('token_set_dict.npy', allow_pickle=True).item()
token_mean_dict =np.load('token_mean_dict.npy', allow_pickle=True).item()
asr_ttr_dict =np.load('asr_ttr_dict.npy', allow_pickle=True).item()
weighted_sum_dict = np.load('weighted_sum_dict.npy', allow_pickle=True).item()
weighted_seq_dict = np.load('weighted_seq_dict.npy', allow_pickle=True).item()
tfidf_vec_dict_corpus = np.load('tfidf_vec_dict_corpus.npy', allow_pickle=True).item()
emotion_dict = np.load('emotion_dict.npy', allow_pickle=True).item()
valence_dict = np.load('valence_dict.npy', allow_pickle=True).item()
arousal_dict = np.load('arousal_dict.npy', allow_pickle=True).item()
gaze_dict = np.load('gaze_dict.npy', allow_pickle=True).item()
pose_dict = np.load('pose_dict.npy', allow_pickle=True).item()

## 2. Merge Data
dict_ls = [
speech_emo_mean_dict, #5
emo_class_dict,  #5
diari_per_dict, #5
asr_speed_mean_dict , #1
asr_count_mean_dict , #1
asr_count_sum_dict , #1
asr_count_max5_dict ,#1
token_set_dict , #1
token_mean_dict, #1
asr_ttr_dict ,#1
emotion_dict ,
valence_dict ,
arousal_dict, 
gaze_dict ,
pose_dict
]
ls_set = get_video_set()

def get_video_set():
    ls_set = set(list(score_dict.keys()))
    for dic in dict_ls:
        ls_set = ls_set & set(list(dic.keys()))
    return ls_set

ls_set = get_video_set()
def get_x_y(video_id,index,threshold):
    x = np.array([])
    for dic in dict_ls:
        x = np.append(x,dic[video_id])
    y_0 = score_dict[video_id][index]
    if y_0 < threshold:
        y = 1
    else:
        y = 0
    return x,y
def get_ls(index,threshold):
    x_ls = []
    y_ls = []
    for video_id in ls_set:
        x,y = get_x_y(video_id,index,threshold)
        x_ls.append(x)
        y_ls.append(y)
    return x_ls,y_ls

def get_x_y_all(video_id,index):
    x = np.array([])
    for dic in dict_ls:
        x = np.append(x,dic[video_id])
    y= score_dict[video_id][index]
    return x,y
def get_ls_all(index):
    x_ls = []
    y_ls = []
    for video_id in ls_set:
        x,y = get_x_y_all(video_id,index)
        x_ls.append(x)
        y_ls.append(y)
    return x_ls,y_ls

## 3. Binary Classification


def fit_model_bi(x_ls,y_ls):
    y_true = []
    y_pred = []
    prob_ls = []
    for i in range(len(x_ls)):
        X_test = [x_ls[i]]
        X_train = x_ls[:i]+x_ls[i+1:]
        y_test = y_ls[i]
        y_train = y_ls[:i]+y_ls[i+1:]
    #=============================================#
        clf = SVC(probability = True)
    #===========================================================#
        clf.fit(X_train, y_train)
        output_pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)
        y_pred.append(output_pred[0])
        y_true.append(y_test)
        prob_ls.append(prob[0])
#     print(prob_ls)
    return y_true,y_pred,prob_ls

def get_report(y_multi):
    multi={}
    for key in y_multi.keys():
        y_true = y_multi[key][0]
        y_pred = y_multi[key][1] 
        t = classification_report(y_true, y_pred,digits = 4)
        print(t)
def get_matrix_bi(y_multi):
    multi={}
    for key in y_multi.keys():
#         (y_true,y_pred) = y_multi[key]
        y_true = y_multi[key][0]
        y_pred = y_multi[key][1]
        con_mat = confusion_matrix(y_true, y_pred)   
        score = accuracy_score(y_true,y_pred)
        TN = con_mat[0][0]
        FN = con_mat[0][1]
        FP = con_mat[1][0]
        TP = con_mat[1][1]
        Recall = TP/(TP+FN)
#         specificity = TN/(TN+FP)
        Precision = TP/(TP+FP)
        F1 = 2*(Precision*Recall)/(Precision+Recall)
        multi[key]=(con_mat,score,Precision,Recall,F1)
    return multi



# fit svm random oversampling
y_bi_clf_dict_mm_svm_os = {}
for i in range(5):
    mm = MinMaxScaler()
    index = i
    threshold = 3
    x_ori,y_ori = get_ls(index,threshold)
    x_trans = list(mm.fit_transform(x_ori))
    ros = RandomOverSampler(random_state=0)
    x_ay,y_ay = ros.fit_sample(x_trans, y_ori)
    x_ls = list(x_ay)
    y_ls = list(y_ay)

    y_true,y_pred,prob_ls = fit_model_bi(x_ls,y_ls)
#     print(prob_ls)
    y_bi_clf_dict_mm_svm_os[index] = (y_true,y_pred,prob_ls)

bi_clf_dict_mm_svm_os = get_matrix_bi(y_bi_clf_dict_mm_svm_os)





## 4. Multiclass  Classification
def fit_model(x_ls,y_ls):
    y_true = []
    y_pred = []
    for i in range(len(x_ls)):
        X_test = [x_ls[i]]
        X_train = x_ls[:i]+x_ls[i+1:]
        y_test = y_ls[i]
        y_train = y_ls[:i]+y_ls[i+1:]
    #=========#此处修改模型#=====================================#
        clf = SVC(decision_function_shape='ovo')
        clf.decision_function_shape = "ovr"
    #===========================================================#
        output_pred = clf.fit(X_train, y_train).predict(X_test)
        y_pred.append(output_pred[0])
        y_true.append(y_test)
    return y_true,y_pred


# fit svm randomversampling

y_multi_clf_dict_mm_svm_os = {}
for i in range(5):
    mm = MinMaxScaler()
    index = i
    x_ori,y_ori = get_ls_all(index)
    x_trans = list(mm.fit_transform(x_ori))
    sample_solver= RandomOverSampler(random_state=0)
    x_ay,y_ay = sample_solver.fit_sample(x_trans,y_ori)
    x_ls = list(x_ay)
    y_ls = list(y_ay)
    y_true,y_pred = fit_model(x_ls,y_ls)
    y_multi_clf_dict_mm_svm_os[index] = (y_true,y_pred)

def get_matrix(y_multi):
    multi={}
    for key in y_multi.keys():
        (y_true,y_pred) = y_multi[key] 
        con_mat = confusion_matrix(y_true, y_pred)   
        score = accuracy_score(y_true,y_pred)
        multi[key]=(con_mat,score)
    return multi


def get_clf_rpt(y_multi):
    multi={}
    for key in y_multi.keys():
        (y_true,y_pred) = y_multi[key] 
#         print(y_true)
#         print(y_pred)
        t = classification_report(y_true, y_pred,digits = 4)
        print(t)

multi_clf_dict_mm_svm_os = get_matrix(y_multi_clf_dict_mm_svm_os)
get_clf_rpt(y_multi_clf_dict_mm_svm_os)


