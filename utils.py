# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:34:47 2024

@author: Administrator
"""

import dgl
import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BW import *
from sklearn.model_selection import train_test_split

def negative_node(nnodes):
    a = np.array(range(nnodes))
    p = torch.LongTensor(a)
    n1 = torch.LongTensor(np.random.permutation(a))
    n2 = torch.LongTensor(np.random.permutation(a))
    n3 = torch.LongTensor(np.random.permutation(a))
    return p, n1, n2, n3

def test_model(probs, idx_test, labels, str0, thres):
    labels = labels.cpu().numpy()
    
    preds = np.zeros_like(labels)
    preds[probs[:, 1].cpu() > thres] = 1
    
    auc_gnn =         roc_auc_score(labels[idx_test], probs[idx_test][:,1].detach().cpu())
    precision_gnn = precision_score(labels[idx_test], preds[idx_test], average="macro")
    recall_gnn =       recall_score(labels[idx_test], preds[idx_test], average="macro")
    f1_gnn =               f1_score(labels[idx_test], preds[idx_test], average="macro")
    
    print(str0 + 'Auc: {}, Recall: {}, F1-score:{}'.format(auc_gnn, recall_gnn, f1_gnn))
    
    return auc_gnn, precision_gnn, recall_gnn, f1_gnn

def test_model1(result, idx_val, label, str0=''):

    labels = label[idx_val].cpu().numpy()
    result = F.softmax(result, dim=1)
    gnn_prob = result[idx_val]
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1_gnn = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    
    print(str0 + ': Auc: {}, Recall: {}, F1-score:{}'.format(auc_gnn, recall_gnn, f1_gnn))
    return auc_gnn, precision_gnn, recall_gnn, f1_gnn

def noisify_with_P1(label, idx_train, noise_ratio, random_state, nclass):

    train_label = label[idx_train]
    P = np.float64(noise_ratio) / np.float64(nclass - 1) * np.ones((nclass, nclass))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise_ratio)) * np.ones(nclass))

    train_num = len(train_label)
    flipper = np.random.RandomState(random_state)
    train_label_noise = np.array(train_label)

    for idx in range(train_num):
        i = train_label[idx]
        random_result = flipper.multinomial(1, P[i, :], 1)[0]
        train_label_noise[idx] = np.where(random_result == 1)[0]

    actual_noise = (np.array(train_label_noise) != np.array(train_label)).mean()
    print('actual_noise: {}'.format(actual_noise))
    train_idx_noise = np.array(idx_train)[np.where(train_label_noise != np.array(train_label))]
    train_idx_clean = np.array(idx_train)[np.where(train_label_noise == np.array(train_label))]
    train_label_noise = torch.tensor(train_label_noise)

    label_noise = np.array(label)
    label_noise[idx_train] = train_label_noise
    label_noise = torch.tensor(label_noise)
    train_idx_noise = torch.tensor(train_idx_noise)
    train_idx_clean = torch.tensor(train_idx_clean)

    return label_noise, train_label, train_label_noise

def node_probability1(g, idx_train, y_train, w):
    node_degree = (g.in_degrees()).clamp(min=1)
    node_degree = np.array(node_degree.cpu())
    fraud_rate = torch.sum(y_train)/len(y_train)
    norm_rate = 1 - fraud_rate
    node_prob = np.zeros(len(y_train))
    for i in range(len(y_train)):
        if y_train[i] == 1:
            node_prob[i] = node_degree[i]/fraud_rate
        elif y_train[i] == 0:
            node_prob[i] = node_degree[i]/norm_rate
    node_prob = np.array([np.power(i,0.5) for i in node_prob])
    node_prob = node_prob/np.sum(node_prob)
    return node_prob

def node_probability2(g, idx_train, y_train, w):
    node_degree = (g.in_degrees()).clamp(min=1)
    node_degree = np.array(node_degree.cpu())
    fraud_rate = torch.sum(y_train)/len(y_train)
    norm_rate = 1 - fraud_rate
    node_prob = np.zeros(len(y_train))
    for i in range(len(y_train)):
        if y_train[i] == 1:
            node_prob[i] = node_degree[idx_train[i]]/fraud_rate
        elif y_train[i] == 0:
            node_prob[i] = node_degree[idx_train[i]]/norm_rate
    node_prob = np.array([np.power(i,0.5) for i in node_prob])
    node_prob = node_prob/np.sum(node_prob)
    return node_prob

def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]  
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges