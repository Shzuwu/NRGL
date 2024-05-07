# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:36:00 2024

@author: Administrator
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
import numpy as np
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from utils import *
from model import *
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import f1_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def ContrastiveLoss(z_i, z_j, batch_size, temperature, negatives_mask):

    representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
    
    sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
    sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
    positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
    
    nominator = torch.exp(positives / temperature)             # 2*bs
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)             # 2*bs, 2*bs

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss

def train_division(feature, edges, idx_train, label_noise, epoch, temperature, negatives_mask, w=0.5):
    discriminator.train()
    criterion.train()
    
    adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
    emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
    emb_lp = emb_lp.detach()
    emb_hp = emb_hp.detach()
    
    index, score, y_train_noise = criterion.division(idx_train, emb_lp, emb_hp, label_noise, epoch)
    xx = torch.ones(len(idx_train)).to(device)
    xx[index] = score
    xx = xx.detach()
    a=np.array(range(len(idx_train)))
    
    num_batch = int(len(idx_train)/batch_size)
    weight = torch.tensor([1,((1-y_train_noise).sum() / y_train_noise.sum())]).to(device)
    loss_list = []
    
    node_prob = node_probability1(graph, idx_train, y_train_noise, w)
    
    for i in range(num_batch):
        optimizer.zero_grad()
        
        adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
        emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
        
        x_lp, x_hp = criterion(emb_lp[idx_train], emb_hp[idx_train])
        
        new_train_batch = np.random.choice(a, size = batch_size, replace=True, p=node_prob).tolist()
        
        z_i = emb_lp[idx_train][new_train_batch]
        z_j = emb_hp[idx_train][new_train_batch]
        
        loss1 =  ContrastiveLoss(z_i, z_j, batch_size, temperature, negatives_mask)
        
        score_temp = xx[new_train_batch]
        loss_lp = torch.mean(score_temp*F.cross_entropy(x_lp[new_train_batch], y_train_noise[new_train_batch], reduction='none'))
        loss_hp = torch.mean(score_temp*F.cross_entropy(x_hp[new_train_batch], y_train_noise[new_train_batch], reduction='none'))
        loss = loss1 + (loss_lp + loss_hp)/2
        
        loss.backward()
        loss_list.append(np.array(loss.cpu().data))
        optimizer.step()
        
    loss_result = np.mean(loss_list)
    return loss_result

def dataset_split(dataset, labels, seed=2):
    if dataset == 'amazon':
        index = list(range(3305, len(labels)))
    else:
        index = list(range(len(labels)))
    
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], 
                                                            train_size=args.train_ratio,
                                                            random_state=seed, shuffle=True)
    idx_val, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.6667,
                                                            random_state=2, shuffle=True)
    return idx_train, idx_val, idx_test


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="elliptic", help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
parser.add_argument("--homo", type=int, default=1, help="1 for Homo and 0 for Hetero")
parser.add_argument("--epochs", type=int, default=50, help="The max number of epochs")
parser.add_argument("--run", type=int, default=3, help="Running times")
parser.add_argument("--noise", type=bool, default=True)
parser.add_argument("--ptb_rate", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--nclass", type=int, default=2)
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument("--use_gpu", type=bool, default=False)
parser.add_argument("--gpu_device",type=int,default=0)
args = parser.parse_args()

if args.use_gpu:
    torch.cuda.set_device(args.gpu_device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph = Dataset(args.dataset, args.homo, use_gpu = args.use_gpu,gpu_device = 0).graph

features = graph.ndata['feature']
labels = graph.ndata['label']
edges = torch.cat([graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)], dim=0)
nnodes = features.shape[0]
input_dim = features.shape[1]
idx_train, idx_val, idx_test = dataset_split(args.dataset, labels.cpu(), args.seed)

label_noise, train_label, train_label_noise = noisify_with_P1(labels,idx_train=idx_train,noise_ratio=args.ptb_rate,random_state=args.seed,nclass=args.nclass)


features = features.to(device)
label_noise = label_noise.to(device)
edges = edges.to(device)

# model
negatives_mask = ~torch.eye(args.batch_size * 2, args.batch_size * 2, dtype=bool).to(device)
discriminator = Edge_Discriminator(args.nlayers, nnodes, input_dim, args.hidden_dim, input_dim, args.hidden_dim, 0.1, 0.5).to(device)
criterion = LabelDivision(args, input_dim).to(device)

# optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.05)
# optimizer_criterion = torch.optim.Adam(criterion.parameters(), lr=args.lr)
optimizer = torch.optim.Adam([{'params':  discriminator.parameters(), 'lr':5*args.lr}, {'params':criterion.parameters()}], lr=args.lr)

f1_best_lp = 0
f1_best_hp = 0
temperature = 0.5
batch_size = args.batch_size
auc_best_lp = 0
auc_best_hp = 0

print('Begin to training')
for epoch in range(1, args.epochs):
    # loss1 = train_discriminator1(discriminator, features, edges, args, batch_size, temperature, negatives_mask)
    loss2 = train_division(features, edges, idx_train, label_noise, epoch, temperature, negatives_mask)
    print('epoch;{}, loss2:{}'.format(epoch, loss2))

    if epoch % 1 == 0:
        discriminator.eval()
        criterion.eval()
        adj_lp, adj_hp, weights_lp, weights_hp = discriminator(features, edges)
        emb_lp, emb_hp = discriminator.get_embedding(features, adj_lp, adj_hp)
        
        x_lp_score, x_hp_socre = criterion.to_prob(emb_lp, emb_hp)
        str1 = 'x_lp_score'
        str2 = 'x_hp_socre'
    
        auc, precision, recall, f1 = test_model1(x_lp_score, idx_test, label_noise, str1)
        auc1, precision1, recall1, f11 = test_model1(x_hp_socre, idx_test, label_noise, str2)
        
        if (auc+f1)>auc_best_lp:
            auc_best_lp = auc + f1
            a_lp= [epoch, auc, precision, recall, f1]
            
        if (auc1+f11)>auc_best_hp:
            auc_best_hp = auc1 + f11
            a_hp= [epoch, auc1, precision1, recall1, f11]
        
print('LP: epoch:{}, auc:{}, recall:{}, F1:{}'.format(a_lp[0], a_lp[1], a_lp[3], a_lp[4]))
print('HP: epoch:{}, auc:{}, recall:{}, F1:{}'.format(a_hp[0], a_hp[1], a_hp[3], a_hp[4]))
print('args:', args)
print('train (doule loss)')












