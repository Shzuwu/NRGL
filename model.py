# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:35:07 2024

@author: Administrator
"""

import torch
import os
import torch.nn as nn
import dgl
from scipy import sparse
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl.function as fn
import random
from dgl.nn import EdgeWeightNorm
import numpy as np

EOS = 1e-10
norm = EdgeWeightNorm(norm='both')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
norm = EdgeWeightNorm(norm='both')

class Edge_Discriminator(nn.Module):
    def __init__(self, nlayers, nnodes, in_dim, emb_dim, input_dim, hidden_dim, alpha, dropout=0.5, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()
        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(2*hidden_dim, 1)
        self.nnodes = nnodes
        self.temperature = temperature
        self.bias = bias
        self.alpha = alpha
        
        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout) 
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout)
        
    def get_embedding(self, features, adj_lp, adj_hp, source='all'):
        emb_lp = self.encoder1(features, adj_lp)
        emb_hp = self.encoder2(features, adj_hp)
        return emb_lp, emb_hp

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h
    
    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1+s2)/2
        
    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()
    
    def weight_forward(self, features, edges):
        embeddings = self.get_node_embedding(features)                             
        edges_weights_raw = self.get_edge_weight(embeddings, edges)                 
        weights_lp = self.gumbel_sampling(edges_weights_raw)                        
        weights_hp = 1 - weights_lp                                                 
        return weights_lp, weights_hp
    
    def weight_to_adj(self, edges, weights_lp, weights_hp):                        
        adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device=device)            
        adj_lp = dgl.add_self_loop(adj_lp)
        weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).to(device))) + EOS 
        # weights_lp = weights_lp + EOS
        weights_lp = norm(adj_lp, weights_lp)
        weights_lp[-self.nnodes:] = 1
        adj_lp.edata['w'] = weights_lp
        
        adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device=device)
        adj_hp = dgl.add_self_loop(adj_hp)
        weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).to(device))) + EOS
        # weights_hp = weights_hp + EOS
        weights_hp = norm(adj_hp, weights_hp)
        weights_hp *= - self.alpha                                                  
        weights_hp[-self.nnodes:] = 1 
        adj_hp.edata['w'] = weights_hp 
        return adj_lp, adj_hp
        
    def forward(self, features, edges):
        weights_lp, weights_hp = self.weight_forward(features, edges)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp) 
        return adj_lp, adj_hp, weights_lp, weights_hp

class SGC(nn.Module):
    def __init__(self, nlayer, in_dim, emb_dim, dropout):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.k = nlayer
        self.linear = nn.Linear(in_dim, emb_dim)
       
    def forward(self, x, g):
        x_list = [x]
        x = torch.relu(self.linear(x))
        
        with g.local_scope():
            g.ndata['h'] = x
            for _ in range(self.k):
                g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                x_list.append(g.ndata['h'])
            result = torch.cat(x_list,dim=1)
            return result
        
class LabelDivision(nn.Module):
    def __init__(self, args, input_dim):
        super(LabelDivision, self).__init__()
        self.args = args
        self.epochs = args.epochs
        
        self.gate1 = nn.Linear(args.hidden_dim*args.nlayers+input_dim, 2).to(device)
        self.gate2 = nn.Linear(args.hidden_dim*args.nlayers+input_dim, 2).to(device)
        torch.nn.init.xavier_uniform_(self.gate1.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.gate2.weight, gain=1.414)
        
        self.increment = 0.5/args.epochs
    
    def to_prob(self, emb_lp, emb_hp):        
        x_lp = self.gate1(emb_lp)
        x_hp = self.gate2(emb_hp)
        
        return x_lp, x_hp
        
    def forward(self, z_lp, z_hp):
        x_lp = self.gate1(z_lp)
        x_hp = self.gate2(z_hp)
        return x_lp, x_hp
        
    def division(self, idx_train, emb_lp, emb_hp, label_noise, epoch):
        z_lp = emb_lp[idx_train]
        z_hp = emb_hp[idx_train]
        
        x_lp, x_hp = self.forward(z_lp, z_hp)
        
        y_train_noise = label_noise[idx_train]
        
        loss_pick_lp = F.cross_entropy(x_lp, y_train_noise, reduction='none')
        loss_pick_hp = F.cross_entropy(x_hp, y_train_noise, reduction='none')
        loss_pick = loss_pick_lp + loss_pick_hp
        
        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = self.increment*epoch
        remember_rate = 1 - forget_rate
        
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted < mean_v)[0]
        remember_rate_small = idx_small.shape[0]/y_train_noise.shape[0]
        remember_rate = max(remember_rate, remember_rate_small)
        
        num_remember = int(remember_rate * len(loss_sorted))
        
        '''Loss for clean labels'''
        label_clean = ind_sorted[:num_remember]
        label_clean_score = torch.ones(len(label_clean)).to(device)
        # weight1 = torch.sum(1-y_train_noise[label_clean]) /(1+torch.sum(y_train_noise[label_clean]))
        # label_clean_loss = F.cross_entropy(x_lp[label_clean], y_train_noise[label_clean],weight=torch.tensor([1., weight1]).to(device)) + F.cross_entropy(x_hp[label_clean], y_train_noise[label_clean],weight=torch.tensor([1., weight1]).to(device))
        # label_clean_loss = 0.5 * label_clean_loss * label_clean_score                  
        
        ind_all = torch.arange(y_train_noise.shape[0]).long() 
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(label_clean.detach().cpu().numpy()))).to(device)
        
        p_1 = F.softmax(x_lp, dim=-1)
        p_2 = F.softmax(x_hp, dim=-1)
        result = p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1]
        
        filter_condition = ((x_lp.max(dim=1)[1][ind_update_1] != y_train_noise[ind_update_1]) &
                            (x_lp.max(dim=1)[1][ind_update_1] == x_hp.max(dim=1)[1][ind_update_1]) &
                            (result > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.args.epochs)))
        
        label_correct = ind_update_1[filter_condition]
        y_train_noise[label_correct] = 1 - y_train_noise[label_correct]
        label_correct_score = torch.mean(torch.sqrt(result[filter_condition]))*torch.ones(len(label_correct)).to(device)
        # weight2 = torch.sum(y_train_noise[label_clean])/(1+torch.sum(1-y_train_noise[label_clean]))
        # label_correct_loss = F.cross_entropy(x_lp[label_correct], (1 - y_train_noise[label_correct]), weight=torch.tensor([1., weight2]).to(device)) + F.cross_entropy(x_hp[label_correct], (1 - y_train_noise[label_correct]), weight=torch.tensor([1., weight2]).to(device))
        # label_correct_loss = 0.5 * label_correct_loss *label_correct_score 
        
        label_remain = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(label_correct.detach().cpu().numpy()))).to(device)
        label_remain_score = 0.5 * torch.ones(len(label_remain)).to(device)
        # weight3 = torch.sum(1-y_train_noise[label_remain])/(1+torch.sum(y_train_noise[label_remain]))
        # label_remain_loss = F.cross_entropy(x_lp[label_remain], 1 - y_train_noise[label_remain], weight=torch.tensor([1., weight3]).to(device)) + F.cross_entropy(x_hp[label_remain], 1 - y_train_noise[label_remain], weight=torch.tensor([1., weight3]).to(device))
        # label_remain_loss = 0.5 * label_remain_loss * label_remain_score
        
        index = torch.cat((label_clean, label_correct, label_remain))
        score = torch.cat((label_clean_score, label_correct_score, label_remain_score))
        
        return index, score, y_train_noise
          
        
        
        
        