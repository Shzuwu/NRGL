# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:35:35 2024

@author: Administrator
"""

from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import pickle as pkl

def normalize_row(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

def normalize_col(mx):
    """Row-normalize sparse matrix"""
    colmean = np.array(mx.mean(0))
    c_inv = np.power(colmean, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = torch.tensor(np.diag(c_inv))
    mx = np.array(mx).dot(c_mat_inv)
    return mx

class Dataset:
    def __init__(self, name, homo=True, anomaly_alpha=None, anomaly_std=None,use_gpu=None,gpu_device=None):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == 'tsocial':
            graph, label_dict = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name =='sichuan_fei':
            graph = dgl.load_graphs("/data01/social_network_group/m21_huangzijun/sichuan/data_after/tele_max_graph.bin")
            graph = graph[0][0]['second-order']
            graph.ndata['feature'] = torch.FloatTensor(normalize_col(graph.ndata['feature']))
            pass
        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == "elliptic":
            f=open("elliptic.dat","rb") # 已经带方向了
            data=pkl.load(f)
            adj = data.edge_index.numpy()
            features=torch.FloatTensor(data.x.numpy())
            labels = torch.LongTensor(data.y.numpy())
            graph = dgl.graph((adj[0],adj[1]),num_nodes=len(labels))
            graph.ndata['feature']=features
            graph.ndata['label']=labels
            if homo:
                graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'])
                graph = dgl.add_self_loop(graph)
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        # graph.ndata['feature'] = 
        print(graph)
        if use_gpu:
            graph=graph.to(gpu_device)
        self.graph = graph
