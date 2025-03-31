import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import functools
from torch.nn import MSELoss, Linear, LayerNorm, BatchNorm1d, Dropout, ReLU, Sigmoid
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group
from Device import device
from net_utils import *


class CONT2E_Net(torch.nn.Module): #pool_seq:["GMPool_G", "SelfAtt", "GMPool_I"]
    def __init__(self, 
                 in_features=None, 
                 edge_dim=None, 
                 bias=None,
                 linear_block_dims=None, 
                 conv_block_dims=None, 
                 adj_conv=None, 
                 conv=None, 
                 dropout=None, 
                 pool=None, 
                 pool_dropout=None,
                 pool_ratio=None, 
                 pool_heads=None, 
                 pool_seq=None,
                 pool_layer_norm=None, 
                 pool_type=None):

        super(CONT2E_Net, self).__init__()
        
        linear_block_dims = linear_block_dims if linear_block_dims else []
        linear_block_dims = linear_block_dims + [conv_block_dims[0]]
        self.num_linear_layers = len(linear_block_dims)
        linear_block_dims = [in_features] + linear_block_dims

        self.num_conv_layers = len(conv_block_dims)
        conv_block_dims = [linear_block_dims[-1]] + conv_block_dims

        self.edge_dim = edge_dim
        self.conv = conv

        self.adj_conv = adj_conv    
        self.dropout = dropout
        self.pool_type = pool_type

        self.linear_block = torch.nn.ModuleList([Linear(linear_block_dims[n], linear_block_dims[n+1], bias=bias) 
                                                 for n in range(self.num_linear_layers)])
        if self.edge_dim:
            print("Edge_involved in Net!")
            self.conv_block = torch.nn.ModuleList([self.conv(conv_block_dims[n], conv_block_dims[n+1], edge_dim = edge_dim, bias=bias, dropout=self.dropout) 
                                                   for n in range(self.num_conv_layers)])
        else:
            self.conv_block = torch.nn.ModuleList([self.conv(conv_block_dims[n], conv_block_dims[n+1], bias=bias, dropout=self.dropout) 
                                                   for n in range(self.num_conv_layers)])
        if self.adj_conv:
            self.conv_block = torch.nn.ModuleList([Linear(conv_block_dims[n], conv_block_dims[n+1], bias=bias, dropout=self.dropout) 
                                                   for n in range(self.num_conv_layers)])
            
        self.pool = pool(in_channels=conv_block_dims[-1], 
                         hidden_channels=conv_block_dims[-1], 
                         out_channels=1, 
                         num_nodes=300,         
                         pooling_ratio=pool_ratio, 
                         pool_sequences=pool_seq, 
                         num_heads=pool_heads, 
                         layer_norm=pool_layer_norm,
                         Conv=conv)

        self.pool_dropout = torch.nn.Dropout(pool_dropout)     
        self.lin2 = Linear(in_features=conv_block_dims[-1]*4, out_features=conv_block_dims[-1]) 
        self.lin3 = Linear(in_features=conv_block_dims[-1], out_features=1)  
        self.pool_mean = global_mean_pool
        self.pool_add = global_add_pool
        self.pool_max = global_max_pool

        print(f'CONT2E_Net init!!!conv:{self.conv}')                                                 
        
    def forward(self, data, return_embedding=False, mode='eval'):  # or "eval"
        if self.edge_dim:
            edge_attr = data.edge_attr_pred if mode == 'eval' else data.edge_attr_real
        out = data.x
        
        for layer in self.linear_block:
            out = F.relu(layer(out))

        if self.adj_conv:
            for (adj_layer, conv_layer) in zip(self.adj_block, self.conv_block):
                out = F.relu(adj_layer(out))
                out = F.relu(conv_layer(out, data.edge_index))
        else:
            for conv_layer in self.conv_block:
                if self.edge_dim:
                    out = F.relu(conv_layer(out, data.edge_index, edge_attr))
                else:
                    out = F.relu(conv_layer(out, data.edge_index))
        if self.edge_dim:
            out_tsfm = self.pool(x=out, batch=data.batch, edge_index=data.edge_index, edge_attr=edge_attr)
        else:
            out_tsfm = self.pool(x=out, batch=data.batch, edge_index=data.edge_index)

        out_mean = self.pool_mean(out, data.batch)
        out_add = self.pool_add(out, data.batch)
        out_max = self.pool_max(out, data.batch)

        if self.pool_type=='all':
            out1 = torch.cat((out_tsfm, out_mean, out_max, out_add),dim=1)
            out1 = self.lin2(out1)
            if return_embedding:
                return out1
            out = self.lin3(out1)
        elif self.pool_type=='tsfm':
            if return_embedding:
                return out_tsfm
            out = self.lin3(out_tsfm)
        elif self.pool_type=='max':
            if return_embedding:
                return out_max
            out = self.lin3(out_max)
        elif self.pool_type=='add':
            if return_embedding:
                return out_add
            out = self.lin3(out_add)
        elif self.pool_type=='mean':
            if return_embedding:
                return out_mean
            out = self.lin3(out_mean)
        out = self.pool_dropout(out)
        
        return out.view(-1)


class POS2COHP_Net(torch.nn.Module):
    def __init__(self, 
                 in_feats, 
                 hidden_feats=None, 
                 activation=None, 
                 predictor_hidden_feats=None, 
                 n_tasks=None, 
                 predictor_dropout=None):
        super(POS2COHP_Net, self).__init__()

        self.gnn = GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation)
        
        gnn_out_feats = hidden_feats[-1]
        self.predict = MLPPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x, edge_index, MN_edge_index = data.x, data.edge_index, data.MN_edge_index
        node_feats = self.gnn(x, edge_index)

        edge_embedding = node_feats[MN_edge_index[0]] + node_feats[MN_edge_index[1]]
        #edge_embedding = torch.cat([node_feats[MN_edge_index[0]], node_feats[MN_edge_index[1]]], dim=1)

        predicted_feats = self.predict(edge_embedding)

        return torch.squeeze(predicted_feats)


class POS2COHP_Net_Hetero(nn.Module):
    def __init__(self, 
                 atom_feats, 
                 bond_feats, 
                 hidden_feats=None, 
                 activation=None, 
                 predictor_hidden_feats=None, 
                 n_tasks=None, 
                 predictor_dropout=None):
        super(POS2COHP_Net_Hetero, self).__init__()
        
        self.uni_trans_atoms = nn.Linear(atom_feats, hidden_feats[0])
        self.uni_trans_bonds = nn.Linear(bond_feats, hidden_feats[0])

        self.gnn = Hetero_GCN(in_feats=hidden_feats[0], hidden_feats=hidden_feats, activation=activation)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = MLPPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = {k:f(v) for (k,v),f in zip(x_dict.items(), [self.uni_trans_atoms, self.uni_trans_bonds])}
        
        node_feats = self.gnn(x_dict, edge_index_dict)["cohps"]
        predicted_feats = self.predict(node_feats)
        return torch.squeeze(predicted_feats) 


class Hetero_CONT2E_Net(torch.nn.Module):
    def __init__(self, node_feats=None, edge_feats=None, bias=True,
                 linear_block_dims=[64], conv_block_dims=[96, 128, 256], adj_conv=False, 
                 conv=Hetero_GCNLayer, dropout=0.,
                 pool=Hetero_GMT, pool_dropout=0.,
                 pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G", "SelfAtt", "GMPool_I"],
                 pool_layer_norm=False, pool_type='all'):

        super(Hetero_CONT2E_Net, self).__init__()
        
        linear_block_dims = [] if linear_block_dims else linear_block_dims
        linear_block_dims.append(conv_block_dims[0])
        self.num_linear_layers = len(linear_block_dims)
        
        node_linear_block_dims = [node_feats] + linear_block_dims
        self.uni_trans_nodes = torch.nn.ModuleList([Linear(node_linear_block_dims[n], node_linear_block_dims[n+1], bias=bias) 
                                                    for n in range(self.num_linear_layers)])
        edge_linear_block_dims = [edge_feats] + linear_block_dims
        self.uni_trans_edges = torch.nn.ModuleList([Linear(edge_linear_block_dims[n], edge_linear_block_dims[n+1], bias=bias) 
                                                    for n in range(self.num_linear_layers)])
        self.uni_trans_dict = {"atoms":self.uni_trans_nodes, "cohps":self.uni_trans_edges}
        
        self.adj_conv = adj_conv    
        self.conv = conv
        self.pool_type = pool_type

        self.num_conv_layers = len(conv_block_dims)
        conv_block_dims = [linear_block_dims[-1]] + conv_block_dims
        self.conv_block = torch.nn.ModuleList([self.conv(conv_block_dims[n], conv_block_dims[n+1]) 
                                               for n in range(self.num_conv_layers)])
            
        self.pool = pool(conv_block_dims[-1], conv_block_dims[-1], 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq, num_heads=pool_heads, layer_norm=pool_layer_norm,
                         Conv=self.conv)

        self.pool_dropout = torch.nn.Dropout(pool_dropout)     
        self.lin2 = Linear(in_features=conv_block_dims[-1]*4, out_features=conv_block_dims[-1]) 
        self.lin3 = Linear(in_features=conv_block_dims[-1], out_features=1)
        print(f'CONT2E_Net init!!!conv:{self.conv}')                                                 
        
    def forward(self, data): 
        x_dict, edge_index_dict, batch_dict = data.x_dict, data.edge_index_dict, data.batch_dict
        x_dict = {k: functools.reduce(lambda v, f: f(v), self.uni_trans_dict[k], v) for k, v in x_dict.items()}
        
        for conv_layer in self.conv_block:
            x_dict = conv_layer(x_dict, edge_index_dict)

        out_tsfm = self.pool(x_dict, batch_dict, edge_index_dict)
        out = self.lin3(out_tsfm)
        return out






"""
class POS2COHP2E_Net(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout_COHP=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None
                 
                 , dim=128, adj_conv=False, in_features=None, bias=True,
                 linear_dim_list=[[128,128]],
                 conv_dim_list=[[128,128],[128,128],[128,128]],
                 conv=SAGEConv, dropout_E=0., pool=GraphMultisetTransformer, pool_dropout=0.,
                 pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G", "SelfAtt", "GMPool_I"],
                 pool_layer_norm=False, pool_type='all',
                 edge_dim = None,noise=False,noise_type=None,noise_mae=None,real_cohp=None):
        super(POS2COHP2E_Net, self).__init__()
        self.pos2cohp_net = POS2COHP_Net(in_feats, hidden_feats, 
                 activation, residual, batchnorm, dropout_COHP, 
                 predictor_hidden_feats, n_tasks, predictor_dropout)
        self.poscohp2e_net = CONT2E_Net(dim, adj_conv, in_features, bias,
                 linear_dim_list,
                 conv_dim_list,
                 conv, dropout_E, pool, pool_dropout,
                 pool_ratio, pool_heads, pool_seq,
                 pool_layer_norm, pool_type,
                 edge_dim,noise,noise_type,noise_mae,real_cohp)
    def forward(self, data):
        pred_cohp = self.pos2cohp_net(data)
        # print(pred_cohp.shape)
        # print(data.cohp_num)
        # print(data.CCorN_edge_num)
        # print(torch.sum(data.cohp_num))
        # print(pred_cohp[:10])
        edge_attr = []
        COHP_idx_l = 0
        for i,v in enumerate(data.CCorN_edge_num):
            COHP_idx_r = COHP_idx_l + data.cohp_num[i]
            edge_attr_tmp = [-5.0]*data.CCorN_edge_num[i] + pred_cohp[COHP_idx_l:COHP_idx_r].tolist()
            edge_attr += edge_attr_tmp
        edge_attr = torch.tensor(edge_attr).to(device)
        out = self.poscohp2e_net(data=data, edge_attr_serial = edge_attr)
        return (torch.tensor(pred_cohp),out)


class POS2COHP_Net_force_inverse_edge_equal(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(POS2COHP_Net, self).__init__()

        self.gnn = pyg_GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, 
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = pyg_MLPPredictor(gnn_out_feats*2, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x, edge_index, MN_edge_index = data.x, data.edge_index, data.MN_edge_index
        node_feats = self.gnn(x, edge_index)
        
        edge_embedding = torch.cat([node_feats[MN_edge_index[0]], node_feats[MN_edge_index[1]]], dim=1)

        predicted_feats = self.predict(edge_embedding)
        predicted_feats = torch.squeeze(predicted_feats)
        
        # Ensure that the prediction for adjacent edges are the same
        # Assuming MN_edge_index has pairs of edges in the format [1, 1, 2, 2, 3, 3, ...]
        paired_predicted_feats = (predicted_feats[0::2] + predicted_feats[1::2]) / 2
        paired_predicted_feats = torch.cat([paired_predicted_feats, paired_predicted_feats], dim=0)
        
        return paired_predicted_feats
"""