import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MSELoss, Linear, LayerNorm, BatchNorm1d, Dropout, ReLU, Sigmoid
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group
from Device import device


class GraphConv_edge_attn(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.edge_lin = Linear(edge_dim, out_channels)
        self.attn = Linear(2 * out_channels, 1)
        self.sigmoid = Sigmoid()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.edge_lin.reset_parameters()
        self.attn.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        edge_attr = self.edge_lin(edge_attr)
        z = torch.cat([x_i, edge_attr], dim=-1)
        alpha = self.sigmoid(self.attn(z))
        
        return alpha * (x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out


class GraphConv_edge_cat(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.edge_lin = Linear(edge_dim, out_channels)
        self.interaction = Linear(2 * out_channels, out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        self.edge_lin.reset_parameters()
        self.interaction.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        edge_attr = self.edge_lin(edge_attr)
        interaction = torch.cat([x_j, edge_attr], dim=-1)
        interaction = self.interaction(interaction)
        return interaction
    
    def update(self, aggr_out):
        return aggr_out


class GraphConv_edge(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_dim, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.edge_lin = Linear(edge_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):  #make it same as the DGL package algorithm
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        edge_attr = self.edge_lin(edge_attr)
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out


class GCNLayer_edge(torch.nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim=1,
                 bias=True, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer_edge, self).__init__()

        self.activation = activation
        
        self.graph_conv = GraphConv_edge(in_channels=in_feats, out_channels=out_feats, edge_dim=edge_dim, aggr="add", bias=bias)

        self.residual = residual
        if residual:
            self.res_connection = Linear(in_feats, out_feats, bias = bias)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = BatchNorm1d(out_feats)

        self.dropout = Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, feats, edge_index, edge_attr):
        new_feats = self.activation(self.graph_conv(feats, edge_index, edge_attr))            #lacking of the g
        new_feats = self.dropout(new_feats)

        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats

        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
    
    def forward(self, x, edge_index): 
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x)
        return out


class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, 
                 bias=True, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        
        self.graph_conv = GraphConv(in_channels=in_feats, out_channels=out_feats, aggr="add", bias=bias)

        self.residual = residual
        if residual:
            self.res_connection = Linear(in_feats, out_feats, bias = bias)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = BatchNorm1d(out_feats)

        self.dropout = Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, feats, edge_index):
        new_feats = self.activation(self.graph_conv(feats, edge_index))            #lacking of the g
        new_feats = self.dropout(new_feats)

        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats

        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats
    

class GCN(torch.nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, activation=None):
        super(GCN, self).__init__()

        n_layers = len(hidden_feats)

        activation = [activation for _ in range(n_layers)]

        bias = [True for _ in range(n_layers)]
        residual = [True for _ in range(n_layers)]
        batchnorm = [True for _ in range(n_layers)]
        dropout = [0. for _ in range(n_layers)]
        
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], bias[i], activation[i], 
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, feats, edge_index):
        for gnn in self.gnn_layers:
            feats = gnn(feats, edge_index)
        return feats


class Hetero_Transform(torch.nn.Module):
    def __init__(self, conv, aggr="sum"):
        super().__init__()
        self.conv = conv
        self.aggr = aggr

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self,x_dict, edge_index_dict):
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            
            src, rel, dst = edge_type
            str_edge_type = '__'.join(edge_type)

            out = self.conv((x_dict[src], x_dict[dst]), edge_index)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict


class Hetero_GraphConv(MessagePassing):

    def __init__(self, in_feats, out_feats, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.lin_rel = nn.Linear(in_feats, out_feats, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_rel.weight)
        torch.nn.init.zeros_(self.lin_rel.bias)

    def forward(self, x, edge_index):  #make it same as the DGL package algorithm
        x_src, x_dst = x
        x_src = self.lin_rel(x_src)  
        x_dst = self.lin_rel(x_dst)  
        x = (x_src, x_dst)
        out = self.propagate(edge_index, x=x)
        return out

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x[0], reduce=self.aggr)
    
    
class Hetero_GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(Hetero_GCNLayer, self).__init__()

        self.activation = activation

        self.graph_conv = Hetero_Transform(Hetero_GraphConv(in_feats, hidden_feats), aggr="sum")
        
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, hidden_feats, bias = True)
            
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer_1 = nn.BatchNorm1d(hidden_feats)
            self.bn_layer_2 = nn.BatchNorm1d(hidden_feats)
            self.bn_dict = {"atoms":self.bn_layer_1, "cohps":self.bn_layer_2}

        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        self.graph_conv.reset_parameters()

        if self.residual:
            self.res_connection.reset_parameters()

        if self.bn:
            self.bn_layer_1.reset_parameters()
            self.bn_layer_2.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        new_feats = self.graph_conv(x_dict, edge_index_dict)

        if self.residual:
            res_feats = {k: self.activation(self.res_connection(v)) for k,v in x_dict.items()}
            new_feats = {k: v + res_feats[k] for k,v in new_feats.items()}

        new_feats = {k: self.dropout(v) for k,v in new_feats.items()}

        if self.bn:
            new_feats = {k: self.bn_dict[k](v) for k,v in new_feats.items()}

        return new_feats

    
class Hetero_GCN(nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, activation=None):
        super(Hetero_GCN, self).__init__()

        n_layers = len(hidden_feats)
        activation = [activation for _ in range(n_layers)]

        residual = [True for _ in range(n_layers)]
        batchnorm = [True for _ in range(n_layers)]
        dropout = [0. for _ in range(n_layers)]
        
        self.gnn_Hetero_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_Hetero_layers.append(Hetero_GCNLayer(in_feats, hidden_feats[i], 
                                                          activation[i], residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        for gnn in self.gnn_Hetero_layers:
            x_dict = gnn(x_dict, edge_index_dict)
        return x_dict



class MLPPredictor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks))

    def forward(self, feats):
        return self.predict(feats)


class MAB(torch.nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, Conv = None, layer_norm = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(self, Q, K, graph = None):
        Q = self.fc_q(Q)

        if graph is not None:
            if len(graph)==4:
                x, edge_index, batch,edge_attr = graph
            else:
                x, edge_index, batch = graph
            if len(graph)==4:
                K, V = self.layer_k(x, edge_index,edge_attr), self.layer_v(x, edge_index,edge_attr)
            else:

                K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads # 40
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)  # calculate the attension between graph and nodes

        A = torch.softmax(attention_score, 1) # Find the weights of each node embedding
            
        out = Q_ + A.bmm(V_)  # A*V means that weighted sum the node embedding to graph embedding
        out = torch.cat(out.split(Q.size(0), 0), 2) # erase the heads

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class SAB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, Conv = None, layer_norm = False):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads,Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x, graph = None):
        return self.mab(x, x, graph)


class PMA(torch.nn.Module):
    def __init__(self, channels, num_heads, num_seeds, Conv = None, layer_norm = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(self, x, graph=None):    
        seed = self.S.repeat(x.size(0), 1, 1)
        return self.mab(seed, x, graph) #x is actually the batch_x, and graph contains dense_x.


class GMT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Conv=None, num_nodes=300,
        pooling_ratio=0.25, pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads=4, layer_norm=False):
        
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv # Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        
        for i, pool_type in enumerate(pool_sequences):
            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(PMA(hidden_channels, num_heads, num_out_nodes, Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'GMPool_I':
                print("Applied GMPool_I")
                self.pools.append(PMA(hidden_channels, num_heads, num_out_nodes, Conv=None, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'SelfAtt':
                print("Applied SelfAtt")
                self.pools.append(SAB(hidden_channels, hidden_channels, num_heads, Conv=None, layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, x, batch, edge_attr=None, edge_index=None):
        x = self.lin1(x)
        batch_x, _ = to_dense_batch(x, batch)

        for (name, pool) in zip(self.pool_sequences, self.pools):
            # print(type(edge_index))
            if edge_attr is not None:
                graph = (x, edge_index, batch,edge_attr) if name == 'GMPool_G' else None
            else:
                graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            batch_x = pool(batch_x, graph)  #batch_x = pool(batch_x, graph, mask)
        return batch_x. squeeze(1)


###                                                         ###
### Hetero-Graph Specified Multihead Tranformer Pooling Net ###
###                                                         ###
class Hetero_MAB(torch.nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, Conv = None, layer_norm = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm
    
        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)
            
        self.fc_q1 = Linear(dim_Q, dim_V)
        self.fc_q2 = Linear(dim_Q, dim_V)
        self.fc_k1 = Linear(dim_K, dim_V)
        self.fc_k2 = Linear(dim_K, dim_V)
        self.fc_v1 = Linear(dim_K, dim_V)
        self.fc_v2 = Linear(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters() 
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        
        self.fc_q1.reset_parameters()
        self.fc_q2.reset_parameters()
        self.fc_k1.reset_parameters()
        self.fc_k2.reset_parameters()
        self.fc_v1.reset_parameters()
        self.fc_v2.reset_parameters()
        
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(self, Q, K, graph = None):
        if isinstance(Q,dict):
            Q_dict = {k:fc(Q[k]) for fc,k in zip([self.fc_q1, self.fc_q2], list(Q.keys()))}
        else:
            Q = self.fc_q(Q)

        if isinstance(K,dict):
            K_dict = K
        
        if graph is not None:
            x_dict, edge_index_dict, batch_dict = graph
            K_dict, V_dict = self.layer_k(x_dict, edge_index_dict), self.layer_v(x_dict, edge_index_dict)
            K_dict = {k:to_dense_batch(v, batch_dict[k])[0] for k,v in K_dict.items()}
            V_dict = {k:to_dense_batch(v, batch_dict[k])[0] for k,v in V_dict.items()}
        else:
            if isinstance(K,dict):
                K_dict = {k:fc(K_dict[k]) for fc,k in zip([self.fc_k1, self.fc_k2], list(K_dict.keys()))}
                V_dict = {k:fc(K_dict[k]) for fc,k in zip([self.fc_v1, self.fc_v2], list(K_dict.keys()))}
            else:
                K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads # 40

        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        if isinstance(K,dict):
            K_dict_ = {k:torch.cat(v.split(dim_split,2), dim=0) for k,v in K_dict.items()}
            V_dict_ = {k:torch.cat(v.split(dim_split,2), dim=0) for k,v in V_dict.items()}
            K_ = torch.cat(list(K_dict_.values()),dim=1)
            V_ = torch.cat(list(V_dict_.values()),dim=1)
        else:
            K_ = torch.cat(K.split(dim_split,2), dim=0)
            V_ = torch.cat(V.split(dim_split,2), dim=0)


        attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)  # calculate the attension between graph and nodes

        A = torch.softmax(attention_score, 1) # Find the weights of each node embedding

        out = Q_ + A.bmm(V_)  # A*V means that weighted sum the node embedding to graph embedding

        out = torch.cat(out.split(Q.size(0), 0), 2) # erase the heads

        if self.layer_norm:
            out = self.ln0(out)
            
        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class Hetero_SAB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, Conv = None, layer_norm = False):
        super().__init__()
        self.mab = Hetero_MAB(in_channels, in_channels, out_channels, num_heads,Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x, graph = None):
        return self.mab(x, x, graph)


class Hetero_PMA(torch.nn.Module):
    def __init__(self, channels, num_heads, num_seeds, Conv = None, layer_norm = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = Hetero_MAB(channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(self, x, graph): 
        if isinstance(x,dict):
            seed = self.S.repeat(x["atoms"].size(0), 1, 1)
        else:
            seed = self.S.repeat(x.size(0), 1, 1)
        return self.mab(seed, x, graph)


class Hetero_GMT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Conv=None, num_nodes=300,
        pooling_ratio=0.25, pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads=4, layer_norm=False):
        
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv # Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(in_channels, hidden_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        
        for i, pool_type in enumerate(pool_sequences):
            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(Hetero_PMA(hidden_channels, num_heads, num_out_nodes, Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'GMPool_I':
                print("Applied GMPool_I")
                self.pools.append(Hetero_PMA(hidden_channels, num_heads, num_out_nodes, Conv=None, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'SelfAtt':
                print("Applied SelfAtt")
                self.pools.append(Hetero_SAB(hidden_channels, hidden_channels, num_heads, Conv=None, layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, x_dict, batch_dict, edge_index_dict):
        x_dict = {k:lin(v) for lin,(k,v) in zip([self.lin1, self.lin2], x_dict.items())}
        batch_x_dict = {k:to_dense_batch(v, batch_dict[k])[0] for k,v in x_dict.items()}

        for (name, pool) in zip(self.pool_sequences, self.pools):
            graph_dict = (x_dict, edge_index_dict, batch_dict) if name == 'GMPool_G' else None
            batch_x = pool(batch_x_dict, graph_dict)
            batch_x_dict = batch_x

        return batch_x.squeeze(1)
