import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MSELoss, Linear, LayerNorm, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group


class pyg_GraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        #self.lin_root = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        #self.lin_root.reset_parameters()
    
    
    def forward(self, x, edge_index):  #make it same as the DGL package algorithm
        x = self.lin_rel(x)
        out = self.propagate(edge_index, x=x)
        return out

    """
    def forward(self, x, edge_index):   # here is the original script from the pyg package, but more parameters!!!!

        x = (x, x)
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)
        return out
    """


class pyg_GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, 
                 bias=True, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(pyg_GCNLayer, self).__init__()

        self.activation = activation
        
        self.graph_conv = pyg_GraphConv(in_channels=in_feats, out_channels=out_feats, aggr="add", bias=bias)

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



class pyg_GCN(torch.nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, bias=None, activation=None, 
                 residual=None, batchnorm=None, dropout=None):
        super(pyg_GCN, self).__init__()

        n_layers = len(hidden_feats)

        if bias is None:
            bias = [True for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(pyg_GCNLayer(in_feats, hidden_feats[i], bias[i], activation[i], 
                                               residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, feats, edge_index):
        for gnn in self.gnn_layers:
            feats = gnn(feats, edge_index)
        return feats


class pyg_Hetero_Conv(torch.nn.Module):
    def __init__(self, convs, aggr="sum"):
        super().__init__()
        self.convs = convs
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(self,x_dict, edge_index_dict):
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            
            src, rel, dst = edge_type
            str_edge_type = '__'.join(edge_type)
            conv = self.convs

            out = conv((x_dict[src], x_dict[dst]), edge_index)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict


class pyg_Hetero_GraphConv(MessagePassing):

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
    
    
class pyg_Hetero_GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, activation, residual, batchnorm, dropout):
        super(pyg_Hetero_GCNLayer, self).__init__()

        self.activation = activation

        self.graph_conv = pyg_Hetero_Conv(pyg_Hetero_GraphConv(in_feats, hidden_feats), aggr="sum")
        
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
            #new_feats = {k: f(v) for (k,v),f in zip(new_feats.items(), [self.bn_layer_1, self.bn_layer_2])}
            new_feats = {k: self.bn_dict[k](v) for k,v in new_feats.items()}

        return new_feats

    
class pyg_Hetero_GCN(nn.Module):
    def __init__(self, in_feats=None, hidden_feats=None, activation=None, residual=None, batchnorm=None, dropout=None):
        super(pyg_Hetero_GCN, self).__init__()

        n_layers = len(hidden_feats)

        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        
        self.gnn_Hetero_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.gnn_Hetero_layers.append(pyg_Hetero_GCNLayer(in_feats, hidden_feats[i], 
                                                              activation[i], residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        for gnn in self.gnn_Hetero_layers:
            x_dict = gnn(x_dict, edge_index_dict)
        return x_dict



class pyg_MLPPredictor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(pyg_MLPPredictor, self).__init__()

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
            x, edge_index, batch = graph
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

        # Here can adding a BatchNorm Layer. 

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
        return self.mab(seed, x, graph)



class GraphMultisetTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Conv=None, num_nodes=300,
        pooling_ratio=0.25, pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads=4, layer_norm=False):
        
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = pyg_GCNLayer # Conv or GCNConv
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

    def forward(self, x, batch, edge_index=None):
        x = self.lin1(x)
        batch_x, _ = to_dense_batch(x, batch)

        for (name, pool) in zip(self.pool_sequences, self.pools):
            graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            batch_x = pool(batch_x, graph)  #batch_x = pool(batch_x, graph, mask)
        return batch_x. squeeze(1)
        # return self.lin2(batch_x. squeeze(1))


##########################################################################
############################### OUR MODELS ###############################
##########################################################################


class CONT2E_Net(torch.nn.Module):
    def __init__(self, dim=128, adj_conv=False, in_features=None, bias=True,
                 linear_dim_list=[[128,128]],
                 conv_dim_list=[[128,128],[128,128],[128,128]],
                 conv=SAGEConv, dropout=0., pool=GraphMultisetTransformer, pool_dropout=0.,
                 pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G", "SelfAtt", "GMPool_I"], pool_layer_norm=False):

        super(CONT2E_Net, self).__init__()

        self.num_conv_layers = len(conv_dim_list)
        self.num_linear_layers = len(linear_dim_list)
        self.adj_conv = adj_conv    
        self.dropout = dropout

        self.lin = Linear(in_features, linear_dim_list[0][0] if linear_dim_list else conv_dim_list[0][0],
                           bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(dim_in, dim_out, bias=bias) for [dim_in, dim_out] in linear_dim_list])
        self.conv_block = torch.nn.ModuleList([conv(dim_in, dim_out, bias=bias, dropout=self.dropout) for [dim_in, dim_out] in conv_dim_list])
        
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(dim, dim, bias=bias) for _ in range(self.num_conv_layers)])
            
        self.pool = pool(dim, dim, 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq, num_heads=pool_heads, layer_norm=pool_layer_norm)

        self.pool_dropout = torch.nn.Dropout(pool_dropout)     
        self.lin2 = Linear(in_features=conv_dim_list[-1][-1]*4, out_features=conv_dim_list[-1][-1]) 
        self.lin3 = Linear(in_features=conv_dim_list[-1][-1], out_features=1)  
        self.pool_mean = global_mean_pool
        self.pool_add = global_add_pool
        self.pool_max = global_max_pool                                                      
        
    def forward(self, data):    
        out = F.relu(self.lin(data.x))  # Input layer
        
        for layer in self.linear_block:
            out = F.relu(layer(out))

        if self.adj_conv:
            for (adj_layer, conv_layer) in zip(self.adj_block, self.conv_block):
                out = F.relu(adj_layer(out))
                out = F.relu(conv_layer(out, data.edge_index))
        else:
            for conv_layer in self.conv_block:
                out = F.relu(conv_layer(out, data.edge_index))

        # out_tsfm = self.pool(out, data.batch, data.edge_index)
        # print(out_tsfm.shape)
        out_mean = self.pool_mean(out, data.batch)
        # print(out_mean.shape)
        # out_add = self.pool_add(out, data.batch)
        # print(out_add.shape)
        # out_max = self.pool_max(out, data.batch)
        # print(out_max.shape)
        # out1 = torch.cat((out_tsfm, out_mean, out_max, out_add),dim=1)
        # print(out1.shape)
        # out1 = self.lin2(out1)
        # print(out1.shape)
        # out2 = out_tsfm+out_mean+out_max+out_add
        # print(out2.shape)
        # out = torch.mul(out1, out2)
        out = self.lin3(out_mean)
        # print(out.shape)
        
        out = self.pool_dropout(out)
        
        return out.view(-1)



class POS2COHP_Net(torch.nn.Module):
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
        return torch.squeeze(predicted_feats)


class POS2COHP_Net_Hetero(nn.Module):
    def __init__(self, atom_feats, bond_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(POS2COHP_Net_Hetero, self).__init__()
        
        self.uni_trans_atoms = nn.Linear(atom_feats, hidden_feats[0])
        self.uni_trans_bonds = nn.Linear(bond_feats, hidden_feats[0])

        self.gnn = pyg_Hetero_GCN(in_feats=hidden_feats[0], hidden_feats=hidden_feats, 
                                  activation=activation, residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = pyg_MLPPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = {k:f(v) for (k,v),f in zip(x_dict.items(), [self.uni_trans_atoms, self.uni_trans_bonds])}
        
        node_feats = self.gnn(x_dict, edge_index_dict)["cohps"]
        predicted_feats = self.predict(node_feats)
        return torch.squeeze(predicted_feats)



class POS2CLS_Net(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(POS2CLS_Net, self).__init__()

        self.gnn = pyg_GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, 
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]

        self.predict = pyg_MLPPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_feats = self.gnn(x, edge_index)
        
        graph_feats = global_mean_pool(node_feats, batch)
        
        if return_embedding:
            return graph_feats
        else:
            predicted_feats = self.predict(graph_feats)
            return predicted_feats