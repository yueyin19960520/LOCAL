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

class pyg_GraphConv_with_edge_attn(MessagePassing):
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
        # 对边特征进行线性变换
        edge_attr = self.edge_lin(edge_attr)
        
        # 计算注意力系数
        z = torch.cat([x_i, edge_attr], dim=-1)
        alpha = self.sigmoid(self.attn(z))
        
        return alpha * (x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out
    
class pyg_GraphConv_with_edge_cat(MessagePassing):
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
    
class pyg_GraphConv_with_edge(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_dim, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        #self.lin_root = Linear(in_channels, out_channels, bias=bias)
        self.edge_lin = Linear(edge_dim, out_channels)  # 对边特征的线性变换

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        #self.lin_root.reset_parameters()
    
    
    def forward(self, x, edge_index, edge_attr):  #make it same as the DGL package algorithm
        x = self.lin_rel(x)
        # print(type(edge_index))
        # print(edge_index)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        # 对边特征进行线性变换，并加到邻居节点特征上
        edge_attr = self.edge_lin(edge_attr)
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out

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


class pyg_GCNLayer_with_edge(torch.nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim=1,
                 bias=True, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(pyg_GCNLayer_with_edge, self).__init__()

        self.activation = activation
        
        self.graph_conv = pyg_GraphConv_with_edge(in_channels=in_feats, out_channels=out_feats, edge_dim=edge_dim, aggr="add", bias=bias)

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
        # print(type(edge_index))
        new_feats = self.activation(self.graph_conv(feats, edge_index, edge_attr))            #lacking of the g
        new_feats = self.dropout(new_feats)

        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats

        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats





class pyg_GraphConv_without_edge_attr(MessagePassing):

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


class pyg_GCNLayer_without_edge_attr(torch.nn.Module):
    def __init__(self, in_feats, out_feats, 
                 bias=True, activation=F.relu, residual=True, batchnorm=True, dropout=0.):
        super(pyg_GCNLayer_without_edge_attr, self).__init__()

        self.activation = activation
        
        self.graph_conv = pyg_GraphConv_without_edge_attr(in_channels=in_feats, out_channels=out_feats, aggr="add", bias=bias)

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
            self.gnn_layers.append(pyg_GCNLayer_without_edge_attr(in_feats, hidden_feats[i], bias[i], activation[i], 
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
            if len(graph)==4:
                x, edge_index, batch,edge_attr = graph
            else:
                x, edge_index, batch = graph
            if len(graph)==4:
                K, V = self.layer_k(x, edge_index,edge_attr), self.layer_v(x, edge_index,edge_attr)
            else:
                # print(self.layer_k)
                # print(self.layer_v)
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
        # print(graph)
        return self.mab(seed, x, graph)



class GraphMultisetTransformer(torch.nn.Module):
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

    def forward(self, x, batch,edge_attr=None, edge_index=None):
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
        # return self.lin2(batch_x. squeeze(1))


##########################################################################
############################### OUR MODELS ###############################
##########################################################################


class CONT2E_Net(torch.nn.Module):
    def __init__(self, dim=128, adj_conv=False, in_features=None, bias=True,
                 linear_dim_list=[[128,128]],
                 conv_dim_list=[[128,128],[128,128],[128,128]],
                 conv=SAGEConv, dropout=0., pool=GraphMultisetTransformer, pool_dropout=0.,
                 pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G", "SelfAtt", "GMPool_I"],
                 pool_layer_norm=False, pool_type='all',
                 edge_dim=None, noise=False, noise_type=None, noise_mae=None, real_cohp=None):

        super(CONT2E_Net, self).__init__()

        self.num_conv_layers = len(conv_dim_list)
        self.num_linear_layers = len(linear_dim_list)
        self.adj_conv = adj_conv    
        self.dropout = dropout
        self.pool_type = pool_type

        self.lin = Linear(in_features, linear_dim_list[0][0] if linear_dim_list else conv_dim_list[0][0], bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(dim_in, dim_out, bias=bias) for [dim_in, dim_out] in linear_dim_list])
        if edge_dim:
            self.conv_block = torch.nn.ModuleList([conv(dim_in, dim_out, edge_dim = edge_dim, bias=bias, dropout=self.dropout) for [dim_in, dim_out] in conv_dim_list])
        else:
            self.conv_block = torch.nn.ModuleList([conv(dim_in, dim_out, bias=bias, dropout=self.dropout) for [dim_in, dim_out] in conv_dim_list])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(dim, dim, bias=bias) for _ in range(self.num_conv_layers)])
            
        self.pool = pool(dim, dim, 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq, num_heads=pool_heads, layer_norm=pool_layer_norm,
                         Conv=conv)

        self.pool_dropout = torch.nn.Dropout(pool_dropout)     
        self.lin2 = Linear(in_features=conv_dim_list[-1][-1]*4, out_features=conv_dim_list[-1][-1]) 
        self.lin3 = Linear(in_features=conv_dim_list[-1][-1], out_features=1)  
        self.pool_mean = global_mean_pool
        self.pool_add = global_add_pool
        self.pool_max = global_max_pool
        self.edge_dim = edge_dim
        self.noise = noise
        self.noise_type = noise_type
        self.noise_mae = noise_mae
        self.real_cohp = real_cohp
        print(f'CONT2E_Net init!!!conv:{conv}, noise:{noise},noise_type:{noise_type},noise_mae:{noise_mae}')                                                 
        
    def forward(self, data, return_embedding=False, edge_attr_serial = None, train_with_real=False, is_trainning=False): 
        if train_with_real: # train with real COHP , valid with pred!!!
            if is_trainning:
                edge_attr = data.edge_attr_real
            else:
                edge_attr = data.edge_attr_pred
        else:
            if edge_attr_serial is not None:
                edge_attr = edge_attr_serial
            else:
                if hasattr(data,'edge_attr_real'):
                    if self.real_cohp:
                        edge_attr = data.edge_attr_real
                    else:
                        edge_attr = data.edge_attr_pred
                    if self.noise:
                        if self.noise_type == 'mean':
                            # 给edge_attr添加均匀分布的随机噪声
                            noise = (torch.rand_like(edge_attr) - 0.5) * self.noise_mae*4  # 生成均匀分布噪声，范围在[-0.2, 0.2]
                            edge_attr = edge_attr + noise
                        if self.noise_type == 'gaussian':
                            # 计算目标标准差，使得绝对平均误差为 0.2
                            desired_mae = self.noise_mae
                            sigma = desired_mae / torch.sqrt(torch.tensor(2 / torch.pi)).item()
                            # 给 edge_attr 添加均值为 0，标准差为 sigma 的正态分布噪声
                            noise = torch.randn_like(edge_attr) * sigma
                            edge_attr = edge_attr + noise
                    else:
                        edge_attr = edge_attr
                else:
                    edge_attr = data.edge_attr
        out = F.relu(self.lin(data.x))  # Input layer
        
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
            out_tsfm = self.pool(x=out, batch=data.batch,edge_index=data.edge_index,edge_attr=edge_attr)
        else:
            out_tsfm = self.pool(x=out, batch=data.batch,edge_index=data.edge_index)
        # print(out_tsfm.shape)
        out_mean = self.pool_mean(out, data.batch)
        # print(out_mean.shape)
        out_add = self.pool_add(out, data.batch)
        # print(out_add.shape)
        out_max = self.pool_max(out, data.batch)
        # print(out_max.shape)
        if self.pool_type=='all':
            out1 = torch.cat((out_tsfm, out_mean, out_max, out_add),dim=1)
            # print(out1.shape)
            out1 = self.lin2(out1)
            if return_embedding:
                # print('out1.shape',out1.shape)
                return out1
            # print(out1.shape)
            # out2 = out_tsfm+out_mean+out_max+out_add
            # print(out2.shape)
            # out = torch.mul(out1, out2)
            out = self.lin3(out1)
            # print(out.shape)
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

class POS2COHP_Net(torch.nn.Module): # _cat_change_to_add
    def __init__(self, in_feats, hidden_feats=None, 
                 activation=None, residual=None, batchnorm=None, dropout=None, 
                 predictor_hidden_feats=None, n_tasks=None, predictor_dropout=None):
        super(POS2COHP_Net, self).__init__()

        self.gnn = pyg_GCN(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, 
                          residual=residual, batchnorm=batchnorm, dropout=dropout)
        
        gnn_out_feats = hidden_feats[-1]
        # print(n_tasks)
        self.predict = pyg_MLPPredictor(gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, data):
        x, edge_index, MN_edge_index = data.x, data.edge_index, data.MN_edge_index
        node_feats = self.gnn(x, edge_index)
        
        # edge_embedding = torch.cat([node_feats[MN_edge_index[0]], node_feats[MN_edge_index[1]]], dim=1)
        edge_embedding = node_feats[MN_edge_index[0]]+ node_feats[MN_edge_index[1]]

        predicted_feats = self.predict(edge_embedding)
        # print(predicted_feats.shape)

        return torch.squeeze(predicted_feats)

class POS2COHP_Net_ori(torch.nn.Module):
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
###################################### Force inverse edge equals 
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
