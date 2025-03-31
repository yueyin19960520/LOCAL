"""
其中主要是两个模型，代码里面分别是 POS2COHP_Net 和 CONT2E_Net

其中POS2COHP_Net的输出要作为GCN的边的特征跟原本的POS2COHP_Net的输入，一起进入 CONT2E

POS2COHP_Net 里的GCN都是无边特征的， POS2COHP很简单，就是更新节点特征，然后最后用首尾节点特征来预测边的数值，回归问题

CONT2E 里的GCN都是带有边特征的

CONT2E 稍微复杂，跟边一起更新节点特征，然后一个GMT transformer pooling层映射到一个scaler，也是回归问题

其中GMT需要详细画出来QKV的细节，还有pooling
"""


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



########################################
###########GMT GMT GMT##################
########################################
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

