1.在dataset.py中的POS2COHP_Dataset做了修改，不再使用onehot，而是使用多种物理性质来表示原子。
2.POS2COHP.py，POS2E.py中的node_feats（特征数）要改。
3.POS2E_Dataset不需要改，
for g_index, graph in enumerate(self.src_dataset):
            if self.Hetero_Graph:
                nodes_features = graph.x_dict["atoms"]
                ori_edge_index = filter_pairs(graph.edge_index_dict['atoms', 'interacts', 'atoms'], {56,57})
            else:
                nodes_features = graph.x
因为它用的是src_dataset的x

4.此外还实验了把onehot和物理性质拼接起来

#################################################################################################

1.只用物理性质test mae是0.170eV，比使用onehot要差
2.onehot+物理性质test mae是0.151eV，与使用onehot基本相等
