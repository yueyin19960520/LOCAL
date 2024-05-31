from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from pymatgen.io.vasp import Poscar
import os
import torch
import numpy as np
import pickle
import re
from sklearn.preprocessing import OneHotEncoder
from functions import *
from itertools import product, combinations, combinations_with_replacement
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

def draw(Data):
    G = to_networkx(Data)
    # 添加边
    src, dst = Data.edge_index
    edges = [(int(src[i]), int(dst[i])) for i in range(len(src))]
    # nx.draw(G)
    # 添加节点标签
    node_labels = {i: str(i) for i in range(58)}
    node_colors = ['red' if i >= Data.num_nodes - 2 else 'white' for i in range(Data.num_nodes)]  # 最后两个节点涂红色

    # 确定与最后两个节点相连的节点
    connected_nodes = set()
    for edge in edges:
        if edge[0] >= Data.num_nodes - 2 and edge[1] < Data.num_nodes - 2:
            connected_nodes.add(edge[1])
        if edge[1] >= Data.num_nodes - 2 and edge[0] < Data.num_nodes - 2:
            connected_nodes.add(edge[0])

    for i in connected_nodes:
        node_colors[i]='yellow'

    pos = nx.spring_layout(G)  # 选择一种布局方式
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=10)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
    plt.show()
class POS2E_Dataset_all(Dataset):
    '''这个class是在POS2CLS的基础上改的，
    1.for num_C in range(0, 1+len(ALL_N_idx)): # consider all combinations
    2.data.MN_edge_index = MN_edge_index因为要获得ICOHP必须有MN_edge_index
    3.要使用Fc
    4.加上data.cohp_num
    5.分为有没有name_cohpoutput_dict两种模式，没有时是给POS2COHP_net用的，得到name_cohpoutput_dict
    后，再生成一个使用了name_cohpoutput_dict的数据集，从而给POS2E用生成所有嵌入构建KDTree
    '''
    def __init__(self, root, Element_List, Metals, name_cohpoutput_dict=None,
                 transform=None, pre_transform=None, mode='normal',plotplot=False):
        self.Element_List = Element_List
        self.Metals = Metals
        
        '''运行super之前要有self.name_cohpoutput_dict'''
        self.name_cohpoutput_dict = name_cohpoutput_dict
        self.mode = mode
        self.plotplot = plotplot
        super(POS2E_Dataset_all, self).__init__(root, transform, pre_transform) 
        '''这行必须有，因为self.data的定义在process中，如果已经有数据集了，就不会运行，从而使得self.data没有定义
        如果没有数据集，我猜测这行应该会返回空，也不会报错'''
        self.data = torch.load(self.processed_paths[0])
        
        

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if self.name_cohpoutput_dict:
            return["POS2E_all_with_cohp_indicated_edge.pt"]
        return ["POS2E_all.pt"]
    
    def download(self):
        pass
    
    def process(self):
        print('POS2E_Dataset_all not exist, so begin to process')
        cohp_num_dict = {'QV1':24,'QV2':24,'QV3':24,'QV4':28,'QV5':32,'QV6':28,}
        data_list = []
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))
        
        unsym = list(map(lambda x:x[0]+"_"+x[1],list(product(self.Metals, self.Metals))))
        sym = list(map(lambda x:x[0]+"_"+x[1],list(combinations_with_replacement(self.Metals, 2)))) 

        sample_space_by_names = {}


        for qv in list(filter(lambda x:".vasp" in x, os.listdir("sample_space"))):   # Only consider all N structure
            ALL_N_structure = Poscar.from_file(os.path.join("sample_space", qv)).structure
            connectivity = np.array(get_connectivity(ALL_N_structure))
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()
            qv = qv[:3]
            sample_space_by_names[qv] = {}
            # 金属邻近的CN索引
            ALL_N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ALL_N_structure[x].specie.name == "N" else None, range(len(ALL_N_structure))))))
            if self.mode == 'debug':
                num_C_max = 1
            else:
                num_C_max = 1+len(ALL_N_idx)
            for num_C in range(0, num_C_max): # consider all combinations
                sample_space_by_names[qv][num_C] = []
                candi_list = list(combinations(ALL_N_idx, num_C))
                # print('candi_list',candi_list)
                # 对于碳数为num_C的所有CN构型
                for candi in candi_list:
                    number_name = "".join(str(x) for x in [x-min(ALL_N_idx) for x in candi])
                    ele_pairs = unsym if "QV4" in qv else sym
                    # 对于所有金属对
                    for ele_pair in ele_pairs:
                        # 名字
                        specific_name = qv + "_" + number_name + "_" + ele_pair
                        
                        # print('ele_pair',ele_pair)
                        sample_space_by_names[qv][num_C].append(specific_name) 
                        
                        # 获得C的索引，1 2 3->51 52 53
                        C_idx = [int(c) for c in number_name]
                        changed_C_idx = np.array(ALL_N_idx)[C_idx]

                        # 获取全N poscar
                        # ori_poscar = Poscar.from_file("./sample_space/%s.vasp"%qv).structure
                        # connectivity = np.array(get_connectivity(ori_poscar))

                        #获取两个金属的索引
                        idx1, idx2 = len(ALL_N_structure)-2, len(ALL_N_structure)-1
                        # N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ALL_N_structure[x].specie.name == "N" else None, range(len(ALL_N_structure))))))
                        # first_N = min(ALL_N_idx)

                        #拷贝原始qv结构，N换C
                        new_structure = copy.deepcopy(ALL_N_structure)
                        for idx in changed_C_idx:
                            # idx = first_N+int(idx)
                            new_structure.replace(idx, "C")

                        # 含有金属的边
                        temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
                        # print('第一个temp',temp)# [None,...,None,array([56,53],array([56,54])...]

                        # 所有参与金属成键的索引列表 [50, 51, 52, 53, 54, 55, 56, 57]
                        temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
                        # print('第二个temp',temp)# [50, 51, 52, 53, 54, 55, 56, 57]

                        # 在temp中找C，从而找到Fc的索引
                        Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
                        # print(Fc_idx)#[50, 53]

                        fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + list(ele_pair.split('_')))
                        # print(fake_eles)
                        # ['C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'Fc' 'N' 'N' 'Fc'
                        # 'N' 'N' 'Pd' 'Sb']
                        eles = np.array([site.specie.name for site in ALL_N_structure][0:-2] + ele_pair.split("_"))
                        # print(eles)
                        # ['C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'N' 'N' 'N' 'N'
                        # 'N' 'N' 'Pd' 'Sb']
                        eles[changed_C_idx] = "C"
                        # print(eles)
                        # ['C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C'
                        # 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'N' 'N' 'C'
                        # 'N' 'N' 'Pd' 'Sb']
                        Fake_C = True
                        if Fake_C:
                            onehot = encoder.transform(fake_eles.reshape(-1,1))
                            # print(onehot.shape)
                        x = torch.tensor(onehot, dtype=torch.float)
                        y = torch.tensor(np.array([0.]*12)).unsqueeze(0)
                        
                        

                        if self.name_cohpoutput_dict==None:# 不知道COHP的预测值
                            cohp_res=None # 不知道ICOHP的真实值，但是这个参数是下面的函数必须的
                            MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)
                            # print('n',edge_index)
                            data = Data(x=x, edge_index=edge_index, cls="sample", name=specific_name)
                            # print('no cohp info')
                            if self.plotplot:
                                print(specific_name)
                                draw(data)
                        elif self.name_cohpoutput_dict:# 已经拿到了所有结构的COHP
                            cohp_res=None # 不知道ICOHP的真实值，但是这个参数是下面的函数必须的
                            MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)
                            cohp_output = self.name_cohpoutput_dict[specific_name]
                            candi_edge_index = MN_edge_index.T.numpy()
                            Binary_COHP=True#使用
                            if Binary_COHP:
                                temp_MN_index = np.array([x[0] for x in list(filter(lambda x:x[1][0] > x[1][1], list(zip(candi_edge_index, cohp_output))))])
                            edge_index_without_M = filter_pairs(edge_index, {56,57})
                            good_MN_index = torch.tensor(temp_MN_index).T
                            edge_index = torch.hstack((edge_index_without_M, good_MN_index)).to(torch.int64)
                            # print('y',edge_index)
                            # print(specific_name)
                            data = Data(x=x, edge_index=edge_index, cls="sample", name=specific_name)
                            # print('with cohp info')
                            if self.plotplot:
                                print(specific_name)
                                draw(data)

                        data.MN_edge_index = MN_edge_index 
                        data.cohp_num = cohp_num_dict[qv]
                        data_list.append(data)
                   
        # total_num_samples = sum(list(map(lambda x:sum(list(map(lambda y:len(y), x.values()))), list(sample_space_by_names.values()))))
        # all_name_list = [i for j in list(map(lambda x:[i for j in list(x.values()) for i in j], list(sample_space_by_names.values()))) for i in j]
        self.data = data_list
        torch.save(data_list, self.processed_paths[0])

      
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
class Find_analogue():
    def __init__(self,Metals,Slabs,mode='1') -> None:
        self.Metals = Metals
        self.Slabs = Slabs
        self.mode = mode
        Fake_C = True
        if Fake_C:
            if 'Fc' not in self.Metals + self.Slabs:
                self.Element_List = self.Metals + self.Slabs + ['Fc']
    def get_all_cohp_prediction_value(self):
        '''加载POS2COHP模型'''
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pos2cohp_model_path = "./models/POS2COHP_Net_FcN_BC_Homo.pth"
        model = torch.load(pos2cohp_model_path).to(device)
        print('getting all cohp pred value...')
        print("POS2COHP model loaded from %s!"%(pos2cohp_model_path))
        all_pos2cls_dataset = POS2E_Dataset_all("./", self.Element_List, self.Metals,mode=self.mode)#.shuffle()
        all_loader = DataLoader(all_pos2cls_dataset, batch_size=48, shuffle=False, drop_last=False)
        all_names = [data.to("cpu").name for data in all_loader]
        def split_batch_data(batch, batch_pred):
            batch = batch.to("cpu")
            batch_pred = batch_pred.to("cpu")
            lengths = list(batch.cohp_num.numpy())
            slices = np.cumsum(lengths)[:-1]
            # print(batch_pred[0])
            # print(batch_pred)
            # print(np.array(batch_pred))
            pred_slice = np.split(np.array(batch_pred), slices)
            return pred_slice
        model.eval()
        with torch.no_grad():
            PRED = list(map(lambda data:split_batch_data(data, model(data.to(device))), all_loader))
        all_pos2cohp_output = [i for j in PRED for i in j] # 列表推导式，把二维列表展开成一维列表
        # print('len(all_pos2cohp_output)',len(all_pos2cohp_output))
        # print(len(PRED[0]))
        all_names = [i for j in all_names for i in j]
        print('POS2E dataset length: ',len(all_names))
        # print(all_names[:5])
        # print(PRED[:5])
        # print('len(all_names[0])',len(all_names[0]))
        # print(all_names)

        # print(len(all_pos2cohp_output))
        # print('len(all_pos2cohp_output[0])',len(all_pos2cohp_output[0]))
        # print('all_pos2cohp_output[0]',all_pos2cohp_output[0])
        # print(1152/48)
        name_pos2cohp_output_dict = {name: output for name, output in zip(all_names, all_pos2cohp_output)}
        # print(name_pos2cohp_output_dict['QV1__Sc_Sc'])
        # print(len(name_pos2cohp_output_dict['QV1__Sc_Sc']))

        import pickle
        name_pos2cohp_output_dict_path = './name_pos2cohp_output_dict'
        with open(name_pos2cohp_output_dict_path,'wb') as file:
            print('name_pos2cohp_output_dict written to %s!'%(name_pos2cohp_output_dict_path))
            pickle.dump(name_pos2cohp_output_dict,file)
        # print(108*48)
        # import pickle
        # with open('name_pos2cohp_output_dict','rb') as file:
        #     name_pos2cohp_output_dict = pickle.load(file)
        # # print(name_pos2cohp_output_dict)

    def get_all_embeddings_of_POS2E_net_and_build_KDTree(self):
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pos2e_model_path = "./models/POS2E_Net_FcN_BC_Homo.pth"
        model = torch.load(pos2e_model_path).to(device)
        print("POS2E model loaded from %s, then get_all_embeddings_of_POS2E_net_and_build_KDTree!"%(pos2e_model_path))
        
        # ori_pos2cls_dataset = POS2CLS_Dataset("./", self.structure_sets, self.Element_List, self.Metals).shuffle()
        # print("POS2CLS dataset used for training is ready.")
        with open('name_pos2cohp_output_dict','rb') as file:
            name_pos2cohp_output_dict = pickle.load(file)
        all_pos2e_dataset = POS2E_Dataset_all("./", self.Element_List, self.Metals,name_cohpoutput_dict=name_pos2cohp_output_dict,
                                              mode=self.mode)
        print("POS2E_ALL_ICOHP_edge dataset made!")
        # spl_pos2cls_dataset = all_pos2cls_dataset[:int(0.02*len(all_pos2cls_dataset))]

        # ori_loader = DataLoader(ori_pos2cls_dataset, batch_size=48, shuffle=True, drop_last=False)
        all_loader = DataLoader(all_pos2e_dataset, batch_size=48, shuffle=False, drop_last=False)
        # print('len(all_pos2e_dataset)',len(all_pos2e_dataset))
        # spl_loader = DataLoader(spl_pos2cls_dataset, batch_size=48, shuffle=True, drop_last=False)

        # ori_embeddings, ori_classes, ori_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
        #                                                 data.to("cpu").cls, 
        #                                                 data.to("cpu").name) for data in ori_loader])
        # print('ori_embeddings',len(ori_embeddings,ori_embeddings[0].shape)
        all_embeddings, all_classes, all_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                                        data.to("cpu").cls, 
                                                        data.to("cpu").name) for data in all_loader])

        # self.spl_embeddings, self.spl_classes, self.spl_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
        #                                                                data.to("cpu").cls, 
        #                                                                data.to("cpu").name) for data in spl_loader])
        # print(len(all_embeddings))
        # print(len(all_embeddings[0]))
        # print(len(all_names))
        # print(len(all_names[0]))
        # print(all_embeddings)
        # print(all_names)
        # print(tree_embeddings)
        tree_embeddings = np.vstack(all_embeddings)
        # print(len(tree_embeddings))
        # print(tree_embeddings.shape)
        tree_names = np.hstack(all_names)
        # print(tree_embeddings)
        # print(tree_names)
        # print('len(tree_names)',len(tree_names))
        # print(len(tree_embeddings))
        self.name_emb_dict = {name: embedding for name, embedding in zip(tree_names, tree_embeddings)}
        self.names = list(self.name_emb_dict.keys())
        # print('len(self.names)',len(self.names))
        self.embeddings = list(self.name_emb_dict.values())
        self.kd_tree = KDTree(self.embeddings)
        
        name_pos2e_embedding_path = "./name_pos2e_embedding_all.pkl"
        file_save = open(os.path.join("models", name_pos2e_embedding_path),'wb') 
        pickle.dump(self.name_emb_dict, file_save) 
        file_save.close()
        print("POS2E embeddings of all structure  saved to %s!"%(name_pos2e_embedding_path))

    def find_k_nearest_neighbors(self, target_point_key, k=4):
        try:
            target_embedding = self.name_emb_dict[target_point_key]
        except:
            target_embedding = self.name_emb_dict[self.rvs(target_point_key)]
        distances, indices = self.kd_tree.query(target_embedding, k=k+1) # except itself
        nearest_neighbors = dict([(self.names[i], self.embeddings[i]) for i in indices])
        # bad_neighbors_dict = dict([(target_point_key,[self.names[i] for i in indices])])
        return list(nearest_neighbors.keys())[1:] # except itself
        # return list(nearest_neighbors.keys()), bad_neighbors_dict

    def rvs(self, k): 
        return "_".join(k.split("_")[:2] + [k.split("_")[3], k.split("_")[2]])
