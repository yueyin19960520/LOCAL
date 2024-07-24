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
from mendeleev import element
from Device import device

# class CONT2E_Dataset(Dataset):
#     def __init__(self, root, dataset_name='CONT2E_without_COHP', Element_List=None, setting=None, transform=None, pre_transform=None):
#         self.dataset_name = dataset_name
#         self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
#         ###############################################################################################
#         self.Fake_Carbon = setting["Fake_Carbon"]
#         # self.Binary_COHP = setting["Binary_COHP"]
#         self.Hetero_Graph = setting["Hetero_Graph"]
#         # self.threshold = setting["threshold"] # do not need COHP threshold

#         super(CONT2E_Dataset, self).__init__(root, transform, pre_transform) 
#         self.data = torch.load(os.path.join(self.processed_dir, "%s.pt"%self.dataset_name))
        
#     @property
#     def raw_file_names(self):
#         return ["%s_raw_data_dict.pkl"%self.dataset_name]

#     @property
#     def processed_file_names(self):
#         return ["%s.pt"%self.dataset_name]
    
#     def download(self):
#         pass
    
#     def process(self):
#         with open(self.raw_paths[0], 'rb') as f:
#             raw_data_dict = pickle.load(f)
            
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         encoder.fit(np.array(self.Element_List).reshape(-1, 1))

#         data_list = []
#         for name, raw_data in raw_data_dict.items():
#             if name.split('_')[0]!='QV4':
#                 continue
#             structure = raw_data[0]
#             distance_matrix = raw_data[1]
#             stability_energy = raw_data[2]

#             '''use get_connectivity as all other datasets instead of get_coordinated_environment'''
#             # coordinated_idx_matrix = get_coordinated_environment(structure, distance_matrix)
#             # reciprocal = check_validaiton(coordinated_idx_matrix)

#             # if reciprocal:



#             ################ introduce Fc ###########################################################
#             folder = name
#             ori_poscar = structure
#             connectivity = np.array(get_connectivity(ori_poscar))
#             idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
#             N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
#             new_structure = copy.deepcopy(ori_poscar)
#             temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
#             temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
#             Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
#             fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
#             eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])





#             # nodes_ele = list(map(lambda x:x.specie.name, structure.sites))
#             nodes_ele = fake_eles
#             nodes_features = encoder.transform(X = np.array(nodes_ele).reshape(-1,1))
#             nodes_features = torch.tensor(nodes_features, dtype=torch.float32)

#             # edges = [ij for l in list(map(lambda y:list(map(lambda x:(y,x), 
#             #                                                          coordinated_idx_matrix[y])), 
#             #                                        np.arange(len(coordinated_idx_matrix)))) for ij in l]

#             # np_edges = np.array(edges)
#             # edges_src, edges_dst = np_edges[:,0], np_edges[:,1]
#             # edge_index = torch.tensor(np.vstack((edges_src, edges_dst)), dtype=torch.long)

#             dE = torch.tensor(stability_energy, dtype=torch.float32)

#             edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()

#             data = Data(x=nodes_features, edge_index=edge_index, y=dE)
#             data_list.append(data)
#             # else:
#             #     None

#         torch.save(data_list, os.path.join(self.root,"processed", "%s.pt"%self.dataset_name))

      
#     def len(self):
#         return len(self.data)

#     def get(self, idx):
#         return self.data[idx]


class POS2COHP_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None,part_MN=None,icohp_list_keys=None):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.part_MN = part_MN
        self.icohp_list_keys = icohp_list_keys # 在main控制训练集测试集
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None, )
        self.root = root   
        self.data = torch.load(self.processed_paths[0])     
        
    @property
    def raw_file_names(self) :
        return ["icohp_structures_all.pkl"]

    @property
    def processed_file_names(self) :
        return ["POS2COHP_%s.pt"%self.suffix]
    
    def download(self):
        None

    def process(self): 
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        with open(self.raw_paths[0], "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)

        # for folder, cohp_res in icohp_list_dict.items(): # 遍历icohp_list_dict
        for folder in self.icohp_list_keys:
            cohp_res = icohp_list_dict[folder]
            qv,c_index,ele1,ele2 = folder.split("_")
            ori_poscar = Poscar.from_file("./sample_space/%s.vasp"%qv).structure
            connectivity = np.array(get_connectivity(ori_poscar))
            idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
            N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
            first_N = min(N_idx)
            new_structure = copy.deepcopy(ori_poscar)
            for idx in c_index:
                idx = first_N+int(idx)
                new_structure.replace(idx, "C")

            temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
            temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
            Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
            fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
            eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])
            
            if self.Fake_Carbon:
                onehot  = encoder.transform(fake_eles.reshape(-1,1))
            else:
                onehot = encoder.transform(eles.reshape(-1,1))
            
            x = torch.tensor(onehot, dtype=torch.float)
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()

            MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)



            if self.part_MN:
                ###################################### only pred valid edge COHP ####################################
                # 将张量转换为列表以便进行比较
                tensor1_list = edge_index.T.tolist()
                tensor2_list = MN_edge_index.T.tolist()

                # 找到两个张量的交集
                MN_edge_index = [item for item in tensor1_list if item in tensor2_list]
                # 找到交集对应的特征
                # print(MN_edge_index)
                MN_icohp = [MN_icohp[tensor2_list.index(item)] for item in MN_edge_index]
                MN_edge_index = torch.tensor(MN_edge_index).T
                # print(MN_edge_index.shape)
                # print(MN_edge_index)
                # print(len(MN_icohp))
                # print(MN_icohp)
                # print(cohp_res)
                ####################################################################################################





            binary_icohp = torch.from_numpy(np.array(list(map(lambda cohp:[1, 0] if cohp <= self.threshold else [0, 1], MN_icohp)),dtype="float32"))

            fake_node_index = torch.Tensor(np.arange(MN_edge_index.shape[1])).to(torch.int64).unsqueeze(0)
            MN_fake_node_index = torch.vstack((MN_edge_index[0],fake_node_index,MN_edge_index[1]))
            fake_x = torch.vstack(list(map(lambda i:x[MN_edge_index[:,i][0].item()] + x[MN_edge_index[:,i][1].item()], np.arange(MN_edge_index.shape[1]))))

            if self.Hetero_Graph:
                data = HeteroData()
                data['atoms'].x = x
                data['cohps'].x = fake_x

                data['atoms', 'interacts', 'atoms'].edge_index = edge_index
                data['cohps', 'interacts', 'cohps'].edge_index = torch.vstack([fake_node_index, fake_node_index])
                data['atoms', 'interacts', 'cohps'].edge_index = torch.hstack((torch.vstack((MN_fake_node_index[0,:],MN_fake_node_index[1,:])),
                                                                                   torch.vstack((MN_fake_node_index[-1,:],MN_fake_node_index[1,:]))))
                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.MN_edge_index = MN_edge_index
                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data_list.append(data)
            else:
                data = Data(x=x, edge_index=edge_index)
                data.MN_edge_index = MN_edge_index

                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data_list.append(data)                    

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class POS2E_Dataset(Dataset):
    def __init__(self, root, src_dataset=None, predicted_value=None, raw_data_dict=None, setting=None, suffix=None, Data_Augmentation=False,
                 maximum_num_atoms=100):
        self.src_dataset = src_dataset
        self.pred = predicted_value
        self.raw_data_dict = raw_data_dict
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.Data_Augmentation = Data_Augmentation
        self.maximum_num_atoms = maximum_num_atoms
        super(POS2E_Dataset, self).__init__(root, transform=None, pre_transform=None) 
        self.data = torch.load(self.processed_paths[0])  
        

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["POS2E_%s.pt"%self.suffix]
    
    def download(self):
        pass
    
    def process(self):
        data_list = []

        # For Data Augmentation!
        qvs = [f.replace(".vasp", '') for f in os.listdir("sample_space/") if '.vasp' in f]
        ori_poss = [Poscar.from_file(os.path.join("sample_space/", qv+".vasp")).structure for qv in qvs]
        augs = [enumerate_padding_structure(ori_pos, maximum_num_atoms=self.maximum_num_atoms) for ori_pos in ori_poss]
        num_atomss = [[len(s) for s in ss] for ss in augs]

        aug_connectivities = [[get_connectivity(s) for s in sl] for sl in augs]
        aug_edge_index = [[filter_pairs(torch.tensor(aug_conn, dtype=torch.long).t().contiguous(), {56,57}) for aug_conn in aug_conns] for aug_conns in aug_connectivities]
        augs_info_dict = {qvs[i]:{"index":aug_edge_index[i], "nums":num_atomss[i]} for i in range(len(qvs))}

        for g_index, graph in enumerate(self.src_dataset):
            if self.Hetero_Graph:
                nodes_features = graph.x_dict["atoms"]
                ori_edge_index = filter_pairs(graph.edge_index_dict['atoms', 'interacts', 'atoms'], {56,57})
            else:
                nodes_features = graph.x
                ori_edge_index = filter_pairs(graph.edge_index, {56,57})

            key = graph.slab + "_" + graph.metal_pair
            if key in self.raw_data_dict:
                energy = self.raw_data_dict[key]

                candi_edge_index = graph.MN_edge_index.T.numpy()
                cohp_pred = self.pred[g_index]

                if self.Binary_COHP:
                    temp_MN_index = np.array([x[0] for x in list(filter(lambda x:x[1][0] > x[1][1], list(zip(candi_edge_index, cohp_pred))))])
                else:
                    temp_MN_index = np.array([x[0] for x in list(filter(lambda x:x[1] <= self.threshold, list(zip(candi_edge_index, cohp_pred))))])
                
                good_MN_index = torch.tensor(temp_MN_index).T
                edge_index = torch.hstack((ori_edge_index, good_MN_index)).to(torch.int64)
                data = Data(x=nodes_features, edge_index=edge_index, y=energy)
                data.slab = graph.slab
                data.metal_pair = graph.metal_pair
                data.aug_index = 0
                data_list.append(data)

                if self.Data_Augmentation:
                    qv = graph.slab.split("_")[0]

                    for aug_index, (filtered_aug_edge_index, aug_num_atoms) in enumerate(zip(augs_info_dict[qv]["index"],
                                                                                             augs_info_dict[qv]["nums"])):

                        num_C_more = aug_num_atoms - nodes_features.shape[0]
                        C_tensor = nodes_features[0].repeat(num_C_more).reshape(num_C_more, -1)
                        
                        edge_index = torch.hstack((filtered_aug_edge_index, good_MN_index)).to(torch.int64)
                        data = Data(x=torch.vstack((nodes_features, C_tensor)), edge_index=edge_index, y=energy)
                        data.slab = graph.slab
                        data.metal_pair = graph.metal_pair
                        data.aug_index = aug_index+1
                        data_list.append(data)
            else:
                continue

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])
                
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        super(CombinedDataset, self).__init__()
        self.datasets = datasets
        self.data_list = self._combine_datasets()

    def _combine_datasets(self):
        combined_data = []
        for dataset in self.datasets:
            combined_data.extend(dataset)
        return combined_data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]



class POS2CLS_Dataset(Dataset):
    def __init__(self, root, set_names, Element_List, Metals):
        self.set_names = set_names
        self.Element_List = Element_List
        self.Metals = Metals
        super(POS2CLS_Dataset, self).__init__(root, transform=None, pre_transform=None)
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        classes_dict = {c: np.identity(len(self.set_names))[i, :] for i, c in enumerate(self.set_names)}
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))
        unsym = list(map(lambda x:x[0]+"_"+x[1],list(product(self.Metals, self.Metals))))
        sym = list(map(lambda x:x[0]+"_"+x[1],list(combinations_with_replacement(self.Metals, 2)))) 
        
        for structure_set in self.set_names:
            qv = structure_set.split("_")[0]
            ele_combos_list = unsym if "QV4" in structure_set else sym
            ori_poscar = Poscar.from_file(os.path.join("sample_space", "%s.vasp"%qv)).structure
            if "C0N" in structure_set:
                None
            else:
                scopy = copy.deepcopy(ori_poscar)
                none = list(map(lambda idx:scopy.replace(idx, "C") if scopy[idx].specie.name == "N" else None, range(len(scopy))))
                ori_poscar = scopy
            N_index = len([site for site in ori_poscar.sites if site.specie.name == "N"])
            name_index = "".join([str(i) for i in range(N_index)])

            for ele_combo in ele_combos_list:
                connectivity = np.array(get_connectivity(ori_poscar))
                eles = np.array([site.specie.name for site in ori_poscar][0:-2] + re.split("_", ele_combo))
                specific_name = qv + "_" + name_index + "_" + ele_combo
                
                onehot = encoder.transform(eles.reshape(-1,1))
                x = torch.tensor(onehot, dtype=torch.float)
                edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()
                y = torch.from_numpy(np.array(classes_dict[structure_set],dtype="float32")).unsqueeze(0)

                data = Data(x=x, edge_index=edge_index, y=y, cls=structure_set, name=specific_name)
                data_list.append(data)

        self.data = data_list
      
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class POS2CLS_Dataset_ALL(Dataset):
    def __init__(self, root, Element_List, Metals, transform=None, pre_transform=None):
        self.Element_List = Element_List
        self.Metals = Metals
        super(POS2CLS_Dataset_ALL, self).__init__(root, transform, pre_transform) 
        self.data = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["POS2CLS_ALL.pt"]
    
    def download(self):
        pass
    
    def process(self):
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
            ALL_N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ALL_N_structure[x].specie.name == "N" else None, range(len(ALL_N_structure))))))
            for num_C in range(1, len(ALL_N_idx)): # Do not consider all_C and all_N
                sample_space_by_names[qv][num_C] = []
                candi_list = list(combinations(ALL_N_idx, num_C))
                for candi in candi_list:
                    number_name = "".join(str(x) for x in [x-min(ALL_N_idx) for x in candi])
                    ele_pairs = unsym if "QV4" in qv else sym
                    for ele_pair in ele_pairs:
                        specific_name = qv + "_" + number_name + "_" + ele_pair
                        sample_space_by_names[qv][num_C].append(specific_name) 
                        
                        C_idx = [int(c) for c in number_name]
                        changed_C_idx = np.array(ALL_N_idx)[C_idx]
                        eles = np.array([site.specie.name for site in ALL_N_structure][0:-2] + ele_pair.split("_"))
                        eles[changed_C_idx] = "C"
                        onehot = encoder.transform(eles.reshape(-1,1))
                        x = torch.tensor(onehot, dtype=torch.float)
                        y = torch.tensor(np.array([0.]*12)).unsqueeze(0)
                        
                        data = Data(x=x, edge_index=edge_index, cls="sample", name=specific_name)
                        data_list.append(data)
                   
        total_num_samples = sum(list(map(lambda x:sum(list(map(lambda y:len(y), x.values()))), list(sample_space_by_names.values()))))
        all_name_list = [i for j in list(map(lambda x:[i for j in list(x.values()) for i in j], list(sample_space_by_names.values()))) for i in j]
        torch.save(data_list, self.processed_paths[0])

      
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class POSCOHP2E_Dataset(Dataset):
    def __init__(self, root, src_dataset=None, COHP_data_sets=None, E_data_dict=None, transform=None, pre_transform=None):
        self.src_dataset = src_dataset
        self.COHP_data_sets = COHP_data_sets
        self.E_data_dict = E_data_dict
        super(POSCOHP2E_Dataset, self).__init__(root, transform, pre_transform) 
        #self.data = torch.load(self.processed_paths[0])  
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return []
    
    def download(self):
        pass
    
    def process(self):
        COHP_data_dict = {}
        for cohp_set in self.COHP_data_sets:
            with open(os.path.join(self.root, "raw", "icohp_%s.pkl")%(cohp_set), "rb") as pklf:
                COHP_data_dict[cohp_set] = pickle.load(pklf)
                
        data_list = []
        for g_index, graph in enumerate(self.src_dataset):
            nodes_features = graph.x
            energy = self.E_data_dict[graph.slab][graph.metal_pair][-1]

            ori_edge_index = graph.edge_index[:,:-18].to(torch.int64)
            good_MN_index = extract_COHP(graph, COHP_data_dict)
            
            edge_index = torch.hstack((ori_edge_index, good_MN_index))
            data = Data(x=nodes_features, edge_index=edge_index, y=energy)
            data_list.append(data)
            
            assert edge_index.shape[1] % 2 == 0

        self.data = data_list
        #torch.save(data_list, self.processed_paths[0])
       
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

"""
class FF_POS2COHP_Dataset(Dataset):
    def __init__(self, root, structure_sets=None, Element_List=None, Metals=None, transform=None, pre_transform=None, pre_filter=None):
        self.structure_sets = structure_sets
        self.Element_List = Element_List
        self.Metals = Metals
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data = torch.load(self.processed_paths[0])      
        
    @property
    def raw_file_names(self) :
        return ['icohp_%s.pkl'% ss for ss in self.structure_sets]

    @property
    def processed_file_names(self) :
        return ["FF_POS2COHP.pt"]
    
    def download(self):
        None

    def process(self): 
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))
        unsym = list(map(lambda x:x[0]+"_"+x[1],list(product(self.Metals, self.Metals))))
        sym = list(map(lambda x:x[0]+"_"+x[1],list(combinations_with_replacement(self.Metals, 2)))) 

        data_list = []
        for structure_set in self.structure_sets:
            with open(os.path.join("raw", "icohp_%s.pkl")%(structure_set), "rb") as pklf:
                icohp_list_dict = pickle.load(pklf)
            ori_poscar = Poscar.from_file(os.path.join("%s"%structure_set, "%s.vasp"%structure_set)).structure   
            ele_combos_list = unsym if "QV4" in structure_set else sym
            
            for ele_combo in ele_combos_list:
                connectivity = np.array(get_connectivity(ori_poscar))
                eles = np.array([site.specie.name for site in ori_poscar][0:-2] + re.split("_", ele_combo))

                onehot = encoder.transform(eles.reshape(-1,1))
                x = torch.tensor(onehot, dtype=torch.float)
                edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()
                
                MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, None, connectivity)
                
                data_ = Data(x=x, edge_index=edge_index)
                data_.MN_edge_index = MN_edge_index
                data_.slab = structure_set
                data_.metal_pair = ele_combo
                data_list.append(data_)

        torch.save(data_list, self.processed_paths[0])


    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class FF_POS2E_Dataset(Dataset):
    def __init__(self, root, src_dataset=None, predicted_value=None, transform=None, pre_transform=None):
        self.src_dataset = src_dataset
        self.pred = predicted_value
        super(FF_POS2E_Dataset, self).__init__(root, transform, pre_transform) 
        self.data = torch.load(self.processed_paths[0])  
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["FF_POS2E.pt"]
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        for g_index, graph in enumerate(self.src_dataset):
            nodes_features = graph.x

            ori_edge_index = graph.edge_index[:,:-18]
            candi_edge_index = graph.MN_edge_index.T.numpy()
            cohp_pred = self.pred[g_index]

            good_MN_index = torch.tensor(np.array([x[0] for x in list(filter(lambda x:x[1] <= -0.6, list(zip(candi_edge_index, cohp_pred))))])).T
            edge_index = torch.hstack((ori_edge_index, good_MN_index))
            data = Data(x=nodes_features, edge_index=edge_index)
            data.slab = graph.slab
            data.metal_pair = graph.metal_pair
            data_list.append(data)
            
        torch.save(data_list, self.processed_paths[0])

                
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

"""

















class POS2EwithoutCOHP_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None, icohp_list_keys=None):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.icohp_list_keys = icohp_list_keys
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None, )
        self.root = root   
        self.data = torch.load(self.processed_paths[0])     
        
    @property
    def raw_file_names(self) :
        return ["icohp_structures_all.pkl"]

    @property
    def processed_file_names(self) :
        return ["POS2EwithoutCOHP.pt"]
    
    def download(self):
        None

    def process(self): 
        ###                                                                     ###
        def match_key(input1, input2, dictionary):#根据边张量找edge_attr,也就是COHP值
            if input1>input2:
                input1,input2=input2,input1
            # 将输入的两个数字加1
            num1 = input1 + 1
            num2 = input2 + 1
            
            # 构建正则表达式模式，假设C和Co可以是任意字母
            pattern = f"^[A-Za-z]+{num1}_[A-Za-z]+{num2}$"
            
            # 遍历字典的键，匹配正则表达式
            for key in dictionary.keys():
                if re.match(pattern, key):
                    return key
            return None
        ###                                                                    ###
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        with open(self.raw_paths[0], "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)
        ##########################################################################
        with open('raw/raw_energy_data_dict_all.pkl','rb') as file:
            raw_energy_data_dict_all = pickle.load(file)
            exist_energy_structure = raw_energy_data_dict_all.keys()
        # for folder, cohp_res in icohp_list_dict.items(): # 遍历icohp_list
        for folder in self.icohp_list_keys:
            cohp_res = icohp_list_dict[folder]
            if not folder in exist_energy_structure:
                continue
            qv,c_index,ele1,ele2 = folder.split("_")
            ori_poscar = Poscar.from_file("./sample_space/%s.vasp"%qv).structure
            connectivity = np.array(get_connectivity(ori_poscar))
            idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
            N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
            first_N = min(N_idx)
            new_structure = copy.deepcopy(ori_poscar)
            for idx in c_index:
                idx = first_N+int(idx)
                new_structure.replace(idx, "C")

            temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
            temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
            Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
            fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
            eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])
            
            if self.Fake_Carbon:
                onehot  = encoder.transform(fake_eles.reshape(-1,1))
            else:
                onehot = encoder.transform(eles.reshape(-1,1))
            
            x = torch.tensor(onehot, dtype=torch.float)
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()
            
            ###                     加入EDGE_ATTR                    ###
            bonds = edge_index.T
            cohp_real = []
            for bond in bonds:
                num1,num2 = bond[0], bond[1]
                bond_key = match_key(num1,num2,cohp_res)
                if bond_key:
                    cohp_real.append([cohp_res[bond_key]])
                else:
                    cohp_real.append([-5])
            edge_attr = torch.tensor(cohp_real)
            ###                                                      ###

            
            # binary_icohp = torch.from_numpy(np.array(list(map(lambda cohp:[1, 0] if cohp <= self.threshold else [0, 1], MN_icohp)),dtype="float32"))

            # fake_node_index = torch.Tensor(np.arange(MN_edge_index.shape[1])).to(torch.int64).unsqueeze(0)
            # MN_fake_node_index = torch.vstack((MN_edge_index[0],fake_node_index,MN_edge_index[1]))
            # fake_x = torch.vstack(list(map(lambda i:x[MN_edge_index[:,i][0].item()] + x[MN_edge_index[:,i][1].item()], np.arange(MN_edge_index.shape[1]))))

            if self.Hetero_Graph:
                data = HeteroData()
                data['atoms'].x = x
                # data['cohps'].x = fake_x

                data['atoms', 'interacts', 'atoms'].edge_index = edge_index
                # data['cohps', 'interacts', 'cohps'].edge_index = torch.vstack([fake_node_index, fake_node_index])
                # data['atoms', 'interacts', 'cohps'].edge_index = torch.hstack((torch.vstack((MN_fake_node_index[0,:],MN_fake_node_index[1,:])),
                #                                                                    torch.vstack((MN_fake_node_index[-1,:],MN_fake_node_index[1,:]))))
                # if self.Binary_COHP:
                #     data.MN_icohp = binary_icohp
                # else:
                #     data.MN_icohp = torch.Tensor(MN_icohp)

                # data.MN_edge_index = MN_edge_index
                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.y = raw_energy_data_dict_all[folder]
                # data.cohp_num = len(MN_icohp)
                data_list.append(data)
            else:
                data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr)
                # data.MN_edge_index = MN_edge_index

                # if self.Binary_COHP:
                #     data.MN_icohp = binary_icohp
                # else:
                #     data.MN_icohp = torch.Tensor(MN_icohp)

                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.y = raw_energy_data_dict_all[folder]
                # data.cohp_num = len(MN_icohp)
                data_list.append(data)                    

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    









class POS2COHP_onehot_and_physical_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None,mode='only_physical'):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.mode = mode
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None, )
        self.root = root   
        self.data = torch.load(self.processed_paths[0])     
        
    @property
    def raw_file_names(self) :
        return ["icohp_structures_all.pkl"]

    @property
    def processed_file_names(self) :
        return ["POS2COHP_%s_only_phy.pt"%self.suffix, "POS2COHP_%s_onehot_and_phy.pt"%self.suffix]
    
    def download(self):
        None

    def process(self): 
        
        # 使用字典缓存对象 反复构造对象很慢
        element_cache = {}
        if self.Fake_Carbon:
            for symbol in self.Element_List[:-1]:
                # print(symbol)
                element_cache[symbol] = element(symbol)
        else:
            for symbol in self.Element_List:
                # print(symbol)
                element_cache[symbol] = element(symbol)
        
        # 构建字典
        df_radii = pd.read_excel('E:\\ResearchGroup\\AdsorbEPred\\pre_set.xlsx',sheet_name='Radii_X')
        df_ip = pd.read_excel('E:\\ResearchGroup\\AdsorbEPred\\pre_set.xlsx',sheet_name='IP')
        element_pyykko_IP_dict = df_radii.iloc[:,:5].set_index('symbol').T.to_dict('dict')
        for idx,col in enumerate(df_ip.columns):
            if col in element_pyykko_IP_dict:
                for i in range(9):
                    if df_ip.iloc[i,idx]==200000:
                        element_pyykko_IP_dict[col][df_ip.iloc[i,0]] = -1
                    else:
                        element_pyykko_IP_dict[col][df_ip.iloc[i,0]] = df_ip.iloc[i,idx]

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        with open(self.raw_paths[0], "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)

        for folder, cohp_res in icohp_list_dict.items(): 
            qv,c_index,ele1,ele2 = folder.split("_")
            ori_poscar = Poscar.from_file("./sample_space/%s.vasp"%qv).structure
            connectivity = np.array(get_connectivity(ori_poscar))
            idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
            N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
            first_N = min(N_idx)
            new_structure = copy.deepcopy(ori_poscar)
            for idx in c_index:
                idx = first_N+int(idx)
                new_structure.replace(idx, "C")

            temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
            temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
            Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
            fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
            eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])
            
            if self.Fake_Carbon:
                onehot  = encoder.transform(fake_eles.reshape(-1,1))
            else:
                onehot = encoder.transform(eles.reshape(-1,1))

            ################################增加特征####################################
            # 创建一个包含r,m,en的矩阵
            # 注意不用fake_eles因为Fc和C性质一样
            additional_features = np.array([[element_cache[ele].atomic_radius_rahm, 
                                             element_cache[ele].mass,
                                             element_cache[ele].electronegativity('pauling'),
                                             element_cache[ele].electronegativity('allred-rochow'),
                                             element_cache[ele].electronegativity('cottrell-sutton'),
                                             element_cache[ele].electronegativity('gordy'),
                                             element_cache[ele].electronegativity(scale='martynov-batsanov'),
                                             element_cache[ele].electronegativity('mulliken'),
                                             element_cache[ele].electronegativity('nagle'),
                                             element_cache[ele].vdw_radius,
                                             element_cache[ele].vdw_radius_alvarez,
                                             element_cache[ele].vdw_radius_mm3,
                                             element_cache[ele].ionenergies[1],
                                             element_cache[ele].dipole_polarizability,
                                             element_cache[ele].heat_of_formation,
                                             element_pyykko_IP_dict[ele]['single'],
                                             element_pyykko_IP_dict[ele]['double'],
                                             element_pyykko_IP_dict[ele]['triple'],
                                             element_pyykko_IP_dict[ele]['IP1'],
                                             element_pyykko_IP_dict[ele]['IP2'],
                                             element_pyykko_IP_dict[ele]['IP3'],
                                             element_pyykko_IP_dict[ele]['IP4'],
                                             ] for ele in eles])

            # 拼接独热编码矩阵和附加特征矩阵
            # print(onehot.shape,additional_features.shape)
            combined_features = np.hstack((onehot, additional_features))

            # 转换为PyTorch tensor
            if self.mode == 'only_physical':
                x = torch.tensor(additional_features, dtype=torch.float)
            else:
                x = torch.tensor(combined_features, dtype=torch.float)
            # print(folder)
            # print(fake_eles)
            # print(x.shape)
            # print(x)
            ######################################################################



            # x = torch.tensor(onehot, dtype=torch.float)
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()

            MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)
            binary_icohp = torch.from_numpy(np.array(list(map(lambda cohp:[1, 0] if cohp <= self.threshold else [0, 1], MN_icohp)),dtype="float32"))

            fake_node_index = torch.Tensor(np.arange(MN_edge_index.shape[1])).to(torch.int64).unsqueeze(0)
            MN_fake_node_index = torch.vstack((MN_edge_index[0],fake_node_index,MN_edge_index[1]))
            fake_x = torch.vstack(list(map(lambda i:x[MN_edge_index[:,i][0].item()] + x[MN_edge_index[:,i][1].item()], np.arange(MN_edge_index.shape[1]))))

            if self.Hetero_Graph:
                data = HeteroData()
                data['atoms'].x = x
                data['cohps'].x = fake_x

                data['atoms', 'interacts', 'atoms'].edge_index = edge_index
                data['cohps', 'interacts', 'cohps'].edge_index = torch.vstack([fake_node_index, fake_node_index])
                data['atoms', 'interacts', 'cohps'].edge_index = torch.hstack((torch.vstack((MN_fake_node_index[0,:],MN_fake_node_index[1,:])),
                                                                                   torch.vstack((MN_fake_node_index[-1,:],MN_fake_node_index[1,:]))))
                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.MN_edge_index = MN_edge_index
                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data_list.append(data)
            else:
                data = Data(x=x, edge_index=edge_index)
                data.MN_edge_index = MN_edge_index

                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data_list.append(data)                    

        self.data = data_list
        if self.mode == 'only_physical':
            torch.save(data_list, self.processed_paths[0])
        else:
            torch.save(data_list, self.processed_paths[1])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    




class CONT2EwithoutCOHP_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None,icohp_list_keys=None):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.icohp_list_keys = icohp_list_keys
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None, )
        self.root = root   
        self.data = torch.load(self.processed_paths[0])     
        
    @property
    def raw_file_names(self) :
        return ["icohp_structures_all.pkl"]
        

    @property
    def processed_file_names(self) :
        return ["CONT2EwithoutCOHP.pt"]
    
    def download(self):
        None

    def process(self): 
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        with open(self.raw_paths[0], "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)
        ##########################################################################
        with open('raw/raw_energy_data_dict_all.pkl','rb') as file:
            raw_energy_data_dict_all = pickle.load(file)
            exist_energy_structure = raw_energy_data_dict_all.keys()
        # for folder, _ in icohp_list_dict.items(): # 遍历icohp_list
        for folder in self.icohp_list_keys:
            cohp_res = icohp_list_dict[folder]
            # if not folder.split('_')[0] == 'QV4':
            #     continue
            if not folder in exist_energy_structure:
                continue
            qv,c_index,ele1,ele2 = folder.split("_")
            this_contcar = "/root/data/home/hejiangshan/AdsorbEPred/CONTCARs/%s/CONTCAR"%(folder)
            if not os.path.isfile(this_contcar):
                print('%s has E but not CONTCAR'%(this_contcar) )
                continue
            ori_poscar = Poscar.from_file(this_contcar).structure
            connectivity = np.array(get_connectivity(ori_poscar))
            idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
            N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
            # first_N = min(N_idx)
            new_structure = copy.deepcopy(ori_poscar)
            # for idx in c_index:
            #     idx = first_N+int(idx)
            #     new_structure.replace(idx, "C")

            temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
            temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
            Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
            fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
            eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])
            
            if self.Fake_Carbon:
                onehot  = encoder.transform(fake_eles.reshape(-1,1))
            else:
                onehot = encoder.transform(eles.reshape(-1,1))
            
            x = torch.tensor(onehot, dtype=torch.float)
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()

            # MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)
            # binary_icohp = torch.from_numpy(np.array(list(map(lambda cohp:[1, 0] if cohp <= self.threshold else [0, 1], MN_icohp)),dtype="float32"))

            # fake_node_index = torch.Tensor(np.arange(MN_edge_index.shape[1])).to(torch.int64).unsqueeze(0)
            # MN_fake_node_index = torch.vstack((MN_edge_index[0],fake_node_index,MN_edge_index[1]))
            # fake_x = torch.vstack(list(map(lambda i:x[MN_edge_index[:,i][0].item()] + x[MN_edge_index[:,i][1].item()], np.arange(MN_edge_index.shape[1]))))

            if self.Hetero_Graph:
                data = HeteroData()
                data['atoms'].x = x
                # data['cohps'].x = fake_x

                data['atoms', 'interacts', 'atoms'].edge_index = edge_index
                # data['cohps', 'interacts', 'cohps'].edge_index = torch.vstack([fake_node_index, fake_node_index])
                # data['atoms', 'interacts', 'cohps'].edge_index = torch.hstack((torch.vstack((MN_fake_node_index[0,:],MN_fake_node_index[1,:])),
                #                                                                    torch.vstack((MN_fake_node_index[-1,:],MN_fake_node_index[1,:]))))
                # if self.Binary_COHP:
                #     data.MN_icohp = binary_icohp
                # else:
                #     data.MN_icohp = torch.Tensor(MN_icohp)

                # data.MN_edge_index = MN_edge_index
                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.y = raw_energy_data_dict_all[folder]
                # data.cohp_num = len(MN_icohp)
                data_list.append(data)
            else:
                data = Data(x=x, edge_index=edge_index)
                # data.MN_edge_index = MN_edge_index

                # if self.Binary_COHP:
                #     data.MN_icohp = binary_icohp
                # else:
                #     data.MN_icohp = torch.Tensor(MN_icohp)

                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.y = raw_energy_data_dict_all[folder]
                # data.cohp_num = len(MN_icohp)
                data_list.append(data)                    

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    



















class POS2E_edge_Dataset(Dataset):
    def __init__(self, root, src_dataset=None, predicted_value=None, raw_data_dict=None, setting=None, suffix=None, Data_Augmentation=False,
                 maximum_num_atoms=100,real_cohp=True,icohp_list_keys=None):
        self.src_dataset = src_dataset
        self.pred = predicted_value
        self.raw_data_dict = raw_data_dict
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.Data_Augmentation = Data_Augmentation
        self.maximum_num_atoms = maximum_num_atoms
        self.real_cohp = real_cohp
        self.icohp_list_keys = icohp_list_keys
        super(POS2E_edge_Dataset, self).__init__(root, transform=None, pre_transform=None) 
        self.data = torch.load(self.processed_paths[0])  
        

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["POS2E_edge_%s.pt"%self.suffix]
    
    def download(self):
        pass
    
    def process(self):
        data_list = []

        # For Data Augmentation!
        qvs = [f.replace(".vasp", '') for f in os.listdir("sample_space/") if '.vasp' in f]
        ori_poss = [Poscar.from_file(os.path.join("sample_space/", qv+".vasp")).structure for qv in qvs]
        augs = [enumerate_padding_structure(ori_pos, maximum_num_atoms=self.maximum_num_atoms) for ori_pos in ori_poss]
        num_atomss = [[len(s) for s in ss] for ss in augs]

        aug_connectivities = [[get_connectivity(s) for s in sl] for sl in augs]
        aug_edge_index = [[filter_pairs(torch.tensor(aug_conn, dtype=torch.long).t().contiguous(), {56,57}) for aug_conn in aug_conns] for aug_conns in aug_connectivities]
        augs_info_dict = {qvs[i]:{"index":aug_edge_index[i], "nums":num_atomss[i]} for i in range(len(qvs))}

        for g_index, graph in enumerate(self.src_dataset):
            if g_index%100==0:
                print(g_index)
            if self.Hetero_Graph:
                nodes_features = graph.x_dict["atoms"]
                ori_edge_index = filter_pairs(graph.edge_index_dict['atoms', 'interacts', 'atoms'], {56,57})
            else:
                nodes_features = graph.x
                # ori_edge_index = filter_pairs(graph.edge_index, {56,57})
                edge_index = graph.edge_index#键连情况不变

            key = graph.slab + "_" + graph.metal_pair
            if key in self.raw_data_dict:
                energy = self.raw_data_dict[key]

                # candi_edge_index = graph.MN_edge_index.T.numpy()
                cohp_pred = self.pred[g_index]

                # if self.Binary_COHP:
                #     temp_MN_index = np.array([x[0] for x in list(filter(lambda x:x[1][0] > x[1][1], list(zip(candi_edge_index, cohp_pred))))])
                # else:
                #     temp_MN_index = np.array([x[0] for x in list(filter(lambda x:x[1] <= self.threshold, list(zip(candi_edge_index, cohp_pred))))])
                
                # good_MN_index = torch.tensor(temp_MN_index).T
                # edge_index = torch.hstack((ori_edge_index, good_MN_index)).to(torch.int64)
                # print(key)
                # print(edge_index.shape)
                # print(edge_index)
                # print(graph.MN_edge_index.shape)
                # print(graph.MN_edge_index)
                # print(cohp_pred.shape)
                # print(cohp_pred)
                # print(graph.MN_icohp.shape)
                # print(graph.MN_icohp[0].item())
                ###                     加入EDGE_ATTR                    ###
                bonds = edge_index.T
                bonds_with_cohp = graph.MN_edge_index.T
                cohp_pred_for_edge = []
                cohp_real_for_edge = []
                for bond in bonds:
                    # num1,num2 = bond[0], bond[1]
                    bond_has_cohp = False
                    for idx,bond_with_cohp in enumerate(bonds_with_cohp):
                        if bond[0]==bond_with_cohp[0] and bond[1]==bond_with_cohp[1]:
                            # if not self.real_cohp:
                            #     cohp_pred_for_edge.append([cohp_pred[idx]])
                            # elif self.real_cohp:
                            #     cohp_pred_for_edge.append([graph.MN_icohp[idx].item()])
                            cohp_pred_for_edge.append([cohp_pred[idx]])
                            cohp_real_for_edge.append([graph.MN_icohp[idx].item()])
                            bond_has_cohp = True
                            break
                        if bond[0]==bond_with_cohp[1] and bond[1]==bond_with_cohp[0]:
                            # if not self.real_cohp:
                            #     cohp_pred_for_edge.append([cohp_pred[idx]])
                            # elif self.real_cohp:
                            #     cohp_pred_for_edge.append([graph.MN_icohp[idx].item()])
                            cohp_pred_for_edge.append([cohp_pred[idx]])
                            cohp_real_for_edge.append([graph.MN_icohp[idx].item()])
                            bond_has_cohp = True
                            break
                    if not bond_has_cohp:
                        cohp_pred_for_edge.append([-5.0])
                        cohp_real_for_edge.append([-5.0])
                    # bond_key = match_key(num1,num2,cohp_res)
                #     if bond_key:
                #         cohp_real.append([cohp_res[bond_key]])
                #     else:
                #         cohp_real.append([-5])
                edge_attr_real = torch.tensor(cohp_real_for_edge)
                edge_attr_pred = torch.tensor(cohp_pred_for_edge)
                # print(edge_attr)
                ###                                                      ###
                data = Data(x=nodes_features, edge_index=edge_index, y=energy, edge_attr_real=edge_attr_real, edge_attr_pred=edge_attr_pred)
                data.slab = graph.slab
                data.metal_pair = graph.metal_pair
                data.aug_index = 0
                data_list.append(data)

                if self.Data_Augmentation:
                    qv = graph.slab.split("_")[0]

                    for aug_index, (filtered_aug_edge_index, aug_num_atoms) in enumerate(zip(augs_info_dict[qv]["index"],
                                                                                             augs_info_dict[qv]["nums"])):

                        num_C_more = aug_num_atoms - nodes_features.shape[0]
                        C_tensor = nodes_features[0].repeat(num_C_more).reshape(num_C_more, -1)
                        
                        edge_index = torch.hstack((filtered_aug_edge_index, good_MN_index)).to(torch.int64)
                        data = Data(x=torch.vstack((nodes_features, C_tensor)), edge_index=edge_index, y=energy)
                        data.slab = graph.slab
                        data.metal_pair = graph.metal_pair
                        data.aug_index = aug_index+1
                        data_list.append(data)
            else:
                continue

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])
                
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]





class POS2COHP2E_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None, icohp_list_keys=None, part_MN=None,
                 raw_data_dict_E=None):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.icohp_list_keys = icohp_list_keys
        self.part_MN = part_MN
        self.raw_data_dict_E = raw_data_dict_E
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None, )
        self.root = root   
        self.data = torch.load(self.processed_paths[0])     
        
    @property
    def raw_file_names(self) :
        return ["icohp_structures_all.pkl"]

    @property
    def processed_file_names(self) :
        return ["POS2COHP2E.pt"]
    
    def download(self):
        None

    def process(self):      
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        with open(self.raw_paths[0], "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)
        ##########################################################################
        with open('raw/raw_energy_data_dict_all.pkl','rb') as file:
            raw_energy_data_dict_all = pickle.load(file)
            exist_energy_structure = raw_energy_data_dict_all.keys()
        # for folder, cohp_res in icohp_list_dict.items(): # 遍历icohp_list
        for folder in self.icohp_list_keys:
            if not folder in exist_energy_structure:
                continue
            energy = self.raw_data_dict_E[folder]
            cohp_res = icohp_list_dict[folder]
            qv,c_index,ele1,ele2 = folder.split("_")
            ori_poscar = Poscar.from_file("./sample_space/%s.vasp"%qv).structure
            connectivity = np.array(get_connectivity(ori_poscar))
            idx1, idx2 = len(ori_poscar)-2, len(ori_poscar)-1
            N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori_poscar[x].specie.name == "N" else None, range(len(ori_poscar))))))
            first_N = min(N_idx)
            new_structure = copy.deepcopy(ori_poscar)
            for idx in c_index:
                idx = first_N+int(idx)
                new_structure.replace(idx, "C")

            temp = list(map(lambda x: x if idx1 in x or idx2 in x else None, connectivity))
            temp = list(set(sum(map(lambda x:list(x),list(filter(lambda x:x is not None, temp))),[])))
            Fc_idx = list(filter(lambda x:x is not None,list(map(lambda x: x if new_structure[x].specie.name == "C" else None, temp))))
            fake_eles = np.array([new_structure[s].specie.name if s not in Fc_idx else "Fc" for s in range(len(new_structure))][0:-2] + re.split("_", folder)[-2:])
            eles = np.array([site.specie.name for site in new_structure][0:-2] + re.split("_", folder)[-2:])
            
            if self.Fake_Carbon:
                onehot  = encoder.transform(fake_eles.reshape(-1,1))
            else:
                onehot = encoder.transform(eles.reshape(-1,1))
            
            x = torch.tensor(onehot, dtype=torch.float)
            edge_index = torch.tensor(connectivity, dtype=torch.long).t().contiguous()
            
            

            MN_edge_index, MN_icohp = get_MCN_edge_index_and_COHP(eles, cohp_res, connectivity)

            if self.part_MN:
                ###################################### only pred valid edge COHP ####################################
                # 将张量转换为列表以便进行比较
                tensor1_list = edge_index.T.tolist()
                tensor2_list = MN_edge_index.T.tolist()

                # 找到两个张量的交集
                MN_edge_index = [item for item in tensor1_list if item in tensor2_list]
                CCorN_edge_index = [item for item in tensor1_list if item not in MN_edge_index]
                CCorN_edge_num = len(CCorN_edge_index)
                # print(len(tensor1_list), len(MN_edge_index), CCorN_edge_num)
                # 找到交集对应的特征
                # print(MN_edge_index)
                MN_icohp = [MN_icohp[tensor2_list.index(item)] for item in MN_edge_index]
                MN_edge_index = torch.tensor(MN_edge_index).T
                # print(MN_edge_index.shape)
                # print(MN_edge_index)
                # print(len(MN_icohp))
                # print(MN_icohp)
                # print(cohp_res)
                # 除了MN_edge的边数
                
                ####################################################################################################





            binary_icohp = torch.from_numpy(np.array(list(map(lambda cohp:[1, 0] if cohp <= self.threshold else [0, 1], MN_icohp)),dtype="float32"))

            fake_node_index = torch.Tensor(np.arange(MN_edge_index.shape[1])).to(torch.int64).unsqueeze(0)
            MN_fake_node_index = torch.vstack((MN_edge_index[0],fake_node_index,MN_edge_index[1]))
            fake_x = torch.vstack(list(map(lambda i:x[MN_edge_index[:,i][0].item()] + x[MN_edge_index[:,i][1].item()], np.arange(MN_edge_index.shape[1]))))

            if self.Hetero_Graph:
                data = HeteroData()
                data['atoms'].x = x
                data['cohps'].x = fake_x

                data['atoms', 'interacts', 'atoms'].edge_index = edge_index
                data['cohps', 'interacts', 'cohps'].edge_index = torch.vstack([fake_node_index, fake_node_index])
                data['atoms', 'interacts', 'cohps'].edge_index = torch.hstack((torch.vstack((MN_fake_node_index[0,:],MN_fake_node_index[1,:])),
                                                                                   torch.vstack((MN_fake_node_index[-1,:],MN_fake_node_index[1,:]))))
                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.MN_edge_index = MN_edge_index
                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data.CCorN_edge_num = CCorN_edge_num
                data_list.append(data)
            else:
                data = Data(x=x, edge_index=edge_index, y=energy)
                data.MN_edge_index = MN_edge_index

                if self.Binary_COHP:
                    data.MN_icohp = binary_icohp
                else:
                    data.MN_icohp = torch.Tensor(MN_icohp)

                data.slab = qv+"_"+c_index
                data.metal_pair = ele1+"_"+ele2
                data.cohp_num = len(MN_icohp)
                data.CCorN_edge_num = CCorN_edge_num
                data_list.append(data)                 

        self.data = data_list
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
