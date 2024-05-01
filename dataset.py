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


class CONT2E_Dataset(Dataset):
    def __init__(self, root, dataset_name, Element_List=None, transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.Element_List = Element_List
        super(CONT2E_Dataset, self).__init__(root, transform, pre_transform) 
        self.data = torch.load(os.path.join(self.processed_dir, "%s.pt"%self.dataset_name))
        
    @property
    def raw_file_names(self):
        return ["%s_raw_data_dict.pkl"%self.dataset_name]

    @property
    def processed_file_names(self):
        return ["%s.pt"%self.dataset_name]
    
    def download(self):
        pass
    
    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            raw_data_dict = pickle.load(f)
            
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(np.array(self.Element_List).reshape(-1, 1))

        data_list = []
        for name, raw_data in raw_data_dict.items():
            structure = raw_data[0]
            distance_matrix = raw_data[1]
            stability_energy = raw_data[2]

            coordinated_idx_matrix = get_coordinated_environment(structure, distance_matrix)
            reciprocal = check_validaiton(coordinated_idx_matrix)

            if reciprocal:
                nodes_ele = list(map(lambda x:x.specie.name, structure.sites))
                nodes_features = encoder.transform(X = np.array(nodes_ele).reshape(-1,1))
                nodes_features = torch.tensor(nodes_features, dtype=torch.float32)

                edges = [ij for l in list(map(lambda y:list(map(lambda x:(y,x), 
                                                                         coordinated_idx_matrix[y])), 
                                                       np.arange(len(coordinated_idx_matrix)))) for ij in l]

                np_edges = np.array(edges)
                edges_src, edges_dst = np_edges[:,0], np_edges[:,1]
                edge_index = torch.tensor(np.vstack((edges_src, edges_dst)), dtype=torch.long)

                dE = torch.tensor(stability_energy, dtype=torch.float32)

                data = Data(x=nodes_features, edge_index=edge_index, y=dE)
                data_list.append(data)
            else:
                None

        torch.save(data_list, os.path.join(self.root,"processed", "%s.pt"%self.dataset_name))

      
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class POS2COHP_Dataset(Dataset):
    def __init__(self, root, Element_List=None, setting=None, suffix=None):
        self.Element_List = Element_List if not setting["Fake_Carbon"] else Element_List + ["Fc"]
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
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
            
            x = torch.tensor(onehot, dtype=torch.float)
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
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class POS2E_Dataset(Dataset):
    def __init__(self, root, src_dataset=None, predicted_value=None, raw_data_dict=None, setting=None, suffix=None, Data_Augmentation=False):
        self.src_dataset = src_dataset
        self.pred = predicted_value
        self.raw_data_dict = raw_data_dict
        self.Fake_Carbon = setting["Fake_Carbon"]
        self.Binary_COHP = setting["Binary_COHP"]
        self.Hetero_Graph = setting["Hetero_Graph"]
        self.threshold = setting["threshold"]
        self.suffix = suffix
        self.Data_Augmentation = Data_Augmentation
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
        augs = [enumerate_padding_structure(ori_pos,maximum_num_atoms=100) for ori_pos in ori_poss]
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