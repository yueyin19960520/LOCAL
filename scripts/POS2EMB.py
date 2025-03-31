import os
import torch
import numpy as np
import pickle
import re
from scipy.spatial import KDTree
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import DataLoader
from itertools import product, combinations, combinations_with_replacement, chain
from torch_geometric.data import Data, HeteroData, Dataset
from pymatgen.io.vasp import Poscar
from mendeleev import element
from Device import device
import datetime

from functions import *
from dataset import *


class POS2EMB():
    def __init__(self, Element_List=None, Metals=None, setting_dict=None, config=None):

        self.Element_List = Element_List
        self.Metals = Metals
        self.batch_size = config["batch_size"]

        possibles = list(product(*list(setting_dict.values())))
        all_possible_settings = list(map(lambda p:dict(zip(list(setting_dict.keys()),p)), possibles))
        self.combinations_settings = all_possible_settings

        self.root_path = os.path.dirname(os.path.dirname(__file__))


    def build_single_dataset_prelude(self, setting):
        dataset = POS2EMB_Prel_Dataset(root=self.root_path,
                                       Element_List=self.Element_List,
                                       Metals=self.Metals,
                                       setting=setting)
        return dataset


    def build_all_datasets_prelude(self):
        self.prelude_dataset_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)
            prelude_dataset = self.build_single_dataset_prelude(setting)
            self.prelude_dataset_dict[suffix] = prelude_dataset


    def apply_global_cohp_prediction(self, pos2cohp_model_dict):
        self.global_COHP = {}

        for suffix, prelude_dataset in self.prelude_dataset_dict.items():
            pos2cohp_model = pos2cohp_model_dict[suffix][1]
            data_loader = DataLoader(prelude_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

            pos2cohp_model.to(device)
            pos2cohp_model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data: split_batch_data(data, pos2cohp_model(data.to(device))), data_loader))
            PRED = [i for j in PRED for i in j]

            self.global_COHP[suffix] = [prelude_dataset, PRED]


    def build_single_dataset_coda(self, setting, edge_involved):
        suffix = setting2suffix(setting)
        src_dataset, predicted_value = self.global_COHP[suffix]
        dataset = POS2EMB_Coda_Dataset(root=self.root_path,
                                       setting=setting,
                                       src_dataset=src_dataset,
                                       predicted_value=predicted_value,
                                       edge_involved=edge_involved)
        return dataset


    def build_all_datasets_coda(self,edge_involved):
        self.coda_dataset_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)
            coda_dataset = self.build_single_dataset_coda(setting, edge_involved)
            self.coda_dataset_dict[suffix] = coda_dataset


    def apply_global_emb_prediction(self, pos2e_model_dict):
        self.global_EMB = {}

        for suffix, coda_dataset in self.coda_dataset_dict.items():
            pos2e_model = pos2e_model_dict[suffix][1]
            data_loader = DataLoader(coda_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

            pos2e_model.eval()
            with torch.no_grad():
                embs, names = zip(*[(pos2e_model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                    ['%s_%s'%(data.to("cpu").slab[idx],
                                              data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))])
                                  for data in data_loader])

            embs = np.vstack(embs)
            names = list(chain(*names))

            self.global_EMB[suffix] = [coda_dataset, embs, names]


    def get_targets_for_next_loop(self, pos2e_res, num_for_next_loop=2500): #Only choose the best one?

        for (pos2e_dataset, pos2e_model, setting, PRED_DICT, top_k_MAE, suffix) in pos2e_res: #only one
            sorted_result = sorted(
                list(map(lambda k:(k,*PRED_DICT[k]), list(PRED_DICT.keys()))),
                key=lambda x:x[1],
                reverse=True)
            print(sorted_result[0])
            [coda_dataset, embs, names] = self.global_EMB[suffix]
            print(len(names))
            print(names[0])
            # make sure no existing value
            #excluded_names_set = expand_material_ids([i[0] for i in sorted_result]) 
            #excluded_names_set = set([i[0] for i in sorted_result])
            excluded_names_set = get_selected_names()
            embs, names = zip(*[(e,n) for e,n in zip(embs, names) if n not in excluded_names_set])
            print(len(names))
            kdtree = KDTree(embs)
            selected_indices = []

            # Calculate analogue_n using an exponential function
            analogue_counts = []
            for bad_name, bad_ae, bad_emb in sorted_result:
                if bad_ae >= 1.5:
                    analogue_counts.append(4)
                elif 1.5 > bad_ae and bad_ae >= 1.0:
                    analogue_counts.append(3)
                elif 1.0 > bad_ae and bad_ae >= 0.2:
                    analogue_counts.append(2)
                else:
                    analogue_counts.append(1)

            # Query the KDTree based on the calculated analogue counts
            for i, (bad_name, bad_ae, bad_emb) in enumerate(sorted_result):
                if analogue_counts[i] > 0:  
                    dd, analogue_idx = kdtree.query(bad_emb, analogue_counts[i])
                    selected_indices.append(analogue_idx) if isinstance(analogue_idx, np.int64) else selected_indices.extend(analogue_idx)
                    #selected_indices.extend(analogue_idx)
                    if len(selected_indices) >= num_for_next_loop:
                        break

            selected_names = [names[i] for i in selected_indices]

            with open("./next_loop/selected_names_%s.pkl"%datetime.datetime.now().strftime("%Y%m%d%H%M%S"), "wb") as f:
                pickle.dump(selected_names, f)
        return selected_names
