from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from pymatgen.io.vasp import Poscar
import os
import torch
import numpy as np
import pickle
import re
from scipy.spatial import KDTree
from sklearn.preprocessing import OneHotEncoder
from functions import *
from itertools import product, combinations, combinations_with_replacement
from mendeleev import element
from Device import device


import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from functions import *
from dataset import *
import itertools

from torch_geometric.loader import DataLoader
from datetime import datetime
from Device import device


class POS2EMB():
    def __init__(self, Element_List, Metals, batch_size, setting_dict):

        self.Element_List = Element_List
        self.Metals = Metals
        self.batch_size = batch_size

        possibles = list(product(*list(setting_dict.values())))
        all_possible_settings = list(map(lambda p:dict(zip(list(setting_dict.keys()),p)), possibles))
        self.combinations_settings = all_possible_settings


    def build_single_dataset_prelude(self, setting):
        dataset = POS2EMB_Prel_Dataset(root="./",
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
            data_loader = DataLoader(prelude_dataset, batch_size=self.batch_size, shuffle=True)

            pos2cohp_model.to(device)
            pos2cohp_model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data: split_batch_data(data, pos2cohp_model(data.to(device))), data_loader))
            PRED = [i for j in PRED for i in j]

            self.global_COHP[suffix] = [prelude_dataset, PRED]


    def build_single_dataset_coda(self, setting):
        suffix = setting2suffix(setting)
        src_dataset, predicted_value = self.global_COHP[suffix]
        dataset = POS2EMB_Coda_Dataset(root="./",
                                       setting=setting,
                                       src_dataset=src_dataset,
                                       predicted_value=predicted_value)
        return dataset


    def build_all_datasets_coda(self):
        self.coda_dataset_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)
            coda_dataset = self.build_single_dataset_prelude(setting)
            self.coda_dataset_dict[suffix] = coda_dataset


    def apply_global_emb_prediction(self, pos2e_model_dict):
        self.global_EMB = {}

        for suffix, coda_dataset in self.coda_dataset_dict.items():
            pos2e_model = pos2e_model_dict[suffix][1]
            data_loader = DataLoader(coda_dataset, batch_size=self.batch_size, shuffle=True)

            pos2e_model.eval()
            with torch.no_grad():
                embs, names = zip(*[(pos2e_model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                    ['%s_%s'%(data.to("cpu").slab[idx],
                                              data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))])
                                  for data in data_loader])

            embs = np.vstack(embs)
            names = list(itertools.chain(*names))

            self.global_EMB[suffix] = [coda_dataset, embs, names]


    def get_targets_for_next_loop(self, pos2e_res, selection_ratio=0.05, analogue_n=4): #Only choose the best one?

        for (pos2e_dataset, pos2e_model, setting, PRED_DICT, top_k_MAE, suffix) in pos2e_res: #only one
            sorted_result = sorted(list(map(lambda k:(k,*PRED_DICT[k]), list(PRED_DICT.keys()))),key=lambda x:x[1],reverse=True)
            bad_prediction_list = sorted_result[:int(len(pos2e_dataset) * selection_ratio)]

            [coda_dataset, embs, names] = self.global_EMB[suffix]

            kdtree = KDTree(embs)
            selected_indices = []
            selected_names = []

            for bad_name, bad_ae, bad_emb in bad_prediction_list:
                # analogue_idx starts from 0 but pl index starts from len(tr_indices)
                dd, analogue_idx = kdtree.query(bad_emb, analogue_n)
                selected_indices.extend(analogue_idx)

            selected_names = [names[i] for i in selected_indices]
