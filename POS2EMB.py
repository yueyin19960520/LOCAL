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

    def build_single_dataset(self, setting):
        dataset = POS2EMB_Dataset(root="./",
                                  Element_List=self.Element_List,
                                  Metals=self.Metals,
                                  setting=setting)
        return dataset

    def build_all_datasets(self):
        dataset_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)
            dataset = self.build_single_dataset(setting)
            dataset_dict[suffix] = dataset
        self.dataset_dict = dataset_dict


    def apply_global_cohp_prediction(self, cohp_model_dict):
        global_prediction = {}

        for suffix, emb_dataset in self.dataset_dict.items():
            cohp_model = cohp_model_dict[suffix][1]
            data_loader = DataLoader(emb_dataset, batch_size=self.batch_size, shuffle=True)

            model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data:split_batch_data(data, model(data)), data_loader))
            PRED = [i for j in PRED for i in j]

            global_prediction[suffix] = [emb_dataset, PRED]
        return global_prediction

    def apply_global_E_prediction(self):
        None
