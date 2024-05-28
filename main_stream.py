#%%
import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from nets import *
from functions import *
from dataset import *
from training_utils import *
import itertools

from torch_geometric.loader import DataLoader
from POS2COHP import POS2COHP
from POS2E import POS2E
from POS2CLS import POS2CLS
import pickle


# if __name__ == "__main__":
Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
Slabs = ["C","N"]
Element_List = Metals + Slabs

model_training_flags = {"POS2CLS":True, "POS2COHP":True, "POS2E": True}
#%%
#################### PART_1: Build Embedding Map ####################

ALL_Classes = ['QV1_C0N6', 'QV1_C6N0', 'QV2_C0N6', 'QV2_C6N0', 'QV3_C0N6', 'QV3_C6N0', 
               'QV4_C0N7', 'QV4_C7N0', 'QV5_C0N8', 'QV5_C8N0', 'QV6_C0N7', 'QV6_C7N0']
pos2cls = POS2CLS(ALL_Classes, Element_List, Metals)
if model_training_flags["POS2CLS"]:
    pos2cls.train_POS2CLS_model()
pos2cls.get_all_embeddings_and_build_KDTree()
#%%
#################### PART_2: Train 8 POS2COHP Models ####################
setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}
pos2cohp = POS2COHP(Element_List, setting_dict,
                    split_ratio=0.9, batch_size=64, hidden_feats=[256,256,256,256], 
                    predictor_hidden_feats=128, epochs=100, verbose=True)
pos2cohp.build_raw_data_dict()
if model_training_flags["POS2COHP"]:
    pos2cohp.train_all_models()
dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E()
#%%
##########
import pickle
with open('./dataset_model_dict_with_PRED','wb') as file:
    pickle.dump(dataset_model_dict_with_PRED, file)
#%%
#################### PART_3: Train 8 POS2E Models ####################
with open('./dataset_model_dict_with_PRED', 'rb') as file:
    dataset_model_dict_with_PRED = pickle.load(file)
pos2e = POS2E(dataset_model_dict_with_PRED, split_ratio=0.9, batch_size=48, dim=256, epochs=300, 
              linear_dim_list=[],
              conv_dim_list=[[256,256],[256,256],[256,256],[256,256],],verbose=True,Element_List=Element_List)
pos2e.build_raw_data_dict() 
for suffix, [dataset, model, setting, PRED] in pos2e.COHP_info_dict.items():
    if suffix=="FcN_BC_Homo":
        print(suffix)
        print('dataset',dataset,len(dataset))
        pos2e_dataset, model = pos2e.train_single_model(suffix, dataset, setting, PRED)
#%%
if model_training_flags["POS2E"]:
	pos2e.train_all_models()
sorted_model_with_infos = pos2e.get_all_models_sorted(top_k=0.05)
#%%
#################### PART_4: Get the poscar for next loop ####################
bad_predictions = sorted_model_with_infos[0][5]
print(bad_predictions)
bad_neighbors = sum(map(lambda k:pos2cls.find_k_nearest_neighbors(k), bad_predictions), [])
print(bad_neighbors)
checked_points = list(map(lambda k:k if k in pos2cls.name_emb_dict else pos2cls.rvs(k), os.listdir("./structures_all/")))
poscar_names_next_loop = list(set(bad_neighbors).difference(set(checked_points)))
print(poscar_names_next_loop)
next_loop_folder = "structures_next_loop"
os.mkdir(next_loop_folder) if not os.path.exists(next_loop_folder) else None
none = list(map(lambda name:name2structure(next_loop_folder, name), poscar_names_next_loop))
#################### PART_5: Submit Poscars to VASP  ####################
