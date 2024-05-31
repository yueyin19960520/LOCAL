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




print('torch.cuda.is_available:',torch.cuda.is_available())
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if more than 1 GPU
np.random.seed(seed)
current_seed = seed
print(f"Current random seed: {current_seed}")

# if __name__ == "__main__":
Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
Slabs = ["C","N"]
Element_List = Metals + Slabs

model_training_flags = {"POS2CLS":True, "POS2COHP":True, "POS2E": True}
#%%
#################### PART_1: Train 1 POS2COHP Models ####################
setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}
pos2cohp = POS2COHP(Element_List, setting_dict,
                    split_ratio=0.9, batch_size=64, hidden_feats=[256,256,256,256], 
                    predictor_hidden_feats=128, epochs=100, verbose=True)
pos2cohp.build_raw_data_dict()
if model_training_flags["POS2COHP"]:
    pos2cohp.train_all_models()
dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E()

# Skip the PART by saving the intermediate variable #
import pickle
with open('./dataset_model_dict_with_PRED','wb') as file:
    pickle.dump(dataset_model_dict_with_PRED, file)
#%%
#################### PART_2: Train POS2E Models ####################
with open('./dataset_model_dict_with_PRED', 'rb') as file:
    dataset_model_dict_with_PRED = pickle.load(file)
pos2e = POS2E(dataset_model_dict_with_PRED, split_ratio=0.9, batch_size=48, dim=256, epochs=300, 
              linear_dim_list=[],
              conv_dim_list=[[256,256],[256,256],[256,256],[256,256],],verbose=True)
pos2e.build_raw_data_dict() 
for suffix, [dataset, model, setting, PRED] in pos2e.COHP_info_dict.items():
    if suffix=="FcN_BC_Homo":
        print(suffix)
        print('dataset',dataset,len(dataset))
        pos2e_dataset, model = pos2e.train_single_model(suffix, dataset, setting, PRED,pool_type='all',# pool_type=tsfm/add/max/mean/all
                                                        aug=False, maximum_num_atoms=100)
        
#%%
#################### PART_3: Active learning in exist dataset ####################
with open('./dataset_model_dict_with_PRED', 'rb') as file:
    dataset_model_dict_with_PRED = pickle.load(file)
pos2e = POS2E(dataset_model_dict_with_PRED, split_ratio=0.9, batch_size=48, dim=256, epochs=100, 
              linear_dim_list=[],
              conv_dim_list=[[256,256],[256,256],[256,256],[256,256],],verbose=True)
pos2e.build_raw_data_dict() 
for suffix, [dataset, model, setting, PRED] in pos2e.COHP_info_dict.items():
    if suffix=="FcN_BC_Homo":
        print(suffix)
        print('dataset',dataset,len(dataset))
        pos2e_dataset, model = pos2e.train_single_model_with_active_learning(suffix, dataset, setting, PRED, verbose=True,
                                                base_lr=1e-3,base_weight_decay=1e-4, 
                                                lr_iteration_decay_ratio=1,weight_decay_decay_ratio=1,
                                                Metals=Metals,
                                                selection_ratio=0.03,
                                                analogue_n=2)

#%%
#################### PART_4: Get the poscars for next loop ####################
from POS2E_dataset_all import *
sorted_model_with_infos = pos2e.get_all_models_sorted(top_k=0.05) # To get top_k worst 
bad_predictions = sorted_model_with_infos[0][5]
with open('./bad_predictions','wb') as file:
    pickle.dump(bad_predictions, file)
find_analogue_0 = Find_analogue(Metals,Slabs)
find_analogue_0.get_all_cohp_prediction_value() # write name_pos2cohp_output_dict, get all cohp pred value
find_analogue_0.get_all_embeddings_of_POS2E_net_and_build_KDTree() # use name_pos2cohp_output_dict to make all graphs and embeddings
bad_neighbors = sum(map(lambda k:find_analogue_0.find_k_nearest_neighbors(target_point_key=k,k=2), bad_predictions), [])

print('number of bad predictions:',len(bad_predictions)) # top_k
print('number of bad neighbors:',len(bad_neighbors))
print('examples of bad predictions: ',bad_predictions[:2])
print('examples of bad neighbors: ',bad_neighbors[:4])


def rvs(k): 
    return "_".join(k.split("_")[:2] + [k.split("_")[3], k.split("_")[2]])
checked_points = []
for i in os.listdir("structures_all"):
    qv = i.split('_')[0]
    if qv=='QV4':
        checked_points += [i]
    else:
        checked_points += [i,rvs(i)]
print('len(checked_points)',len(checked_points))
'''do not use structures repeatly'''
poscar_names_next_loop = list(set(bad_neighbors).difference(set(checked_points)))
print('len(poscar_names_next_loop)',len(poscar_names_next_loop))
print('poscar_names_next_loop',poscar_names_next_loop)
next_loop_folder = "structures_next_loop"
os.mkdir(next_loop_folder) if not os.path.exists(next_loop_folder) else None
none = list(map(lambda name:name2structure(next_loop_folder, name), poscar_names_next_loop))
#################### PART_5: Submit Poscars to VASP  ####################
'''Do DFT calculations, get new icohp_structures_all.pkl, raw_energy_data_dict_all.pkl and run this code again!'''