#%%
import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import shutil
import random
import datetime
import pickle
import itertools
from torch_geometric.loader import DataLoader

from nets import *
from functions import *
from dataset import *
from training_utils import *
from Device import device

from POS2COHP import POS2COHP
from POS2E import POS2E
from POS2EMB import POS2EMB

#from POS2COHP2E import POS2COHP2E
#from POS2E_edge import POS2E_edge
#from CONT2E_without_COHP import CONT2E_without_COHP
#from POS2E_without_COHP import POS2E_without_COHP


print(os.getcwd())
print('torch.cuda.is_available:',torch.cuda.is_available())

clean = True
list_timestamp = ""

if __name__ == "__main__":
    Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
              "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
              "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
              "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
    Slabs = ["C","N"]
    Element_List = Metals + Slabs

    settings = {'Fake_Carbon': [True],
                'Binary_COHP': [False], 
                'Hetero_Graph': [False], 
                'threshold': [-0.6],
                'encode': ['both']}

    ################### PART_1: Train 1 POS2COHP Models ####################
    pos2emb = POS2EMB(Element_List, Metals, batch_size=64, setting_dict=settings)

    emb_dataset_dict = pos2emb.build_all_datasets()

    ################### PART_2: Train 1 POS2COHP Models ####################
    icohp_list_keys = restart(clean=clean, criteria="POS2COHP")
    if not clean:
        with open('saved_lists/list_%s.pkl'%list_timestamp, 'rb') as file:
            icohp_list_keys = pickle.load(file)

    pos2cohp = POS2COHP(Element_List, setting_dict=settings, icohp_list_keys=icohp_list_keys,
                        split_ratio=[0.80, 0.10, 0.10], batch_size=64, hidden_feats=[256,256,256,256], 
                        predictor_hidden_feats=128, epochs=100, verbose=True)
    
    pos2cohp.build_raw_data_dict()

    #pos2cohp.train_all_models()

    dataset_model_dict = pos2cohp.get_all_models()

    dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E(dataset_model_dict)

    res = pos2cohp.get_all_models_sorted(dataset_model_dict_with_PRED)

    global_cohp_prediction = pos2emb.apply_global_cohp_prediction(cohp_model_dict=dataset_model_dict)

    # #################### PART_3: Train POS2E Models ####################
    restart(clean=clean, criteria="POS2E")

    pos2e = POS2E(COHP_info_dict=dataset_model_dict_with_PRED, splitted_keys=pos2cohp.splitted_keys, 
                  batch_size=48, dim=256, epochs=200, verbose=True, active_learning=False,
                  linear_dim_list=[[256,256]], conv_dim_list=[[256,256],[256,256],[256,256]])

    pos2e.build_raw_data_dict()

    pos2e.train_all_models()

    dataset_model_dict = pos2e.get_all_models()

    dataset_model_dict_with_PRED = pos2e.build_bridge_for_(dataset_model_dict)

    res = pos2e.get_all_models_sorted(dataset_model_dict_with_PRED)

    global_E_prediction = pos2emb.apply_global_E_prediction()

    # #################### PART_3: Active learning in exist dataset TEST ####################

    """
    # #################### PART_6: CONT2E without COHP #####################################

    # setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}
    # cont2ewithoutcohp = CONT2E_without_COHP(Element_List, setting_dict,
    #                     split_ratio=[0.80, 0.10, 0.10], batch_size=48, conv_dim_list=[[256,256],[256,256],[256,256],[256,256],], dim=256,
    #                     epochs=300, verbose=True)
    # # cont2ewithoutcohp.build_raw_data_dict()
    # cont2ewithoutcohp.train_all_models(icohp_list_keys)
    # #%%
    # ################### PART_7: POS2E without COHP #######################################

    # setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}
    # pos2ewithoutcohp = POS2E_without_COHP(Element_List, setting_dict,
    #                     split_ratio=[0.80, 0.10, 0.10], batch_size=1, conv_dim_list=[[256,256],[256,256],[256,256],[256,256],], dim=256,
    #                     epochs=300, verbose=True,edge_dim=1)
    # # cont2ewithoutcohp.build_raw_data_dict()
    # pos2ewithoutcohp.train_all_models(icohp_list_keys)
    #%%
    #################### PART_8: Train POS2E_edge Models ####################

    real_cohp = False
    # if real_cohp:
    #     shutil.copy('processed/POS2E_edge_FcN_RG_Homo_real.pt', 'processed/POS2E_edge_FcN_RG_Homo.pt')
    # else:
    #     shutil.copy('processed/POS2E_edge_FcN_RG_Homo_pred.pt', 'processed/POS2E_edge_FcN_RG_Homo.pt')
    with open('./dataset_model_dict_with_PRED', 'rb') as file:
        dataset_model_dict_with_PRED = pickle.load(file)
    pos2e = POS2E_edge(dataset_model_dict_with_PRED, split_ratio=[0.80, 0.10, 0.10], batch_size=48, dim=256, epochs=300, 
                linear_dim_list=[],
                conv_dim_list=[[256,256],[256,256],[256,256],[256,256],],verbose=True,
                edge_dim=1)
    pos2e.build_raw_data_dict() 
    for suffix, [dataset, model, setting, PRED] in pos2e.COHP_info_dict.items():
        if suffix=="FcN_RG_Homo":
            print(suffix)
            print('dataset',dataset,len(dataset))
            pos2e_dataset, model = pos2e.train_single_model(suffix, dataset, setting, PRED,pool_type='all',# pool_type=tsfm/add/max/mean/all
                                                            aug=False, maximum_num_atoms=100,
                                                            real_cohp=real_cohp,noise=False,noise_type='gaussian',noise_mae=0.4,
                                                            icohp_list_keys=icohp_list_keys)
    # #%%
    # #################### PART_9: Serial POS2COHP2E #################### 
    # setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}
    # pos2cohp2e = POS2COHP2E(Element_List, setting_dict,
    #                     split_ratio=[0.80, 0.10, 0.10], batch_size=64, hidden_feats=[256,256,256,256], 
    #                     predictor_hidden_feats=128, epochs=300, verbose=True,part_MN=True,
                        
    #                     dim=256, conv_dim_list=[[256,256],[256,256],[256,256],[256,256],], edge_dim=1)
    # pos2cohp2e.build_raw_data_dict()
    # if model_training_flags["POS2COHP"]:
    #     pos2cohp_dataset = pos2cohp2e.train_all_models(icohp_list_keys=icohp_list_keys) # encode type
    
    #%%
    #################### PART_10: POS2E_edge train with real COHP #################### 
    real_cohp = False
    # if real_cohp:
    #     shutil.copy('processed/POS2E_edge_FcN_RG_Homo_real.pt', 'processed/POS2E_edge_FcN_RG_Homo.pt')
    # else:
    #     shutil.copy('processed/POS2E_edge_FcN_RG_Homo_pred.pt', 'processed/POS2E_edge_FcN_RG_Homo.pt')
    with open('./dataset_model_dict_with_PRED', 'rb') as file:
        dataset_model_dict_with_PRED = pickle.load(file)
    pos2e = POS2E_edge(dataset_model_dict_with_PRED, split_ratio=[0.80, 0.10, 0.10], batch_size=48, dim=256, epochs=300, 
                linear_dim_list=[],
                conv_dim_list=[[256,256],[256,256],[256,256],[256,256],],verbose=True,
                edge_dim=1)
    pos2e.build_raw_data_dict() 
    for suffix, [dataset, model, setting, PRED] in pos2e.COHP_info_dict.items():
        if suffix=="FcN_RG_Homo":
            print(suffix)
            print('dataset',dataset,len(dataset))
            pos2e_dataset, model = pos2e.train_single_model(suffix, dataset, setting, PRED,pool_type='all',# pool_type=tsfm/add/max/mean/all
                                                            aug=False, maximum_num_atoms=100,
                                                            real_cohp=real_cohp,noise=False,noise_type='gaussian',noise_mae=0.4,
                                                            icohp_list_keys=icohp_list_keys,train_with_real=True)
    """