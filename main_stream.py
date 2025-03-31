import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),"scripts"))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import shutil
import random
import datetime
import pickle
import itertools
import yaml

from functions import restart, set_seed
from Device import device
from POS2COHP import POS2COHP
from POS2E import POS2E
from POS2EMB import POS2EMB
from STR2E import STR2E

with open("config.yaml", "r") as file:
    configs = yaml.safe_load(file)


Metals = configs["Metals"]
Slabs = configs["Slabs"]
Element_List = Metals + Slabs
global_embedding = configs["global_embedding"]
loop = 1
finetune = True if loop > 0 else False


set_seed(seed=43)
if __name__ == "__main__":
    splitted_keys = restart(new_split=True, Random=True, split_ratio=[0.9,0.05,0.05], loop=loop)
    splitted_keys['train'].remove("QV2_012345_Co_Tl")
    splitted_keys['valid'].append("QV2_012345_Co_Tl")
    assert "QV2_012345_Co_Tl" not in splitted_keys['train'] and "QV2_012345_Co_Tl" in splitted_keys['valid']

    ################### PART_1: Build Global Embedding Prelude ####################
    if global_embedding:

      restart(criteria="POS2EMB_")

      pos2emb = POS2EMB(Element_List=Element_List, Metals=Metals, setting_dict=configs["setting_dict"], config=configs["POS2EMB"])

      pos2emb.build_all_datasets_prelude()


    ################### PART_2: Train POS2COHP Models ####################
    restart(criteria="POS2COHP_")

    pos2cohp = POS2COHP(Element_List=Element_List, 
                        setting_dict=configs["setting_dict"], 
                        splitted_keys=splitted_keys,
                        config=configs["POS2COHP"],
                        loop=loop)

    pos2cohp.train_all_models(finetune=finetune, mix_ratio=10)

    pos2cohp_dataset_model_dict = pos2cohp.get_all_models(finetune=finetune)

    pos2cohp_dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E(pos2cohp_dataset_model_dict)

    pos2cohp_res = pos2cohp.get_all_models_sorted(pos2cohp_dataset_model_dict_with_PRED)

    #################### PART_3: Build Global Embedding Coda ####################
    if global_embedding:

      pos2emb.apply_global_cohp_prediction(pos2cohp_model_dict=pos2cohp_dataset_model_dict)

      pos2emb.build_all_datasets_coda(edge_involved=configs["POS2E"]["edge_involved"])


    ############################# PART_4: Train POS2E Models ###########################
    restart(criteria="POS2E_edge_")#"POS2E_edge_"

    pos2e = POS2E(COHP_info_dict=pos2cohp_dataset_model_dict_with_PRED, 
                  splitted_keys=splitted_keys, 
                  config=configs["POS2E"],
                  loop=loop)

    pos2e.train_all_models(finetune=finetune)

    pos2e_dataset_model_dict = pos2e.get_all_models(finetune=finetune)

    pos2e_dataset_model_dict_with_PRED_DICT = pos2e.build_bridge_for_EMB(pos2e_dataset_model_dict)

    pos2e_res = pos2e.get_all_models_sorted(pos2e_dataset_model_dict_with_PRED_DICT)      

    #################### PART_5: Active Learning for Next Loop ####################
    if global_embedding:

      pos2emb.apply_global_emb_prediction(pos2e_model_dict=pos2e_dataset_model_dict)

      names_for_next_loop = pos2emb.get_targets_for_next_loop(pos2e_res, num_for_next_loop=configs["POS2EMB"]["NEXT_LOOP_NUM"])

      #list(map(lambda name:name2structure(name), bad_analogues))
