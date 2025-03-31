import sys
import re
import os
import yaml
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
from itertools import chain
sys.path.append("./scripts")
from nets import *
from functions import *
from dataset import *
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"






def evaluate_model(pos2cohp_model_path="models/", 
                   pos2e_model_path="models/", 
                   prelude_dataset_path="processed/",
                   coda_dataset_path="processed/",
                   root_path="./", 
                   configs={},
                   from_begining=False,
                   device='cpu',
                   num_for_next_loop=2500,
                   loop=0):

    finetune = True if loop > 0 else False
    setting = dict(zip(list(configs["setting_dict"].keys()), list(map(lambda x:x[0], list(configs["setting_dict"].values())))))
    Element_List = configs["Metals"] + configs["Slabs"]

    # Determine node features based on configuration settings
    if setting["encode"] == "physical":
        node_feats = 22
    elif setting["encode"] == "onehot":
        node_feats = 41 if setting["Fake_Carbon"]else 40
    else:
        node_feats = 63 if setting["Fake_Carbon"] else 62
    
    # Load the pos2cohp model with configurations
    config = configs["POS2COHP"]
    
    pos2cohp_model = POS2COHP_Net(in_feats=node_feats, 
                                  hidden_feats=config["hidden_feats"],
                                  activation=configs_str_mapping[config["activation"]],
                                  predictor_hidden_feats=config["predictor_hidden_feats"],
                                  n_tasks=1, 
                                  predictor_dropout=0.0)
    
    pos2cohp_model.load_state_dict(torch.load(pos2cohp_model_path))
    
    # Load the dataset
    pos2cohp_dataset = POS2COHP_Dataset(root_path, Element_List, setting)
    if finetune:
        loop_dataset = POS2COHP_Dataset(root_path, Element_List, setting, loop)
        pos2cohp_dataset = CombinedDataset([pos2cohp_dataset, loop_dataset])

    # Recalculate the predicted COHP values
    pos2cohp_dataloader = DataLoader(pos2cohp_dataset, batch_size=config["batch_size"], shuffle=False)
    pos2cohp_model.eval()
    pos2cohp_model.to(device)

    PRED = []
    with torch.no_grad():
        for data in pos2cohp_dataloader:
            data = data.to(device)  # Move data to the same device
            prediction = pos2cohp_model(data)
            PRED.extend(split_batch_data(data, prediction))

    # Load raw energy data
    with open(os.path.join(root_path, "raw", "raw_energy_data_dict_all.pkl"), 'rb') as file_get:
        raw_data_dict = pickle.load(file_get)

    # Initialize the POS2E_edge_Dataset
    POS2E_edge_dataset = POS2E_edge_Dataset(root=root_path, 
                                            setting=setting, 
                                            src_dataset=pos2cohp_dataset, 
                                            predicted_value=PRED,
                                            loop=loop)


    # Perform prediction for POS2E model
    config = configs["POS2E"]
    
    pos2e_model = CONT2E_Net(in_features=node_feats, 
                             edge_dim=1,
                             bias=False,
                             linear_block_dims=config["linear_block_dims"],
                             conv_block_dims=config["conv_block_dims"],
                             adj_conv=config["adj_conv"],
                             conv=configs_str_mapping[config["conv_edge"]], #configs_str_mapping[config["conv_edge"]]
                             dropout=0.0, 
                             pool=configs_str_mapping[config["pool"]], 
                             pool_dropout=0.0,
                             pool_ratio=config["pool_ratio"], 
                             pool_heads=config["pool_heads"], 
                             pool_seq=config["pool_seq"],
                             pool_layer_norm=config["pool_layer_norm"],
                             pool_type=config["pool_type"])
    
    pos2e_model.load_state_dict(torch.load(pos2e_model_path))
    
    pos2e_model.eval()
    pos2e_model.to(device)

    POS2E_edge_dataloader = DataLoader(POS2E_edge_dataset, batch_size=config["batch_size"], shuffle=False)

    aes = []
    names = []
    embeddings = []
    with torch.no_grad():
        for data in POS2E_edge_dataloader:
            data = data.to(device)  # Move data to the same device
            prediction = pos2e_model(data).detach().cpu().numpy()
            embedding = pos2e_model(data, return_embedding=True).detach().cpu().numpy()
            true_values = data.y.to("cpu").numpy()
            ae = np.abs(np.subtract(prediction, true_values))
            aes.extend(ae)
            names.extend(['%s_%s' % (data.slab[idx], data.metal_pair[idx]) for idx in range(len(data.metal_pair))])
            embeddings.extend(embedding)
    
    # Check the sorted MAE
    sorted_result = sorted([(i, j, k) for i, j, k in zip(names, aes, embeddings)], key=lambda x: x[1], reverse=True)


    # Do the global embedding calculation
    config = configs["POS2EMB"]
    
    if from_begining:
        prelude_dataset = POS2EMB_Prel_Dataset(root=root_path,
                                               Element_List=Element_List,
                                               Metals=configs["Metals"],
                                               setting=setting) 
        prelude_loader = DataLoader(prelude_dataset, batch_size=config["batch_size"], shuffle=False, worker_init_fn=worker_init_fn)

        with torch.no_grad():
            PRED = list(map(lambda data: split_batch_data(data, pos2cohp_model(data.to(device))), prelude_loader))
        PRED = [i for j in PRED for i in j]

        coda_dataset = POS2EMB_Coda_Dataset(root=root_path,
                                            setting=setting,
                                            src_dataset=prelude_dataset,
                                            predicted_value=PRED,
                                            edge_involved=True)
    else:
        prelude_dataset = torch.load(prelude_dataset_path)
        coda_dataset = torch.load(coda_dataset_path)

    coda_loader = DataLoader(coda_dataset, batch_size=config["batch_size"], shuffle=False, worker_init_fn=worker_init_fn)

    with torch.no_grad():
        embs, eners, names = zip(*[(pos2e_model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                 pos2e_model(data).to("cpu").detach().numpy(),
                                 ['%s_%s'%(data.to("cpu").slab[idx],
                                      data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))])
                           for data in coda_loader])

    embs = np.vstack(embs)
    eners = np.hstack(eners)
    names = list(chain(*names))

    list_result = list(zip(*[(n,e) for n,e in zip(names,eners)]))
    all_energy_result = dict(zip(list_result[0], list_result[1]))
    
    # make sure no existing value
    excluded_names_set = get_selected_names()                      # CHANGED CHANGED CHANGED CHANGED
    _embs, _names = zip(*[(e,n) for e,n in zip(embs,names) if n in excluded_names_set])
    embs, names = zip(*[(e,n) for e,n in zip(embs,names) if n not in excluded_names_set])
    
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

    #selected_names = []

    return selected_names, sorted_result, all_energy_result, (_embs, _names)



if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        configs = yaml.safe_load(file)

    loop = 2

    top_k = 0.1

    result, sorted_result, all_energy_result, spacing = evaluate_model(
        pos2cohp_model_path="models/backup/POS2COHP_Net_FcN_RG_Homo_oneh_loop%s.pth"%loop, 
        pos2e_model_path="models/backup/POS2E_edge_Net_FcN_RG_Homo_oneh_loop%s.pth"%loop, 
        root_path="./", 
        configs=configs,
        from_begining=True,
        device='cuda:0',
        loop=loop)

    top_k_MAE = np.average([x[1] for x in sorted_result[:round(len(sorted_result)*top_k)]])
    print(top_k_MAE, len(sorted_result))

    with open("all_energy_result_%s.pkl"%loop, "wb") as f:
        pickle.dump(all_energy_result, f)