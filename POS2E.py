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
from torch.utils.data import Subset
import time
from datetime import datetime
from scipy.spatial import KDTree
from mendeleev import element
from Device import device
import random


class POS2E():
    def __init__(self, COHP_info_dict, splitted_keys=None, batch_size=48, dim=256, epochs=300, verbose=True,active_learning=False,
                 linear_dim_list=[[256,256]], conv_dim_list=[[256,256],[256,256],[256,256]]):

        self.COHP_info_dict = COHP_info_dict
        self.splitted_keys = splitted_keys
        self.batch_size = batch_size
        self.dim = dim
        self.linear_dim_list = linear_dim_list
        self.conv_dim_list = conv_dim_list
        self.epochs =  epochs
        self.verbose = verbose
        self.active_learning = active_learning
        self.augmentation = False
        self.maximum_num_atoms = 100

    def build_raw_data_dict(self):

        if not os.path.exists(os.path.join("raw", "raw_energy_data_dict_all.pkl")):
            self.raw_data_dict = build_raw_DSAC_file("./", "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
            file_get = open(os.path.join("raw","raw_energy_data_dict_all.pkl"),'rb') 
            self.raw_data_dict = pickle.load(file_get) 
            file_get.close()
        return None

    def train_single_model(self, setting, dataset, PRED, pool_type="tsfm"):
        current_time = datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print(setting)
        suffix = setting2suffix(setting)
        print(suffix)
        pos2e_dataset = POS2E_Dataset(root="./", 
                                      setting=setting, 
                                      src_dataset=dataset, 
                                      predicted_value=PRED, 
                                      raw_data_dict=self.raw_data_dict)

        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        tr_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["train"]]
        vl_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["test"]]

        pos2e_tr_dataset = Subset(pos2e_dataset, tr_indices)
        pos2e_vl_dataset = Subset(pos2e_dataset, vl_indices)
        pos2e_te_dataset = Subset(pos2e_dataset, te_indices)

        pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
        pos2e_vl_loader = DataLoader(pos2e_vl_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
        pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

        print(f"Net-by-Net: "
              f"Training set: {len(pos2e_tr_dataset)}, "
              f"Validation set: {len(pos2e_vl_dataset)}, "
              f"Test set: {len(pos2e_te_dataset)}.")

        if setting["encode"] == "physical":
            node_feats = 22
        elif setting["encode"] == "onehot":
            node_feats = 41 if setting["Fake_Carbon"] else 40
        else:
            node_feats = 63 if setting["Fake_Carbon"] else 62

        model = CONT2E_Net(dim=self.dim, linear_dim_list=self.linear_dim_list,
                           conv_dim_list=self.conv_dim_list,
                           adj_conv=False, in_features=node_feats, bias=False,
                           conv=pyg_GCNLayer_without_edge_attr, dropout=0., pool=GraphMultisetTransformer,
                           pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False,
                           pool_type=pool_type)

        epochs = self.epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = 1e5
        log_file = "./models/POS2E_Net_%s.txt"%(suffix)
        with open(log_file, 'w') as file:
            file.close()

        for epoch in range(epochs):
            tr_loss, tr_res = cont2e_train(model, pos2e_tr_loader, loss_func, optimizer)
            vl_loss, vl_res = cont2e_evaluate(model, pos2e_vl_loader, loss_func)
            te_loss, te_res = cont2e_evaluate(model, pos2e_te_loader, loss_func)
                    
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
            training_info = (f"epoch: {epoch}, "
                             f"Training MAE: {tr_res:.3f} (eV), "
                             f"Validation MAE: {vl_res:.3f} (eV), "
                             f"Test MAE: {te_res:.3f} (eV), "
                             f"Learning Rate: {learning_rate}")
            scheduler.step()

            if self.verbose:
                print(training_info)
            
            with open(log_file, 'a') as file:
                file.write(training_info + "\n")   

            if learning_rate <= 1e-5 and vl_res < best:
                torch.save(model, "./models/POS2E_Net_%s.pth"%(suffix))
                best = vl_res
                print("Saved Model!") if self.verbose else None
        return None


    def train_all_models(self):
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            if self.active_learning:
                self.train_single_model_with_active_learning(setting, dataset, PRED,
                                                             base_lr=1e-3, lr_decay_ratio=0.8, 
                                                             base_wd=1e-4, wd_decay_ratio=0.8,
                                                             selection_ratio=0.03, analogue_n=2)
            else:
                self.train_single_model(setting, dataset, PRED)
        print("Finish the training of all POS2E models.")

    def get_all_models(self):
        dataset_model_dict = {}
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():

            dataset = POS2E_Dataset(root="./", 
                                    setting=setting, 
                                    src_dataset=dataset, 
                                    predicted_value=PRED, 
                                    raw_data_dict=self.raw_data_dict)

            if self.active_learning:
                model = torch.load("./models/POS2E_active_Net_%s.pth"%suffix).to("cpu")
            else:
                model = torch.load("./models/POS2E_Net_%s.pth"%suffix).to("cpu")
            dataset_model_dict[suffix] = [dataset, model, setting]
        return dataset_model_dict

    def build_bridge_for_EMB(self, dataset_model_dict=None):
        if dataset_model_dict == None:
            dataset_model_dict = self.get_all_models()

        for suffix, [dataset, model, setting] in dataset_model_dict.items():
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            model.to(device)
            model.eval()
            with torch.no_grad():
                embs, aes, names = zip(*[
                    (
                        model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                        list(np.abs(np.subtract(model(data.to(device)).detach().cpu().numpy(), data.y.to("cpu").numpy()))),
                        ['%s_%s'%(data.slab[idx],data.metal_pair[idx]) for idx in range(len(data.metal_pair))]
                    )
                    for data in data_loader
                ])
            embs = np.vstack(embs)
            aes = list(itertools.chain(*aes))
            names = list(itertools.chain(*names))

            assert len(embs) == len(aes) == len(names)

            PRED_DICT = {name: (ae, emb) for name, emb, ae in zip(names, embs, aes)}

            dataset_model_dict[suffix].append(PRED_DICT)

        dataset_model_dict_with_PRED_DICT = dataset_model_dict
        return dataset_model_dict_with_PRED_DICT

    def _measure_model(self, sorted_DIFF, top_k):
        MAE = np.average([x[1] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]])
        names_with_MAE = [x[0] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]]
        return MAE, names_with_MAE

    def get_all_models_sorted(self, dataset_model_dict_with_PRED_DICT=None, top_k=0.1):
        if dataset_model_dict_with_PRED_DICT == None:
            dataset_model_dict_with_PRED_DICT = self.build_bridge_for_EMB()

        model_list = []
        for suffix, [dataset, model, setting, PRED_DICT] in dataset_model_dict_with_PRED_DICT.items():
            sorted_DIFF = sorted([(k, np.float32(v[0])) for k,v in PRED_DICT.items()], key=lambda x:x[1], reverse=True)
            top_k_MAE = np.average([x[1] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]])
            print(suffix, ": %.6f"%top_k_MAE)
            model_list.append((dataset, model, setting, PRED_DICT, top_k_MAE, suffix))

        return sorted(model_list, key=lambda x:x[4], reverse=True)


if __name__ == "__main__":
    None
"""
    
    def train_single_model_with_active_learning(self, setting, dataset, PRED, 
                                                base_lr, lr_decay_ratio, base_wd, wd_decay_ratio,
                                                selection_ratio, analogue_n,
                                                pool_type="all"):
        current_time = datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print('train_single_model_with_active_learning!')
        print(setting)
        suffix = setting2suffix(setting)
        print(suffix)

        pos2e_dataset = POS2E_Dataset(root="./", 
                                      setting=setting, 
                                      src_dataset=dataset, 
                                      predicted_value=PRED, 
                                      raw_data_dict=self.raw_data_dict)

        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        tr_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["train"]]
        vl_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["test"]]

        split_ratio = 0.5
        tr_indices, pl_indices = tr_indices[:int(len(tr_indices) * split_ratio)], tr_indices[int(len(tr_indices) * split_ratio):]

        print(f"Net-by-Net: "
              f"Training set and Pooling set: {len(tr_indices)}, "
              f"Validation set: {len(vl_indices)}, "
              f"Test set: {len(te_indices )}.")

        if setting["encode"] == "physical":
            node_feats = 22
        elif setting["encode"] == "onehot":
            node_feats = 41 if setting["Fake_Carbon"] else 40
        else:
            node_feats = 63 if setting["Fake_Carbon"] else 62

            
        loops = 4
        for iteration in range(loops): 
            pos2e_tr_dataset = Subset(pos2e_dataset, tr_indices)
            pos2e_vl_dataset = Subset(pos2e_dataset, vl_indices)
            pos2e_te_dataset = Subset(pos2e_dataset, te_indices)
            pos2e_pl_dataset = Subset(pos2e_dataset, pl_indices)

            pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
            pos2e_vl_loader = DataLoader(pos2e_vl_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
            pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
            pos2e_pl_loader = DataLoader(pos2e_pl_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

            print(f"Iteration:{iteration}, "
                  f"Trainig set:{len(pos2e_tr_dataset)}, "
                  f"Validation set:{len(pos2e_vl_dataset)}, "
                  f"Test set:{len(pos2e_te_dataset)}.")

            if iteration == 0:
                model = CONT2E_Net(dim=self.dim, linear_dim_list=self.linear_dim_list,
                                   conv_dim_list=self.conv_dim_list,
                                   adj_conv=False, in_features=node_feats, bias=False,
                                   conv=pyg_GCNLayer_without_edge_attr, dropout=0., pool=GraphMultisetTransformer,
                                   pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False,
                                   pool_type=pool_type)

                epochs = self.epochs
                loss_func = MSELoss()
                
                lr = base_lr
                wd = base_wd
                
            lr = base_lr * (lr_decay_ratio ** iteration)
            wd = base_wd * (wd_decay_ratio ** iteration)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

            best = 1e5
            log_file = "./models/POS2E_active_Net_%s.txt"%(suffix)
            with open(log_file, 'w') as file:
                file.close()

            for epoch in range(epochs):
                tr_loss, tr_res = cont2e_train(model, pos2e_tr_loader, loss_func, optimizer)
                vl_loss, vl_res = cont2e_evaluate(model, pos2e_vl_loader, loss_func)
                te_loss, te_res = cont2e_evaluate(model, pos2e_te_loader, loss_func)
                
                learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
                training_info = (f"epoch: {epoch}, "
                                 f"Training MAE: {tr_res:.3f} (eV), "
                                 f"Validation MAE: {vl_res:.3f} (eV), "
                                 f"Test MAE: {te_res:.3f} (eV), "
                                 f"Learning Rate: {learning_rate}")
                scheduler.step()
                
                if self.verbose:
                    print(training_info)
                
                with open(log_file, 'a') as file:
                    file.write(training_info + "\n")   

                if learning_rate <=  5e-4 and te_res < best:
                    torch.save(model, "./models/POS2E_active_Net_%s.pth"%(suffix))
                    best = te_res
                    print("Saved Model!") if self.verbose else None

            # In the each iteration! 
            # Bad Points Chosen!
            tr_datas_mae = cont2e_cal(model, pos2e_tr_loader)
            print('tr_datas_mae: ',tr_datas_mae.shape)
            
            look_up_method = "MAE"
            if look_up_method == "random":
                tr_top_indices = random.sample(list(range(len(tr_datas_mae))), int(len(pos2e_dataset) * selection_ratio))
            else:
                tr_top_indices = np.argsort(tr_datas_mae)[-int(len(pos2e_dataset) * selection_ratio):]
                   
            # Getting Training Embeddings and Corresonding Names!
            tr_embeddings, tr_names = zip(*[(
                model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                ['%s_%s'%(data.to("cpu").slab[idx],data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))]
                                            )
                                                          for data in pos2e_tr_loader])
            pl_embeddings, pl_names = zip(*[(
                model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                ['%s_%s'%(data.to("cpu").slab[idx],data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))]
                                            )
                                                          for data in pos2e_pl_loader])
            
            tr_embeddings = np.vstack(tr_embeddings)
            pl_embeddings = np.vstack(pl_embeddings)
            tr_names = list(itertools.chain(*tr_names))
            pl_names = list(itertools.chain(*pl_names))
            
            # KDTree
            kdtree = KDTree(pl_embeddings)
            selected_indices = []
            selected_names = []
            
            if self.verbose:
                print('Number of bad predictions: %s'%len(tr_top_indices))
            for i in tr_top_indices:
                # analogue_idx starts from 0 but pl index starts from len(tr_indices)
                dd, analogue_idx = kdtree.query(tr_embeddings[i], analogue_n, workers=2)
                selected_indices += [pl_indices[j] for j in analogue_idx] 
                selected_names += [pl_names[j] for j in analogue_idx]
            
            # Rebuild the train and pool dataset!
            tr_indices.extend(selected_indices)
            pl_indices = list(set(pl_indices).difference(set(selected_indices)))
        return None
"""