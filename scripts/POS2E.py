from torch_geometric.loader import DataLoader
from POS2COHP import POS2COHP
from torch.utils.data import Subset
import time
import datetime
from scipy.spatial import KDTree
from mendeleev import element
from Device import device
import random

from nets import *
from functions import *
from dataset import *
from training_utils import *
import itertools


class POS2E():
    def __init__(self, COHP_info_dict=None, splitted_keys=None, config=None, loop=0):

        self.COHP_info_dict = COHP_info_dict
        self.splitted_keys = splitted_keys

        self.edge_involved = config["edge_involved"]
        self.edge_dim = 0 if not self.edge_involved else 1

        self.linear_block_dims=config["linear_block_dims"]
        self.conv_block_dims=config["conv_block_dims"]
        self.adj_conv=config["adj_conv"]
        self.conv=configs_str_mapping[config["conv"]] if not self.edge_involved else configs_str_mapping[config["conv_edge"]]

        self.pool=configs_str_mapping[config["pool"]]
        self.pool_ratio=config["pool_ratio"]
        self.pool_heads=config["pool_heads"]
        self.pool_seq=config["pool_seq"]
        self.pool_layer_norm=config["pool_layer_norm"]
        self.pool_type=config["pool_type"]

        self.batch_size=config["batch_size"]
        self.epochs=config["epochs"]
        self.learning_rate=config["learning_rate"]
        self.weight_decay=config["weight_decay"]

        self.verbose=config["verbose"]
        self.augmentation=config["augmentation"]
        self.maximum_num_atoms=config["maximum_num_atoms"]

        self.loop = loop
        self.root_path = os.path.dirname(os.path.dirname(__file__))

    def train_single_model(self, setting, dataset, PRED, finetune):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print(setting)
        suffix = setting2suffix(setting)
        print(suffix)

        if not self.edge_involved:
            pos2e_dataset = POS2E_Dataset(root=self.root_path, 
                                          setting=setting, 
                                          src_dataset=dataset, 
                                          predicted_value=PRED,
                                          loop=self.loop)
        else:
            print("Edge_involved in Dataset!")
            pos2e_dataset = POS2E_edge_Dataset(root=self.root_path,
                                               setting=setting, 
                                               src_dataset=dataset, 
                                               predicted_value=PRED,
                                               loop=self.loop)

        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        initial_keys = [i for j in self.splitted_keys.values() for i in j]

        tr_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["train"]]
        loop_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) not in initial_keys]
        tr_indices = tr_indices + loop_indices
        vl_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(pos2e_dataset) if temp1(d) in self.splitted_keys["test"]]

        pos2e_tr_dataset = Subset(pos2e_dataset, tr_indices)
        pos2e_vl_dataset = Subset(pos2e_dataset, vl_indices)
        pos2e_te_dataset = Subset(pos2e_dataset, te_indices)

        pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn) 
        pos2e_vl_loader = DataLoader(pos2e_vl_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn) 
        pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn) 

        print(f"POS2E whole dataset: {len(pos2e_dataset)}, "
              f"Training set: {len(pos2e_tr_dataset)}, "
              f"Validation set: {len(pos2e_vl_dataset)}, "
              f"Test set: {len(pos2e_te_dataset)}.")

        if setting["encode"] == "physical":
            node_feats = 22
        elif setting["encode"] == "onehot":
            node_feats = 41 if setting["Fake_Carbon"] else 40
        else:
            node_feats = 63 if setting["Fake_Carbon"] else 62

        model = CONT2E_Net(in_features=node_feats, 
                           edge_dim=self.edge_dim,
                           bias=False,
                           linear_block_dims=self.linear_block_dims, 
                           conv_block_dims=self.conv_block_dims, 
                           adj_conv=self.adj_conv,
                           conv=self.conv, 
                           dropout=0., 
                           pool=self.pool, 
                           pool_dropout=0.,
                           pool_ratio=self.pool_ratio, 
                           pool_heads=self.pool_heads, 
                           pool_seq=self.pool_seq,
                           pool_layer_norm=self.pool_layer_norm,
                           pool_type=self.pool_type)

        if finetune:
            model.load_state_dict(torch.load(os.path.join(self.root_path, "models", "_POS2E_Net_%s.pth"%suffix))) 
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate*1e-1, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        epochs = self.epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = np.float64('inf')
        if finetune:
            if not self.edge_involved:
                log_file = os.path.join(self.root_path, "models", "POS2E_Net_%s_%s.txt"%(suffix, self.loop))
                model_file = os.path.join(self.root_path, "models", "POS2E_Net_%s_%s.pth"%(suffix, self.loop))
            else:
                log_file = os.path.join(self.root_path, "models", "POS2E_edge_Net_%s_%s.txt"%(suffix, self.loop))
                model_file = os.path.join(self.root_path, "models", "POS2E_edge_Net_%s_%s.pth"%(suffix, self.loop))
        else:
            if not self.edge_involved:
                log_file = os.path.join(self.root_path, "models", "POS2E_Net_%s.txt"%(suffix))
                model_file = os.path.join(self.root_path, "models", "POS2E_Net_%s.pth"%(suffix))
            else:
                log_file = os.path.join(self.root_path, "models", "POS2E_edge_Net_%s.txt"%(suffix))
                model_file = os.path.join(self.root_path, "models", "POS2E_edge_Net_%s.pth"%(suffix))
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

            if learning_rate <= 1e-4 and vl_res < best:
                torch.save(model.state_dict(), model_file)
                best = vl_res
                print("Saved Model!") if self.verbose else None
        return None


    def train_all_models(self, finetune=False):
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            self.train_single_model(setting, dataset, PRED, finetune)
        print("Finish the training of all POS2E models.")


    def get_all_models(self):
        dataset_model_dict = {}
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            if not self.edge_involved:
                dataset = POS2E_Dataset(root=self.root_path, 
                                        setting=setting, 
                                        src_dataset=dataset, 
                                        predicted_value=PRED, 
                                        loop = self.loop)
            else:
                dataset = POS2E_edge_Dataset(root=self.root_path, 
                                             setting=setting, 
                                             src_dataset=dataset, 
                                             predicted_value=PRED, 
                                             loop = self.loop)

            if setting["encode"] == "physical":
                node_feats = 22
            elif setting["encode"] == "onehot":
                node_feats = 41 if setting["Fake_Carbon"] else 40
            else:
                node_feats = 63 if setting["Fake_Carbon"] else 62

            model = CONT2E_Net(in_features=node_feats, 
                               edge_dim=self.edge_dim,
                               bias=False,
                               linear_block_dims=self.linear_block_dims, 
                               conv_block_dims=self.conv_block_dims, 
                               adj_conv=self.adj_conv,
                               conv=self.conv, 
                               dropout=0., 
                               pool=self.pool, 
                               pool_dropout=0.,
                               pool_ratio=self.pool_ratio, 
                               pool_heads=self.pool_heads, 
                               pool_seq=self.pool_seq,
                               pool_layer_norm=self.pool_layer_norm,
                               pool_type=self.pool_type)

            if not self.edge_involved:
                model.load_state_dict(torch.load(os.path.join(self.root_path, "models", "POS2E_Net_%s.pth"%suffix)))  
            else:
                model.load_state_dict(torch.load(os.path.join(self.root_path, "models", "POS2E_edge_Net_%s.pth"%suffix))) 
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

            with open("./next_loop/%s.pkl"%datetime.datetime.now().strftime("%Y%m%d%H%M%S"), "wb") as f:
                pickle.dump(sorted_DIFF, f)

            model_list.append((dataset, model, setting, PRED_DICT, top_k_MAE, suffix))

        return sorted(model_list, key=lambda x:x[4], reverse=True)