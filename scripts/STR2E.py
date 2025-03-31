from torch_geometric.loader import DataLoader
from Device import device
import datetime
from torch.utils.data import Subset

from nets import *
from functions import *
from dataset import *
from training_utils import *
import itertools


class STR2E():
    def __init__(self, Element_List, setting_dict, splitted_keys=None, str_type=None, config=None):

        self.Element_List = Element_List  

        possibles = list(product(*list(setting_dict.values())))
        all_possible_settings = list(map(lambda p:dict(zip(list(setting_dict.keys()),p)), possibles))

        self.combinations_settings = all_possible_settings
        self.splitted_keys = splitted_keys
        
        self.edge_dim = 0 

        self.linear_block_dims=config["linear_block_dims"]
        self.conv_block_dims=config["conv_block_dims"]
        self.adj_conv=config["adj_conv"]
        self.conv=configs_str_mapping[config["conv"]]

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
        
        self.str_type = str_type
        self.root_path = os.path.dirname(os.path.dirname(__file__))

    def build_raw_data_dict(self):

        if not os.path.exists(os.path.join(self.root_path, "raw", "raw_energy_data_dict_all.pkl")):
            self.raw_data_dict = build_raw_DSAC_file(self.root_path, "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
            file_get = open(os.path.join(self.root_path, "raw","raw_energy_data_dict_all.pkl"),'rb') 
            self.raw_data_dict = pickle.load(file_get) 
            file_get.close()
        return None

    def train_single_model(self, setting):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print(setting)
        if setting["encode"] == "physical":
            node_feats = 22
        elif setting["encode"] == "onehot":
            node_feats = 41 if setting["Fake_Carbon"] else 40
        else:
            node_feats = 63 if setting["Fake_Carbon"] else 62

        suffix = setting2suffix(setting)
        print(suffix)

        str2e_dataset = STR2E_Dataset(root=self.root_path, 
                                      Element_List=self.Element_List, 
                                      setting=setting,
                                      raw_data_dict=self.raw_data_dict,
                                      str_type=self.str_type)
        
        data_num = len(str2e_dataset)
        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        tr_indices = [i for i,d in enumerate(str2e_dataset) if temp1(d) in self.splitted_keys["train"]]
        vl_indices = [i for i,d in enumerate(str2e_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(str2e_dataset) if temp1(d) in self.splitted_keys["test"]]

        str2e_tr_dataset = Subset(str2e_dataset, tr_indices)
        str2e_vl_dataset = Subset(str2e_dataset, vl_indices)
        str2e_te_dataset = Subset(str2e_dataset, te_indices)

        str2e_tr_loader = DataLoader(str2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
        str2e_vl_loader = DataLoader(str2e_vl_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 
        str2e_te_loader = DataLoader(str2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

        print(f"{self.str_type}2E-Net: "
              f"Training set: {len(str2e_tr_dataset)}, "
              f"Validation set: {len(str2e_vl_dataset)}, "
              f"Test set: {len(str2e_te_dataset)}.")

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

        epochs = self.epochs

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = np.float64('inf')
        log_file = os.path.join(self.root_path, "models", "STR2E_Net_%s_%s.txt"%(self.str_type, suffix))
        model_file = os.path.join(self.root_path, "models", "STR2E_Net_%s_%s.pth"%(self.str_type, suffix))
        with open(log_file, 'w') as file:
            file.close()

        for epoch in range(epochs):
            tr_loss, tr_res = cont2e_train(model, str2e_tr_loader, loss_func, optimizer)
            vl_loss, vl_res = cont2e_evaluate(model, str2e_vl_loader, loss_func)
            te_loss, te_res = cont2e_evaluate(model, str2e_te_loader, loss_func)

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
                torch.save(model, model_file)
                best = vl_res
                print("Saved Model!") if self.verbose else None

        return None

    def train_all_models(self):
        for setting in self.combinations_settings:
            self.train_single_model(setting)
        print("Finish the training of all POS2COHP models.")
        return None

    def get_all_models(self):
        dataset_model_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)

            dataset = STR2E_Dataset(root=self.root_path, 
                                    Element_List=self.Element_List, 
                                    setting=setting,
                                    raw_data_dict=self.raw_data_dict,
                                    str_type=self.str_type)

            model = torch.load(os.path.join(self.root_path, "models","STR2E_Net_%s_%s.pth"%(self.str_type,suffix))).to("cpu")
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
                aes, names = zip(*[
                    (
                        list(np.abs(np.subtract(model(data.to(device)).detach().cpu().numpy(), data.y.to("cpu").numpy()))),
                        ['%s_%s'%(data.slab[idx],data.metal_pair[idx]) for idx in range(len(data.metal_pair))]
                    )
                    for data in data_loader
                ])
            aes = list(itertools.chain(*aes))
            names = list(itertools.chain(*names))

            PRED_DICT = {name: ae for name, ae in zip(names, aes)}

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
            sorted_DIFF = sorted([(k, np.float32(v)) for k,v in PRED_DICT.items()], key=lambda x:x[1], reverse=True)
            top_k_MAE = np.average([x[1] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]])
            print(suffix, ": %.6f"%top_k_MAE)
            model_list.append((dataset, model, setting, PRED_DICT, top_k_MAE, suffix))

        return sorted(model_list, key=lambda x:x[4], reverse=True)

