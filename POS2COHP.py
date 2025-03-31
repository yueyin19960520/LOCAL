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
from datetime import datetime
from Device import device
from torch.utils.data import Subset

class POS2COHP():
    def __init__(self, Element_List, setting_dict, splitted_keys=None,
                       split_ratio=0.8, batch_size=48, 
                       hidden_feats=[256,256,256,256], predictor_hidden_feats=128, 
                       epochs=300, verbose=True):

        self.Element_List = Element_List
        self.splitted_keys = splitted_keys

        possibles = list(product(*list(setting_dict.values())))
        all_possible_settings = list(map(lambda p:dict(zip(list(setting_dict.keys()),p)), possibles))

        self.combinations_settings = all_possible_settings
        
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.hidden_feats = hidden_feats
        self.predictor_hidden_feats = predictor_hidden_feats
        self.epochs =  epochs
        self.verbose = verbose
    
    def build_raw_data_dict(self):   
        if not os.path.exists(os.path.join("raw", "icohp_structures_all.pkl")):
            build_raw_COHP_file("./", "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
        return None

    def train_single_model(self, setting):
        current_time = datetime.now()
        current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print(setting)
        if setting["encode"] == "physical":
            node_feats = 22
        elif setting["encode"] == "onehot":
            node_feats = 41 if setting["Fake_Carbon"] else 40
        else:
            node_feats = 63 if setting["Fake_Carbon"] else 62
        n_tasks = 2 if setting["Binary_COHP"] else 1

        suffix = setting2suffix(setting)
        print(suffix)

        key_list = [i for j in self.splitted_keys.values() for i in j]
        pos2cohp_dataset = POS2COHP_Dataset("./", self.Element_List, setting, icohp_list_keys=key_list)
        
        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        tr_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["train"]]
        vl_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["test"]]

        pos2cohp_tr_dataset = Subset(pos2cohp_dataset, tr_indices)
        pos2cohp_vl_dataset = Subset(pos2cohp_dataset, vl_indices)
        pos2cohp_te_dataset = Subset(pos2cohp_dataset, te_indices)

        pos2cohp_tr_loader = DataLoader(pos2cohp_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        pos2cohp_vl_loader = DataLoader(pos2cohp_vl_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        pos2cohp_te_loader = DataLoader(pos2cohp_te_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        print("COHP whole dataset:%s, Trainig set:%s, Valid set:%s, Test set:%s."%(len(pos2cohp_dataset), 
                                                                                   len(pos2cohp_tr_dataset), 
                                                                                   len(pos2cohp_vl_dataset), 
                                                                                   len(pos2cohp_te_dataset)))

        if setting["Hetero_Graph"]:
            model = POS2COHP_Net_Hetero(atom_feats=node_feats, 
                                        bond_feats=node_feats, 
                                        hidden_feats=self.hidden_feats, 
                                        activation=None, 
                                        residual=None, 
                                        batchnorm=None, 
                                        dropout=None,
                                        predictor_hidden_feats=self.predictor_hidden_feats, 
                                        n_tasks=n_tasks,
                                        predictor_dropout=0.)
        else:
            model = POS2COHP_Net(in_feats=node_feats, 
                                 hidden_feats=self.hidden_feats, 
                                 activation=None, 
                                 residual=None, 
                                 batchnorm=None, 
                                 dropout=None, 
                                 predictor_hidden_feats=self.predictor_hidden_feats, 
                                 n_tasks=n_tasks, 
                                 predictor_dropout=0.)
        epochs = self.epochs

        optimizer = torch.optim.Adam(params=model.parameters(),  lr=10 ** -3, weight_decay=10 ** -4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = np.float64('-inf') if setting["Binary_COHP"] else np.float64('inf')
        log_file = "./models/POS2COHP_Net_%s.txt"%(suffix)
        with open(log_file, 'w') as file:
            file.close()

        for epoch in range(epochs):
            tr_loss, tr_res = pos2cohp_train(model, pos2cohp_tr_loader, setting, optimizer)
            vl_loss, vl_res = pos2cohp_evaluate(model, pos2cohp_vl_loader, setting)
            te_loss, te_res = pos2cohp_evaluate(model, pos2cohp_te_loader, setting)
                
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)

            if setting["Binary_COHP"]:
                training_info = (f"epoch: {epoch},"
                                 f"Training ABS:{tr_res:.2f}%, "
                                 f"Validation ABS:{vl_res:.2f}%, "
                                 f"Test ABS:{te_res:.2f}%, "
                                 f"Learning Rate:{learning_rate}")
            else:
                training_info = (f"epoch: {epoch},"
                                 f"Training MAE:{tr_res:.3f}, "
                                 f"Validation MAE:{vl_res:.3f}, "
                                 f"Test MAE:{te_res:.3f}, "
                                 f"Learning Rate:{learning_rate}")
            scheduler.step()

            if self.verbose: 
                print(training_info)

            with open(log_file, 'a') as file:
                file.write(training_info + "\n")

            if (setting["Binary_COHP"] and vl_res > best and learning_rate <=1e-5) or (not setting["Binary_COHP"] and vl_res < best and learning_rate <=1e-5):
                torch.save(model, "./models/POS2COHP_Net_%s.pth"%(suffix))
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

            key_list = [i for j in self.splitted_keys.values() for i in j]
            dataset = POS2COHP_Dataset("./", self.Element_List, setting, icohp_list_keys=key_list)
            model = torch.load("./models/POS2COHP_Net_%s.pth"%suffix).to("cpu")
            dataset_model_dict[suffix] = [dataset, model, setting]

        """
        len_dataset = len(dataset)
        temp_ratio = [int(len_dataset*self.split_ratio[0]), int(len_dataset*(self.split_ratio[0]+self.split_ratio[1])), len_dataset]
        smp = lambda data:data.slab + "_" + data.metal_pair
        self.splitted_keys = {"train": list(map(smp, dataset[:temp_ratio[0]])),
                              "valid": list(map(smp, dataset[temp_ratio[0]:temp_ratio[1]])),
                              "test": list(map(smp, dataset[temp_ratio[1]:]))}
        """
        return dataset_model_dict

    def build_bridge_for_E(self, dataset_model_dict=None):
        if dataset_model_dict == None:
            dataset_model_dict = self.get_all_models()

        for suffix, [dataset, model, setting] in dataset_model_dict.items():
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data:split_batch_data(data, model(data)), data_loader))
            PRED = [i for j in PRED for i in j]

            dataset_model_dict[suffix].append(PRED)

        dataset_model_dict_with_PRED = dataset_model_dict
        return dataset_model_dict_with_PRED

    def _measure_model(self, dataset, pred_value, setting):
        valid = []
        for true, pred in zip(dataset, pred_value):
            true = true.MN_icohp.numpy()
            if setting["Binary_COHP"]:
                true = [True if x[0] > x[1] else False for x in true]   # [1,0] = True
                pred = [True if x[0] > x[1]  else False for x in pred]
            else:
                true = [True if x <= setting["threshold"] else False for x in true]
                pred = [True if x <= setting["threshold"] else False for x in pred]
            valid += [True if x==y else False for x,y in zip(true,pred)]
        success_ratio = valid.count(True)/(valid.count(True)+valid.count(False))
        return success_ratio

    def get_all_models_sorted(self, dataset_model_dict_with_PRED=None):
        if dataset_model_dict_with_PRED == None:
            dataset_model_dict_with_PRED = self.build_bridge_for_E()

        model_list = []
        for suffix, [dataset, model, setting, PRED] in dataset_model_dict_with_PRED.items():
            success_ratio = self._measure_model(dataset, PRED, setting)
            print(suffix, ": %.2f"%(success_ratio*100))
            model_list.append((dataset, model, setting, PRED, success_ratio, suffix))
 
        return sorted(model_list, key=lambda x:x[4], reverse=True)





if __name__ == "__main__":
    Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
               "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
               "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
               "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
    Slabs = ["C","N"]
    Element_List = Metals + Slabs

    clean = True
    list_timestamp = "20240726120913"

    icohp_list_keys = restart(clean=clean)
    if not clean:
        with open('saved_lists/list_%s.pkl'%list_timestamp, 'rb') as file:
            icohp_list_keys = pickle.load(file)

    settings = {'Fake_Carbon': [True],
                'Binary_COHP': [True,False], 
                'Hetero_Graph': [False], 
                'threshold': [-0.6],
                'encode': ['onehot','physical']}


    ### Calculation Part ###
    pos2cohp = POS2COHP(Element_List, setting_dict=settings, icohp_list_keys=icohp_list_keys,
                        split_ratio=[0.80, 0.10, 0.10], batch_size=64, hidden_feats=[256,256,256,256], 
                        predictor_hidden_feats=128, epochs=10, verbose=True)

    pos2cohp.build_raw_data_dict()

    pos2cohp.train_all_models()

    dataset_model_dict = pos2cohp.get_all_models()

    dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E(dataset_model_dict)

    res = pos2cohp.get_all_models_sorted(dataset_model_dict_with_PRED)