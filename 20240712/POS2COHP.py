import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from nets import *
from functions import *
from dataset import *
from training_utils import *
import itertools#

from torch_geometric.loader import DataLoader
from datetime import datetime
from Device import device

class POS2COHP():
    def __init__(self, Element_List, setting_dict, 
                       split_ratio=0.8, batch_size=48, 
                       hidden_feats=[256,256,256,256], predictor_hidden_feats=128, 
                       epochs=300, verbose=True,part_MN=None):

        self.Element_List = Element_List
        # keys = [key for key in setting_dict if key != "threshold"]
        # self.combinations_settings = list(map(lambda combo: dict(zip(keys, combo), **{'threshold': setting_dict['threshold']}),
        #                                  itertools.product([True, False], repeat=len(keys))))
        self.combinations_settings = [{'Fake_Carbon': True, 'Binary_COHP': False, 'Hetero_Graph': False, 'threshold': -0.6}]
        
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.hidden_feats = hidden_feats
        self.predictor_hidden_feats = predictor_hidden_feats
        self.epochs =  epochs
        self.verbose = verbose
        self.part_MN = part_MN
    
    def build_raw_data_dict(self):   
        if not os.path.exists(os.path.join("raw", "icohp_structures_all.pkl")):
            build_raw_COHP_file("./", "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
        return None

    def train_single_model(self, setting,enc_type='onehot',icohp_list_keys=None):

        # 获取当前时间
        current_time = datetime.now()
        # 格式化时间，生成文件名
        file_name_current_time = current_time.strftime("%Y%m%d_%H%M%S")

        print(setting)
        suffix = "%s_%s_%s"%("FcN" if setting["Fake_Carbon"] else "CN", 
                             "BC" if setting["Binary_COHP"] else "RG",
                             "Hetero" if setting["Hetero_Graph"] else "Homo")

        node_feats = 41 if setting["Fake_Carbon"] else 40
        n_tasks = 2 if setting["Binary_COHP"] else 1

        if enc_type == 'onehot':
            pos2cohp_dataset = POS2COHP_Dataset("./", self.Element_List, setting, suffix,part_MN=self.part_MN,icohp_list_keys=icohp_list_keys)#.shuffle() shuffle bas been done in main()
        elif enc_type == 'only_physical':
            pos2cohp_dataset = POS2COHP_onehot_and_physical_Dataset("./", self.Element_List,
                                                                     setting, suffix, mode=enc_type)#.shuffle()
        else:
            pos2cohp_dataset = POS2COHP_onehot_and_physical_Dataset("./", self.Element_List,
                                                                     setting, suffix, mode=enc_type)#.shuffle()
        
        
        pos2cohp_train_dataset = pos2cohp_dataset[:int(self.split_ratio[0]*len(pos2cohp_dataset))]
        pos2cohp_valid_dataset = pos2cohp_dataset[int(self.split_ratio[0]*len(pos2cohp_dataset)):int((self.split_ratio[0]+self.split_ratio[1])*len(pos2cohp_dataset))]
        pos2cohp_test_dataset = pos2cohp_dataset[int((self.split_ratio[0]+self.split_ratio[1])*len(pos2cohp_dataset)):]

        pos2cohp_train_loader = DataLoader(pos2cohp_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        pos2cohp_valid_loader = DataLoader(pos2cohp_valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        pos2cohp_test_loader = DataLoader(pos2cohp_test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        print("COHP whole dataset:%s, Trainig set:%s, Valid set:%s, Test set:%s."%(len(pos2cohp_dataset), len(pos2cohp_train_dataset), len(pos2cohp_valid_dataset), len(pos2cohp_test_dataset)))

        if setting["Hetero_Graph"]:
            model = POS2COHP_Net_Hetero(atom_feats=node_feats, bond_feats=node_feats, hidden_feats=self.hidden_feats, 
                                            activation=None, residual=None, batchnorm=None, dropout=None,
                                            predictor_hidden_feats=self.predictor_hidden_feats, n_tasks=n_tasks,predictor_dropout=0.)
        else:
            model = POS2COHP_Net(in_feats=node_feats, hidden_feats=self.hidden_feats, activation=None, residual=None, batchnorm=None, dropout=None, 
                                     predictor_hidden_feats=self.predictor_hidden_feats, n_tasks=n_tasks, predictor_dropout=0.)
        epochs = self.epochs

        optimizer = torch.optim.Adam(params=model.parameters(),  lr=10 ** -3, weight_decay=10 ** -4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = np.float64('-inf') if setting["Binary_COHP"] else np.float64('inf')
        log_file = "./models/POS2COHP_Net_%s_%s.txt"%(suffix,file_name_current_time)
        with open(log_file, 'w') as file:
            file.close()

        for epoch in range(epochs):
            train_loss, train_res = pos2cohp_train(model, pos2cohp_train_loader, setting, optimizer)
            valid_loss, valid_res = pos2cohp_evaluate(model, pos2cohp_valid_loader, setting)
            test_loss, test_res = pos2cohp_evaluate(model, pos2cohp_test_loader, setting)
                
            scheduler.step()
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)

            if setting["Binary_COHP"]:
                training_info = "epoch:%s, Train ABS:%.2f%%, Train Loss:%.6f, Test ABS:%.2f%%, Test Loss:%.6f, Learning Rate:%.9f"%(epoch, tr_res, tr_loss, te_res, te_loss, learning_rate)
            else:
                # training_info = "epoch:%s, Train MAE:%.3f(eV), Train Loss:%.6f, Test MAE:%.3f(eV), Test Loss:%.6f, Learning Rate:%.9f"%(epoch, tr_res, tr_loss, te_res, te_loss, learning_rate)
                training_info = "epoch:%s, Tr MAE:%.3f(eV), Va MAE:%.3f(eV), Te MAE:%.3f(eV), Learning Rate:%.9f"%(epoch, train_res, valid_res, test_res, learning_rate)

            if self.verbose:
                print(training_info)

            with open(log_file, 'a') as file:
                file.write(training_info + "\n")
                
            #if learning_rate <= 1e-5 and te_res < best:
            if (setting["Binary_COHP"] and valid_res > best and learning_rate <=1e-5) or (not setting["Binary_COHP"] and valid_res < best and learning_rate <=1e-5):
                torch.save(model, "./models/POS2COHP_Net_%s.pth"%(suffix))
                best = valid_res
                print("Saved Model!") if self.verbose else None

        return pos2cohp_dataset, model

    def train_all_models(self,enc_type,icohp_list_keys):# only 1 setting
        for setting in self.combinations_settings:
            pos2cohp_dataset, model = self.train_single_model(setting, enc_type=enc_type,icohp_list_keys=icohp_list_keys)
        print("Finish the training of all POS2COHP models.")
        return pos2cohp_dataset

    def get_all_models(self,dataset):
        dataset_model_dict = {}
        for setting in self.combinations_settings:

            suffix = "%s_%s_%s"%("FcN" if setting["Fake_Carbon"] else "CN", 
                                 "BC" if setting["Binary_COHP"] else "RG",
                                 "Hetero" if setting["Hetero_Graph"] else "Homo")

            # dataset = POS2COHP_Dataset("./", self.Element_List, setting, suffix).shuffle()
            model = torch.load("./models/POS2COHP_Net_%s.pth"%suffix).to("cpu")
            dataset_model_dict[suffix] = [dataset, model, setting]
            #print("Find the model of %s."%suffix)
        return dataset_model_dict

    def build_bridge_for_E(self,dataset):
        #device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dataset_model_dict = self.get_all_models(dataset)

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

    def get_all_models_sorted(self):
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

    setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}

    pos2cohp = POS2COHP(Element_List, setting_dict,
                        split_ratio=0.9, batch_size=48, hidden_feats=[256,256,256,256], 
                        predictor_hidden_feats=128, epochs=300, verbose=True)
    pos2cohp.build_raw_data_dict()
    pos2cohp.train_all_models()

    dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E()
    print(dataset_model_dict_with_PRED.keys())