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
from Device import device


class CONT2E_without_COHP:
    def __init__(self, Element_List, setting_dict, 
                       split_ratio=0.8, batch_size=48, dim=256,
                       conv_dim_list = [],  linear_dim_list = [], 
                       epochs=300, verbose=True):

        self.Element_List = Element_List
        # keys = [key for key in setting_dict if key != "threshold"]
        # self.combinations_settings = list(map(lambda combo: dict(zip(keys, combo), **{'threshold': setting_dict['threshold']}),
        #                                  itertools.product([True, False], repeat=len(keys))))
        self.combinations_settings = [{'Fake_Carbon': True, 'Binary_COHP': True, 'Hetero_Graph': False, 'threshold': -0.6}]
        
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        
        
        self.epochs =  epochs
        self.verbose = verbose
        self.dim = dim
        self.linear_dim_list = linear_dim_list
        self.conv_dim_list = conv_dim_list
    
    # def build_raw_data_dict(self):   
    #     if not os.path.exists(os.path.join("raw", "icohp_structures_all.pkl")):
    #         build_raw_COHP_file("./", "structures_all")
    #         print("Raw data dict prepare done!")
    #     else:
    #         print("Raw data dict already exist!")
    #     return None

    def train_single_model(self, setting):
        print(setting)
        suffix = "%s_%s_%s"%("FcN" if setting["Fake_Carbon"] else "CN", 
                             "BC" if setting["Binary_COHP"] else "RG",
                             "Hetero" if setting["Hetero_Graph"] else "Homo")

        # node_feats = 41 if setting["Fake_Carbon"] else 40
        # n_tasks = 2 if setting["Binary_COHP"] else 1
        #####################################################################################################################
        cont2e_dataset = CONT2EwithoutCOHP_Dataset("./", Element_List=self.Element_List, setting=setting).shuffle()

        cont2e_tr_dataset = cont2e_dataset[:int(self.split_ratio*len(cont2e_dataset))]
        cont2e_te_dataset = cont2e_dataset[int(self.split_ratio*len(cont2e_dataset)):]

        cont2e_tr_loader = DataLoader(cont2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        cont2e_te_loader = DataLoader(cont2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        print("CONT2EwithoutCOHP whole dataset:%s, Trainig set:%s, Test set:%s."%(len(cont2e_dataset), len(cont2e_tr_dataset), len(cont2e_te_dataset)))

        node_feats = len(self.Element_List)+1 if setting["Fake_Carbon"] else len(self.Element_List)
        # node_feats = 40

        model = CONT2E_Net(dim=self.dim, linear_dim_list=self.linear_dim_list,
                           conv_dim_list=self.conv_dim_list,
                            adj_conv=False, in_features=node_feats, bias=False,
                               conv=pyg_GCNLayer, dropout=0., pool=GraphMultisetTransformer,
                               pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False,
                               pool_type='all')
        print(model)
        epochs = self.epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4) #-3,-4
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = 1e5
        with open("./models/CONT2EwithoutCOHP_Net_%s.txt"%suffix, 'w') as file:
            file.close()

        for i in range(epochs):
            #tr_temp_dataset = pos2e_tr_dataset.shuffle()[:int(len(pos2e_tr_dataset) * (1/79))]
            #te_temp_dataset = pos2e_te_dataset.shuffle()[:int(len(pos2e_te_dataset) * (1/79))]
            #pos2e_tr_loader = DataLoader(tr_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  
            #pos2e_te_loader = DataLoader(te_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

            tr_loss, tr_res = cont2e_train(model, cont2e_tr_loader, loss_func, optimizer)
            te_loss, te_res = cont2e_evaluate(model, cont2e_te_loader, loss_func)
                    
            scheduler.step()
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
            training_info = "epoch:%s, Training MAE:%.3f(eV), Training Loss:%.6f, Test MAE:%.3f(eV), Test Loss:%.6f, Learning Rate:%s"%(i, tr_res, tr_loss, te_res, te_loss, learning_rate)
            
            if self.verbose:
                print(training_info)
            
            with open("./models/CONT2EwithoutCOHP_Net_%s.txt"%suffix, 'a') as file:
                file.write(training_info + "\n")   

            if learning_rate <= 1e-5 and te_res < best:
                torch.save(model, "./models/CONT2EwithoutCOHP_Net_%s.pth"%suffix)
                best = te_res
                print("Saved Model!") if self.verbose else None
        return cont2e_dataset, model

    def train_all_models(self):
        for setting in self.combinations_settings:
            pos2cohp_dataset, model = self.train_single_model(setting)
        print("Finish the training of CONT2EwithoutCOHP models.")

    # def get_all_models(self):
    #     dataset_model_dict = {}
    #     for setting in self.combinations_settings:

    #         suffix = "%s_%s_%s"%("FcN" if setting["Fake_Carbon"] else "CN", 
    #                              "BC" if setting["Binary_COHP"] else "RG",
    #                              "Hetero" if setting["Hetero_Graph"] else "Homo")

    #         dataset = POS2COHP_Dataset("./", self.Element_List, setting, suffix).shuffle()
    #         model = torch.load("./models/POS2COHP_Net_%s.pth"%suffix).to("cpu")
    #         dataset_model_dict[suffix] = [dataset, model, setting]
    #         #print("Find the model of %s."%suffix)
    #     return dataset_model_dict

    # def build_bridge_for_E(self):
    #     #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     dataset_model_dict = self.get_all_models()

    #     for suffix, [dataset, model, setting] in dataset_model_dict.items():
    #         data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    #         model.eval()
    #         with torch.no_grad():
    #             PRED = list(map(lambda data:split_batch_data(data, model(data)), data_loader))
    #         PRED = [i for j in PRED for i in j]

    #         dataset_model_dict[suffix].append(PRED)

    #     dataset_model_dict_with_PRED = dataset_model_dict
    #     return dataset_model_dict_with_PRED

    # def _measure_model(self, dataset, pred_value, setting):
    #     valid = []
    #     for true, pred in zip(dataset, pred_value):
    #         true = true.MN_icohp.numpy()
    #         if setting["Binary_COHP"]:
    #             true = [True if x[0] > x[1] else False for x in true]   # [1,0] = True
    #             pred = [True if x[0] > x[1]  else False for x in pred]
    #         else:
    #             true = [True if x <= setting["threshold"] else False for x in true]
    #             pred = [True if x <= setting["threshold"] else False for x in pred]
    #         valid += [True if x==y else False for x,y in zip(true,pred)]
    #     success_ratio = valid.count(True)/(valid.count(True)+valid.count(False))
    #     return success_ratio

    # def get_all_models_sorted(self):
    #     dataset_model_dict_with_PRED = self.build_bridge_for_E()

    #     model_list = []
    #     for suffix, [dataset, model, setting, PRED] in dataset_model_dict_with_PRED.items():
    #         success_ratio = self._measure_model(dataset, PRED, setting)
    #         print(suffix, ": %.2f"%(success_ratio*100))
    #         model_list.append((dataset, model, setting, PRED, success_ratio, suffix))
 
    #     return sorted(model_list, key=lambda x:x[4], reverse=True)


