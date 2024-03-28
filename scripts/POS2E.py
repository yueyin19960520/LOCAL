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


class POS2E():
    def __init__(self, COHP_info_dict, split_ratio=0.8, batch_size=48, dim=256, epochs=300, verbose=True):

        self.COHP_info_dict = COHP_info_dict
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.dim = dim
        self.epochs =  epochs
        self.verbose = verbose

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

    def train_single_model(self, suffix, dataset, setting, PRED):
        print(setting)
        pos2e_dataset = POS2E_Dataset("./", dataset, PRED, self.raw_data_dict, setting, suffix)#.shuffle()
        #pos2e_dataset = POS2E_Dataset_Augmentation(pos2e_dataset, suffix)

        pos2e_tr_dataset = pos2e_dataset[:int(len(pos2e_dataset)*self.split_ratio)].shuffle()
        pos2e_te_dataset = pos2e_dataset[int(len(pos2e_dataset)*self.split_ratio):].shuffle()

        pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  
        pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

        print("Net-by-Net whole dataset:%s, Trainig set:%s, Test set:%s."%(len(pos2e_dataset), len(pos2e_tr_dataset), len(pos2e_te_dataset)))

        node_feats = 41 if setting["Fake_Carbon"] else 40

        #model = CONT2E_Net(dim=self.dim, N_linear=0, N_conv=3, adj_conv=False, in_features=node_feats, bias=False,
                               #conv=pyg_GCNLayer, dropout=0., pool=GraphMultisetTransformer,
                               #pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False)
        model = CONT2E_add_Net(conv_dims=[64, 128, 256], linear_dims=[64, 128, 256], adj_conv=False, in_features=node_feats, bias=True,
                               conv=pyg_GCNLayer, dropout=0.)


        epochs = self.epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=5*10 ** -4, weight_decay=10 ** -4) #-3,-4
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = 1e5
        with open("./models/POS2E_Net_%s.txt"%suffix, 'w') as file:
            file.close()

        for i in range(epochs):
            #tr_temp_dataset = pos2e_tr_dataset.shuffle()[:int(len(pos2e_tr_dataset) * (1/79))]
            #te_temp_dataset = pos2e_te_dataset.shuffle()[:int(len(pos2e_te_dataset) * (1/79))]
            #pos2e_tr_loader = DataLoader(tr_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  
            #pos2e_te_loader = DataLoader(te_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

            tr_loss, tr_res = cont2e_train(model, pos2e_tr_loader, loss_func, optimizer)
            te_loss, te_res = cont2e_evaluate(model, pos2e_te_loader, loss_func)
                    
            scheduler.step()
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
            training_info = "epoch:%s, Training MAE:%.3f(eV), Training Loss:%.6f, Test MAE:%.3f(eV), Test Loss:%.6f, Learning Rate:%s"%(i, tr_res, tr_loss, te_res, te_loss, learning_rate)
            
            if self.verbose:
                print(training_info)
            
            with open("./models/POS2E_Net_%s.txt"%suffix, 'a') as file:
                file.write(training_info + "\n")   

            if learning_rate <= 1e-5 and te_res < best:
                torch.save(model, "./models/POS2E_Net_%s.pth"%suffix)
                best = te_res
                print("Saved Model!") if self.verbose else None
        return pos2e_dataset, model

    def train_all_models(self):
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            pos2e_dataset, model = self.train_single_model(suffix, dataset, setting, PRED)
        print("Finish the training of all POS2E models.")

    def get_all_models(self):
        dataset_model_dict = {}
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            dataset = POS2E_Dataset("./", dataset, PRED, self.raw_data_dict, setting, suffix).shuffle()
            model = torch.load("./models/POS2E_Net_%s.pth"%suffix).to("cpu")
            dataset_model_dict[suffix] = [dataset, model, setting]
        return dataset_model_dict

    def build_bridge_for_KDT(self):
        dataset_model_dict = self.get_all_models()

        for suffix, [dataset, model, setting] in dataset_model_dict.items():
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data:(list(np.abs(np.subtract(model(data).numpy(), data.y.numpy()))), data.slab, data.metal_pair), data_loader))
            DIFF, slabs, metal_pairs = np.hstack(PRED)
            name_list = [x+"_"+y for x,y in zip(slabs, metal_pairs)]
            PRED_DICT = dict(zip(name_list, DIFF))

            dataset_model_dict[suffix].append(PRED_DICT)

        dataset_model_dict_with_PRED_DICT = dataset_model_dict
        return dataset_model_dict_with_PRED_DICT

    def _measure_model(self, sorted_DIFF, top_k):
        MAE = np.average([x[1] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]])
        names_with_MAE = [x[0] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]]
        return MAE, names_with_MAE

    def get_all_models_sorted(self, top_k=1):
        dataset_model_dict_with_PRED_DICT = self.build_bridge_for_KDT()

        model_list = []
        for suffix, [dataset, model, setting, PRED_DICT] in dataset_model_dict_with_PRED_DICT.items():
            sorted_DIFF = sorted([(k,np.float32(v)) for k,v in PRED_DICT.items()], key=lambda x:x[1], reverse=True)
            top_k_MAE, bad_predictions = self._measure_model(sorted_DIFF, top_k)
            print(suffix, ": %.6f"%top_k_MAE)
            model_list.append((dataset, model, setting, PRED_DICT, top_k_MAE, bad_predictions, suffix))

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
    #pos2cohp.train_all_models()

    dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E()
    print(dataset_model_dict_with_PRED.keys())

    pos2e = POS2E(dataset_model_dict_with_PRED, split_ratio=0.9, batch_size=48, dim=256, epochs=300, verbose=True)
    pos2e.build_raw_data_dict() 
    pos2e.train_all_models()

    sorted_model_with_infos = pos2e.get_all_models_sorted(top_k=0.05)

    bad_predictions = sorted_model_with_infos[0][5]

    print(bad_predictions)