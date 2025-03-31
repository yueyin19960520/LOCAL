from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import datetime
from Device import device
import itertools

from nets import *
from functions import *
from dataset import *
from training_utils import *



class POS2COHP():
    def __init__(self, Element_List=None, setting_dict=None, splitted_keys=None, config=None, loop=0):

        self.Element_List = Element_List
        self.splitted_keys = splitted_keys

        possibles = list(product(*list(setting_dict.values())))
        all_possible_settings = list(map(lambda p:dict(zip(list(setting_dict.keys()),p)), possibles))

        self.combinations_settings = all_possible_settings

        self.hidden_feats=config["hidden_feats"]
        self.predictor_hidden_feats=config["predictor_hidden_feats"]
        self.activation=configs_str_mapping[config["activation"]]

        self.batch_size=config["batch_size"]
        self.epochs=config["epochs"]
        self.learning_rate=config["learning_rate"]
        self.weight_decay=config["weight_decay"]

        self.verbose=config["verbose"]

        self.loop = loop
        self.root_path = os.path.dirname(os.path.dirname(__file__))


    def train_single_model(self, setting, finetune, mix_ratio):
        current_time = datetime.datetime.now()
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
        pos2cohp_dataset = POS2COHP_Dataset(self.root_path, self.Element_List, setting)
        
        temp1 = lambda data:"_".join((data.slab, data.metal_pair))
        tr_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["train"]]
        vl_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["valid"]]
        te_indices = [i for i,d in enumerate(pos2cohp_dataset) if temp1(d) in self.splitted_keys["test"]]

        pos2cohp_tr_dataset = Subset(pos2cohp_dataset, tr_indices)

        if finetune:
            loop_dataset = POS2COHP_Dataset(self.root_path, self.Element_List, setting, self.loop)
            tr_indices = random.sample(tr_indices, min(len(tr_indices),int(mix_ratio * len(loop_dataset))))
            pos2cohp_tr_dataset = Subset(pos2cohp_dataset, tr_indices)
            #self.loop_drop_names = list(set(self.splitted_keys["train"]).difference(set([temp1(d) for d in pos2cohp_tr_dataset])))
            pos2cohp_tr_dataset = CombinedDataset([pos2cohp_tr_dataset, loop_dataset])
        else:
            loop_dataset = []
            # self.loop_drop_names = []

        pos2cohp_vl_dataset = Subset(pos2cohp_dataset, vl_indices)
        pos2cohp_te_dataset = Subset(pos2cohp_dataset, te_indices)

        pos2cohp_tr_loader = DataLoader(pos2cohp_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
        pos2cohp_vl_loader = DataLoader(pos2cohp_vl_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn)
        pos2cohp_te_loader = DataLoader(pos2cohp_te_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn)

        print(f"POS2COHP whole dataset: {len(pos2cohp_dataset) + len(loop_dataset)}, "
              f"Training set: {len(pos2cohp_tr_dataset)}, "
              f"Validation set: {len(pos2cohp_vl_dataset)}, "
              f"Test set: {len(pos2cohp_te_dataset)}.")

        if setting["Hetero_Graph"]:
            model = POS2COHP_Net_Hetero(atom_feats=node_feats, 
                                        bond_feats=node_feats, 
                                        hidden_feats=self.hidden_feats, 
                                        activation=self.activation, 
                                        predictor_hidden_feats=self.predictor_hidden_feats, 
                                        n_tasks=n_tasks,
                                        predictor_dropout=0.)
        else:
            model = POS2COHP_Net(in_feats=node_feats, 
                                 hidden_feats=self.hidden_feats, 
                                 activation=self.activation, 
                                 predictor_hidden_feats=self.predictor_hidden_feats, 
                                 n_tasks=n_tasks, 
                                 predictor_dropout=0.)

        if finetune:
            model.load_state_dict(torch.load(os.path.join(self.root_path, "models", "POS2COHP_Net_%s_loop%s.pth"%(suffix, 0)))) #self.loop-1
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate*0.5, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        epochs = self.epochs
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = np.float64('-inf') if setting["Binary_COHP"] else np.float64('inf')

        if finetune:
            log_file = os.path.join(self.root_path, "models", "POS2COHP_Net_%s_loop%s.txt"%(suffix, self.loop))
            model_file = os.path.join(self.root_path, "models", "POS2COHP_Net_%s_loop%s.pth"%(suffix, self.loop))
        else:
            log_file = os.path.join(self.root_path, "models", "POS2COHP_Net_%s.txt"%(suffix))
            model_file = os.path.join(self.root_path, "models", "POS2COHP_Net_%s.pth"%(suffix))

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

            if (setting["Binary_COHP"] and vl_res > best and learning_rate <=1e-4) or (not setting["Binary_COHP"] and vl_res < best and learning_rate <=1e-4):
                torch.save(model.state_dict(),model_file)
                best = vl_res
                print("Saved Model!") if self.verbose else None

        return None

    def train_all_models(self, finetune=False, mix_ratio=0):
        for setting in self.combinations_settings:
            self.train_single_model(setting, finetune, mix_ratio)
        print("Finish the training of all POS2COHP models.")
        return None

    def get_all_models(self, finetune):
        dataset_model_dict = {}
        for setting in self.combinations_settings:
            suffix = setting2suffix(setting)

            key_list = [i for j in self.splitted_keys.values() for i in j]
            dataset = CombinedDataset([POS2COHP_Dataset(self.root_path, self.Element_List, setting, loop=l) for l in [0, self.loop]])
            #dataset = FilteredDataset(dataset, self.loop_drop_names)
            print("Filtered dataset size is %s."%len(dataset))

            if setting["encode"] == "physical":
                node_feats = 22
            elif setting["encode"] == "onehot":
                node_feats = 41 if setting["Fake_Carbon"] else 40
            else:
                node_feats = 63 if setting["Fake_Carbon"] else 62
            n_tasks = 2 if setting["Binary_COHP"] else 1

            if setting["Hetero_Graph"]:
                model = POS2COHP_Net_Hetero(atom_feats=node_feats, 
                                            bond_feats=node_feats, 
                                            hidden_feats=self.hidden_feats, 
                                            activation=self.activation, 
                                            predictor_hidden_feats=self.predictor_hidden_feats, 
                                            n_tasks=n_tasks,
                                            predictor_dropout=0.)
            else:
                model = POS2COHP_Net(in_feats=node_feats, 
                                     hidden_feats=self.hidden_feats, 
                                     activation=self.activation, 
                                     predictor_hidden_feats=self.predictor_hidden_feats, 
                                     n_tasks=n_tasks, 
                                     predictor_dropout=0.)      
            if finetune:
                model_file = os.path.join(self.root_path, "models", "POS2COHP_Net_%s_loop%s.pth"%(suffix, self.loop))
                model.load_state_dict(torch.load(model_file))
            else:
                model.load_state_dict(torch.load(os.path.join(self.root_path, "models", "POS2COHP_Net_%s.pth"%suffix)))  
            dataset_model_dict[suffix] = [dataset, model, setting]

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
