import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from nets import *
from functions import *
from dataset import *
from training_utils import *

from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from scipy.spatial import KDTree


class POS2CLS():
    def __init__(self, structure_sets, Element_List, Metals, epochs=50):
        self.structure_sets = structure_sets
        self.Element_List = Element_List
        self.Metals = Metals
        self.epochs = epochs

    def train_POS2CLS_model(self):
        dataset = POS2CLS_Dataset("./", self.structure_sets, self.Element_List, self.Metals).shuffle()

        tr_dataset = dataset[:int(0.8*len(dataset))]
        te_dataset = dataset[int(0.8*len(dataset)):]
        tr_loader = DataLoader(tr_dataset, batch_size=48, shuffle=True, drop_last=True)
        te_loader = DataLoader(te_dataset, batch_size=48, shuffle=True, drop_last=True)

        model = POS2CLS_Net(in_feats=40, hidden_feats=[64,64],activation=None, residual=None, batchnorm=None, dropout=None, 
                    predictor_hidden_feats=64, n_tasks=12, predictor_dropout=0.)
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        epochs = self.epochs
        
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -4, weight_decay=10 ** -4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

        for epoch in range(epochs):
            tr_abs, loss = pos2cls_train(model, tr_loader, loss_func, optimizer)
            tr_abs = "%.2f"%tr_abs

            te_abs = pos2cls_evaluate(model, te_loader)
            te_abs = "%.2f"%te_abs
            
            print("Epoch:%s, Train_Rate:%s, Test_Rate:%s."%(epoch,tr_abs,te_abs))
        torch.save(model, "./models/POS2CLS.pth")

    def get_all_embeddings_and_build_KDTree(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = torch.load("./models/POS2CLS.pth").to(device)
        print("POS2CLS model loaded!")
        
        ori_pos2cls_dataset = POS2CLS_Dataset("./", self.structure_sets, self.Element_List, self.Metals).shuffle()
        print("POS2CLS dataset used for training is ready.")
        all_pos2cls_dataset = POS2CLS_Dataset_ALL("./", self.Element_List, self.Metals).shuffle()
        print("POS2CLS_ALL dataset saved!")
        spl_pos2cls_dataset = all_pos2cls_dataset[:int(0.02*len(all_pos2cls_dataset))]

        ori_loader = DataLoader(ori_pos2cls_dataset, batch_size=48, shuffle=True, drop_last=False)
        all_loader = DataLoader(all_pos2cls_dataset, batch_size=48, shuffle=True, drop_last=False)
        spl_loader = DataLoader(spl_pos2cls_dataset, batch_size=48, shuffle=True, drop_last=False)

        ori_embeddings, ori_classes, ori_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                                        data.to("cpu").cls, 
                                                        data.to("cpu").name) for data in ori_loader])
        # print('ori_embeddings',len(ori_embeddings,ori_embeddings[0].shape)
        all_embeddings, all_classes, all_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                                        data.to("cpu").cls, 
                                                        data.to("cpu").name) for data in all_loader])

        self.spl_embeddings, self.spl_classes, self.spl_names = zip(*[(model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                                                                       data.to("cpu").cls, 
                                                                       data.to("cpu").name) for data in spl_loader])
        
        tree_embeddings = np.vstack((np.vstack(ori_embeddings),np.vstack(all_embeddings)))
        tree_names = np.hstack((np.hstack(ori_names),np.hstack(all_names)))
        
        self.name_emb_dict = {name: embedding for name, embedding in zip(tree_names, tree_embeddings)}
        self.names = list(self.name_emb_dict.keys())
        self.embeddings = list(self.name_emb_dict.values())
        self.kd_tree = KDTree(self.embeddings)
        
        file_save = open(os.path.join("models", "embedding_name_all.pkl"),'wb') 
        pickle.dump(self.name_emb_dict, file_save) 
        file_save.close()
        print("All structure embeddings saved!")

    def find_k_nearest_neighbors(self, target_point_key, k=4):
        try:
            target_embedding = self.name_emb_dict[target_point_key]
        except:
            target_embedding = self.name_emb_dict[self.rvs(target_point_key)]
        distances, indices = self.kd_tree.query(target_embedding, k=k+1)
        nearest_neighbors = dict([(self.names[i], self.embeddings[i]) for i in indices])
        return list(nearest_neighbors.keys())[1:]

    def rvs(self, k): 
        return "_".join(k.split("_")[:2] + [k.split("_")[3], k.split("_")[2]])