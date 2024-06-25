import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from Device import device

class measure_matrix(object):
    def __init__(self):
        self.y_pred = []
        self.y_true = []
        
    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def MAE(self):
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_diff = np.subtract(y_pred, y_true)
        Mean_Absolute_Error = np.mean(np.absolute(y_diff))
        return Mean_Absolute_Error

    def absolute_correct_rate(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        
        score = 0
        for i in range(y_pred.shape[0]):
            t = np.where(max(y_true[i]) == y_true[i], 1, 0)
            p = np.where(max(y_pred[i]) == y_pred[i], 1, 0)
            if (t==p).all():
                score += 1
        return score/y_pred.shape[0]
    


def cont2e_train(model, data_loader, loss_func, optimizer):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.train()
    train_matrix = measure_matrix()
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        labels = batch_data.y.to(device)
        model = model.to(device)
        
        outputs = model(batch_data)
        loss = loss_func(outputs, labels) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(outputs, labels)
    res = train_matrix.MAE()
    return loss, res


def cont2e_evaluate(model, data_loader, loss_func):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    eval_matrix = measure_matrix()
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        labels = batch_data.y.to(device)
        model = model.to(device)
        
        outputs = model(batch_data)
        loss = loss_func(outputs, labels) 
        
        torch.cuda.empty_cache()

        eval_matrix.update(outputs, labels)
    res = eval_matrix.MAE()
    return loss, res

def custom_loss(predicted_feats, target, MN_edge_index):
    # Basic loss (e.g., MSE loss)
    loss = F.mse_loss(predicted_feats, target)

    # Ensure that the prediction for adjacent edges are the same
    paired_predicted_feats = predicted_feats.view(-1, 2)
    consistency_loss = F.mse_loss(paired_predicted_feats[:, 0], paired_predicted_feats[:, 1])

    return loss + consistency_loss
    
def pos2cohp_train(model, data_loader, setting, optimizer):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean') if setting["Binary_COHP"] else torch.nn.MSELoss()
    
    model.train()
    train_matrix = measure_matrix()
    for data in data_loader:
        model = model.to(device)
        data = data.to(device)
        pred = model(data)
        #pred = model(data.x.to(device), data.edge_index.to(device), data.MN_edge_index)
    
        target = data.MN_icohp.to(device)

        # loss = custom_loss(pred, target, data.MN_edge_index) # 增加惩罚
        
        loss = loss_func(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(pred, target)
        
    res = train_matrix.absolute_correct_rate()*100 if setting["Binary_COHP"] else train_matrix.MAE()
    return loss, res


def pos2cohp_evaluate(model, data_loader, setting):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean') if setting["Binary_COHP"] else torch.nn.MSELoss()
    
    model.eval()
    eval_matrix = measure_matrix()
    for data in data_loader:
        model = model.to(device)
        data = data.to(device)
        pred = model(data)
        #pred = model(data.x.to(device), data.edge_index.to(device), data.MN_edge_index)
    
        target = data.MN_icohp.to(device)
        loss = loss_func(pred, target)
        torch.cuda.empty_cache()

        eval_matrix.update(pred, target)
        
    res = eval_matrix.absolute_correct_rate()*100 if setting["Binary_COHP"] else eval_matrix.MAE()
    return loss, res


def pos2cls_train(model, data_loader, loss_func, optimizer):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.train()
    train_matrix = measure_matrix()

    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        labels = batch_data.y.to(device)
        model = model.to(device)

        outputs = model(batch_data)

        loss = loss_func(outputs, labels).mean()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        train_matrix.update(outputs, labels)

    abs_score = train_matrix.absolute_correct_rate() *  100

    return abs_score, loss



def pos2cls_evaluate(model, data_loader):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    eval_matrix = measure_matrix()

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            labels = batch_data.y.to(device)
            model = model.to(device)

            outputs = model(batch_data)
                
            torch.cuda.empty_cache()
            eval_matrix.update(outputs, labels)
    abs_score = eval_matrix.absolute_correct_rate() *  100
    
    return abs_score




















def cont2e_cal(model, data_loader):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    eval_matrix = measure_matrix()
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        labels = batch_data.y.to(device)
        model = model.to(device)
        
        outputs = model(batch_data)
        torch.cuda.empty_cache()

        eval_matrix.update(outputs, labels)

    y_pred = torch.cat(eval_matrix.y_pred, dim=0).numpy()
    y_true = torch.cat(eval_matrix.y_true, dim=0).numpy()
    np_absolute = np.absolute(np.subtract(y_pred, y_true))
    return np_absolute

def plot(l,xl,yl,title):#画一个列表的密度分布
    plt.figure(figsize=(10,5))
    accuracy = 30
    min_=min(l)
    max_=max(l)
    y = []
    data_range = np.linspace(min_, max_, accuracy)
    for i in data_range:
        n = 0
        for j in l:
            if j>=i and j<i+(max_-min_)/accuracy:
                n+=1
        y.append(n/len(l))
    x = list(data_range)
    if xl=='adsorb_energy':
        plt.xlim((-1.56, 16.35))
        my_x_ticks = np.linspace(-1.56, 16.35, 10)
    if xl=='sum_of_r':
        plt.xlim((380, 580))
        my_x_ticks = np.linspace(380, 580, 10)
    plt.xticks(my_x_ticks)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.scatter(x,y)
    plt.show()
def count_element(l,Metals):
    Metals_ = Metals
    result = {}
    for m in Metals_:
        result[m] = l.count(m)/len(l)
    return result
def plot_rect(l,Metals):
    plt.figure(figsize=(10,5))
    # 构建x与颜色的映射关系
    x = Metals
    color_map = {}
    for i, category in enumerate(x):
        color_map[category] = plt.cm.viridis(i / len(x))
    x = [i[0] for i in l]
    y = [i[1] for i in l]
    # 创建柱形图
    colors = [color_map[category] for category in x]
    plt.ylabel('(unacc-all)/all')
    plt.bar(x, y, color=colors)
    plt.show()

def show_performance(bad_predictions):
    all_N = ['QV1__','QV2__','QV3__','QV4__','QV5__','QV6__']
    all_C = ['QV1_012345','QV2_012345','QV3_012345','QV4_0123456','QV5_01234567','QV6_0123456']
    bad_pred_total_number = {}
    for i in all_N+all_C:
        bad_pred_total_number[i] = 0
    for i in all_N+all_C:
        for j in bad_predictions:
            if i in j:
                bad_pred_total_number[i] += 1
    print(bad_pred_total_number)

    bad_pred_cis_number = {}
    bad_pred_trans_number = {}
    for i in all_N+all_C:
        bad_pred_cis_number[i] = 0
        bad_pred_trans_number[i] = 0
    for i in all_N+all_C:
        for j in bad_predictions:
            if i in j:
                contcar_path = os.path.join('E:\\ResearchGroup\\AdsorbEPred\\tmp\\CONTCARs\\CONTCARs',j,'CONTCAR')
                if os.path.getsize(contcar_path)!=0:
                    with open(contcar_path,'r') as f:
                        lines = f.readlines()
                        graphene_coord_list = [lines[i] for i in range(8,63)]
                        graphene_z_list = [float(i.split()[2]) for i in graphene_coord_list]
                        graphene_z = sum(graphene_z_list)/len(graphene_z_list)
                        M1_coord = lines[64]
                        M2_coord = lines[65]
                        M1_z = float(M1_coord.split()[2])
                        M2_z = float(M2_coord.split()[2])
                        
                        if (M1_z-graphene_z)*(M2_z-graphene_z)>0:
                            bad_pred_cis_number[i] += 1
                            # print(i,'同面')
                        else:
                            bad_pred_trans_number[i] += 1
                            # print(i,'异面')
                        # print(i,round(M1_z,3),round(M2_z,3),round(graphene_z,3))
    print('同面',bad_pred_cis_number)
    print('异面',bad_pred_trans_number)