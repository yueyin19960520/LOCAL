import torch
import numpy as np

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    
def pos2cohp_train(model, data_loader, setting, optimizer):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean') if setting["Binary_COHP"] else torch.nn.MSELoss()
    
    model.train()
    train_matrix = measure_matrix()
    for data in data_loader:
        model = model.to(device)
        data = data.to(device)
        pred = model(data)
        #pred = model(data.x.to(device), data.edge_index.to(device), data.MN_edge_index)
    
        target = data.MN_icohp.to(device)
                     
        loss = loss_func(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(pred, target)
        
    res = train_matrix.absolute_correct_rate()*100 if setting["Binary_COHP"] else train_matrix.MAE()
    return loss, res


def pos2cohp_evaluate(model, data_loader, setting):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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