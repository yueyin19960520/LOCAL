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


#### POS2COHP ####
def pos2cohp_train(model, data_loader, setting, optimizer):
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean') if setting["Binary_COHP"] else torch.nn.MSELoss()
    model.train()
    train_matrix = measure_matrix()

    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        cohp_real = batch_data.MN_icohp.to(device)
        model = model.to(device)
        
        cohp_pred = model(batch_data)
        loss = loss_func(cohp_pred, cohp_real)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(cohp_pred, cohp_real)
    res = train_matrix.absolute_correct_rate()*100 if setting["Binary_COHP"] else train_matrix.MAE()
    return loss, res


def pos2cohp_evaluate(model, data_loader, setting):
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean') if setting["Binary_COHP"] else torch.nn.MSELoss()
    model.eval()
    eval_matrix = measure_matrix()

    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        cohp_real = batch_data.MN_icohp.to(device)
        model = model.to(device)

        cohp_pred = model(batch_data)
        loss = loss_func(cohp_pred, cohp_real)

        torch.cuda.empty_cache()

        eval_matrix.update(cohp_pred, cohp_real)
    res = eval_matrix.absolute_correct_rate()*100 if setting["Binary_COHP"] else eval_matrix.MAE()
    return loss, res


#### CONT2E ####
def cont2e_train(model, data_loader, loss_func, optimizer, train_with_real=False, is_trainning=False):
    model.train()
    train_matrix = measure_matrix()
    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        energy_real = batch_data.y.to(device)
        model = model.to(device)

        energy_pred = model(batch_data, train_with_real=train_with_real, is_trainning=is_trainning)
        loss = loss_func(energy_pred, energy_real) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(energy_pred, energy_real)
    res = train_matrix.MAE()
    return loss, res


def cont2e_evaluate(model, data_loader, loss_func, train_with_real=False, is_trainning=False):
    model.eval()
    eval_matrix = measure_matrix()
    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        energy_real = batch_data.y.to(device)
        model = model.to(device)
        
        energy_pred = model(batch_data, train_with_real=train_with_real, is_trainning=is_trainning)
        loss = loss_func(energy_pred, energy_real) 
        
        torch.cuda.empty_cache()

        eval_matrix.update(energy_pred, energy_real)
    res = eval_matrix.MAE()
    return loss, res


def cont2e_cal(model, data_loader):  #Getiing the each MAE
    model.eval()
    eval_matrix = measure_matrix()
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        energy_real = batch_data.y.to(device)
        model = model.to(device)
        
        energy_pred = model(batch_data)
        torch.cuda.empty_cache()

        eval_matrix.update(energy_pred, energy_real)

    y_pred = torch.cat(eval_matrix.y_pred, dim=0).numpy()
    y_true = torch.cat(eval_matrix.y_true, dim=0).numpy()
    np_absolute = np.absolute(np.subtract(y_pred, y_true))
    return np_absolute














def combined_loss(out1, out2, label1, label2, weight1=1.0, weight2=1.0):
    loss1 = F.mse_loss(out1, label1)
    loss2 = F.mse_loss(out2, label2)
    total_loss = weight1 * loss1 + weight2 * loss2
    return total_loss


def cont2e_train_serial(model, data_loader, loss_func, optimizer, weight1=1.0, weight2=1.0):
    # Adding the cohp matrix.
    model.train()
    train_matrix = measure_matrix()
    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        energy_real = batch_data.y.to(device)
        cohp_real = batch_data.MN_icohp.to(device)
        model = model.to(device)
        
        cohp_pred, energy_pred = model(batch_data)
        loss = loss_func(energy_pred, cohp_pred, energy_real, cohp_real, weight1=weight1, weight2=weight2) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_matrix.update(energy_pred, energy_real)
    res = train_matrix.MAE()
    return loss, res


def cont2e_evaluate_serial(model, data_loader, loss_func):
    model.eval()
    eval_matrix = measure_matrix()
    for batch_data in data_loader:

        batch_data = batch_data.to(device)
        energy_real = batch_data.y.to(device)
        model = model.to(device)
        
        cohp_pred, energy_pred = model(batch_data)
        loss = loss_func(energy_pred, energy_real) 
        
        torch.cuda.empty_cache()

        eval_matrix.update(energy_pred, energy_real)
    res = eval_matrix.MAE()
    return loss, res




def custom_loss(predicted_feats, target, MN_edge_index):
    # Basic loss (e.g., MSE loss)
    loss = F.mse_loss(predicted_feats, target)

    # Ensure that the prediction for adjacent edges are the same
    paired_predicted_feats = predicted_feats.view(-1, 2)
    consistency_loss = F.mse_loss(paired_predicted_feats[:, 0], paired_predicted_feats[:, 1])

    return loss + consistency_loss
