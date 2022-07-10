# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:50:21 2022

@author: sindh
"""

import optuna
import torch
import pandas as pd
import numpy as np
import utils
from sklearn.preprocessing import StandardScaler

EPOCHS = 100 # doesn't matter how many epochs we choose because we are hypertuning it
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run_training(params, save_model = False):
    
    train_df =  pd.read_csv('train_dataset_regression.csv')
    valid_df =  pd.read_csv('val_dataset_regression.csv')
    #read the data
    
    
    features_columns = ['E', 'nu', 'B_x', 'C_x', 'C_y', 'D_x', 'D_y']   
    target_columns = ['cf1', 'cf2', 'cf3', 'cf4']
    
    #creating numpy array of the data
    xtrain = train_df[features_columns].to_numpy() 
    ytrain = train_df[target_columns].to_numpy()
    
    xvalid = valid_df[features_columns].to_numpy()
    yvalid = valid_df[target_columns].to_numpy()   
    
    #Scaling input features using standard scaler
    scaler=StandardScaler()
    xtrain=scaler.fit_transform(xtrain)
    xvalid=scaler.transform(xvalid)
    
    #returns dictionary of X and y tensors
    train_dataset =utils.RegDataset(features=xtrain, targets=ytrain)
    valid_dataset =utils.RegDataset(features=xvalid, targets=yvalid)
    
    #loads the data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, num_workers = 8,  shuffle=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1024, num_workers = 8
    )
    
    #creating model object
    model = utils.Model(
        nfeatures = xtrain.shape[1],
        ntargets = ytrain.shape[1], 
        num_layers = params['num_layers'], 
        hidden_sizes = params['hidden_sizes'], 
        dropout = params['dropout']
    )
    
    
    model.to(DEVICE)
    
    #using adam as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    
    #creating object of the Engine class
    eng = utils.Engine(model, optimizer, device=DEVICE)
    
    #for early stopping
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    
    
    #iterating the training process over the range of EPOCHS.
    for epoch in range(EPOCHS):
        
        train_loss = eng.train(train_loader)
        valid_loss = eng.evaluate(valid_loader)
        print(f'epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), 'model.bin')
        else:
            early_stopping_counter+=1
        
        if early_stopping_counter > early_stopping_iter:
            break
            
    return best_loss
            
def objective(trial):
    params = {
        'num_layers' : trial.suugest_int('num_layers', 1, 7),
        'hidden_sizes' :tuple([trial.suggest_int('n_units_l{}'.format(i), 16, 2048) for i in range(params['num_layers'])]),
        'dropout' : tuple([trial.suggest_uniform('dropout_l{}'.format(i), 0.1, 0.7) for i in range(params['num_layers'])]),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    }
     
    loss = run_training(params, save_model=False) 
        
    return loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print('best_trial:')
    trial_ = study.best_trial
    
    print(trial_.values)
    print(trial_.params)
    

    score =  run_training(trial_.params, save_model=True)

        
    print(score)
    
