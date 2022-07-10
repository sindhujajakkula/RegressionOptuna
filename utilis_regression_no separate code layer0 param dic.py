import torch 
import torch.nn as nn


class RegDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.features[item, :], dtype = torch.float),
            'y': torch.tensor(self.targets[item, :], dtype = torch.float)
        }


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimzer = optimizer
        
    @staticmethod
    def loss_fn(targets, outputs):
        return nn.MSELoss()(outputs, targets)
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss +=loss.item()
        return final_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            final_loss +=loss.item()
        return final_loss / len(data_loader)
    
class Model(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_sizes, dropout):
        super().__init__()
        layers = []
        
        in_features = nfeatures
        #nlayers = trial.suugest_int('num_layers', 1, 7)
        
        for i in range(nlayers):
            #hidden_size = trial.suggest_int('hidden_size', 16, 2048)
            layers.append(nn.Linear(in_features, hidden_sizes[i]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            #dropout = trial.suggest_uniform('dropout', 0.1, 0.7)
            layers.append(nn.Dropout(dropout[i]))
            layers.append(nn.ReLU())
            in_features = hidden_sizes[i]
                
        layers.append(nn.Linear(in_features, ntargets))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)