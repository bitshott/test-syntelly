import torch

from torch import nn
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, Sequential, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import deepchem as dc
import pandas as pd
import numpy as np

import mlflow
import json

import sys
sys.path.append('../')

from config.experiment_config import config
from utils.feature_extractor import featurizer


# In[2]:


import os
import random

def set_seed(SEED):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)   # True
    torch.backends.cudnn.deterministic = True  # True
    torch.backends.cudnn.benchmark = False       # False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(config.RANDOM_SEED)


# In[5]:

class MeltDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        self.dataset = pd.read_csv(config.DATA_PATH / "melt_clean_filtered.csv")

        self.graphs = []
        for _, row in self.dataset.iterrows():
            data = self.featurizer.featurize(row['canonical_smiles'])[0]
            graph = Data(
                x=torch.tensor(data.node_features, dtype=torch.float32),
                y=torch.tensor(row['melt_value'], dtype=torch.float32),
                edge_index=torch.tensor(data.edge_index, dtype=torch.long),
                edge_attr=torch.tensor(data.edge_features, dtype=torch.float32),
                smiles=row['canonical_smiles']
            )
            self.graphs.append(graph)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)



# In[6]:


class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(x)

        return x
    
class GATBlock(torch.nn.Module):
    def __init__(self, in_channels : int, out_channels : int, dropout_rate: float, n_heads: int, concat=True):
        super(GATBlock, self).__init__()
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.gat = GATConv(
                        in_channels, 
                        out_channels, 
                        heads=self.n_heads, 
                        concat=concat, 
                        dropout=self.dropout_rate,
                        residual=True)

        self.bn_channels = out_channels * n_heads if concat else out_channels
        self.bn = BatchNorm(self.bn_channels)
    
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)   
        x = self.bn(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(x)

        return x


# In[7]:


class NetRDkit(torch.nn.Module):
    def __init__(self, hidden_dims: list[int], dropout_rate: float):
        super(NetRDkit, self).__init__()
        self.gcns = nn.ModuleList()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            gcn_block = GCNBlock(in_channels=in_dim,
                                 out_channels=out_dim,
                                 dropout_rate=self.dropout_rate)

            self.gcns.append(gcn_block)

        self.fusion_dim = self.hidden_dims[-1] + 182

        self.output_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.fusion_dim//2, self.hidden_dims[-1]//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dims[-1]//2, 1)
        )

    def _calc_rdkit(self, smiles_list: list[str]):
        descriptors = []
        for smiles in smiles_list:
            desc_dict = featurizer.smiles_to_rdkit_desc(smiles=smiles, columns_to_drop=featurizer.melt_columns_to_drop)
            values = torch.tensor(list(desc_dict.values()), dtype=torch.float32)
            descriptors.append(values)

        return torch.stack(descriptors, dim=0)

    def forward(self, data):
        x, edge_index, batch, smiles = data.x, data.edge_index, data.batch, data.smiles
        for block in self.gcns:
            x = block(x, edge_index)

        x = global_mean_pool(x, batch)
        rdkit_vec = self._calc_rdkit(smiles).to(x.device)
        cr = torch.cat([x, rdkit_vec], dim=1)
        out = self.output_layer(cr).view(-1)
        return out
    

class Net(torch.nn.Module):
    def __init__(self, hidden_dims: list[int], dropout_rate: float):
        super(Net, self).__init__()
        self.gcns = nn.ModuleList()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            gcn_block = GCNBlock(in_channels=in_dim,
                                 out_channels=out_dim,
                                 dropout_rate=self.dropout_rate)

            self.gcns.append(gcn_block)

        self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for block in self.gcns:
            x = block(x, edge_index)

        x = global_mean_pool(x, batch)
        out = self.output_layer(x).view(-1)
        return out
    
class NetGAT(torch.nn.Module):
    def __init__(self, hidden_dims: list[int], dropout_rate: float, n_heads: int = 3):
        super(NetGAT, self).__init__()
        self.gats = nn.ModuleList()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        last_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            gat_block = GATBlock(in_channels=last_dim, 
                                    out_channels=out_dim, 
                                    n_heads=n_heads, 
                                    dropout_rate=dropout_rate)
            
            last_dim = out_dim * n_heads
            self.gats.append(gat_block)

        self.final_dim = self.hidden_dims[-1] * n_heads
        self.output_layer = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.final_dim//2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for block in self.gats:
            x = block(x, edge_index)

        x = global_mean_pool(x, batch)
        out = self.output_layer(x).view(-1)
        return out

# In[8]:


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            loss = F.mse_loss(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs

    y_true = np.concatenate(y_true)    
    y_pred = np.concatenate(y_pred)
    print(y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)   
    mae = mean_absolute_error(y_true, y_pred)
    return total_loss / len(loader.dataset), rmse, r2, mae


# In[9]:


def cross_validation(dataset, model_class, train_one_epoch_fn, evaluate_fn,
                device, generator, seed_worker, k_folds=5, batch_size=64, l2_norm=1e-4, momentum=0.97, epochs=None, 
                learning_rate=None, split_path=None, net_parameters=None, config=None, tags=None):

    cv_results = {
        f'fold_{key}':  {
                    'loss_train': [],
                    'loss_val': [],
                    'loss_test': [],
                    'val_rmse': [],
                    'val_r': [],
                    'val_mae': [],
                    'test_rmse': [],
                    'test_r': [],
                    'test_mae': [],
                    'epoch': []
        } for key in range(0, k_folds)
    }
    mlflow.set_experiment('GCN_arch')
    with mlflow.start_run(run_name=f'{config.TIMESTAMP}', tags=tags):   
        for fold in range(k_folds):
            train_idx, val_idx, test_idx = [], [], []
            data_split = json.load((split_path).open())
            split_dict = data_split[fold]

            for index, idx in enumerate(dataset.dataset['index'].astype(str)):
                split = split_dict.get(idx)

                if split == "train":
                    train_idx.append(index)
                elif split == "val":
                    val_idx.append(index)
                elif split == "test":
                    test_idx.append(index)

            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            test_dataset = Subset(dataset, test_idx)
            print(f'Fold: {fold}:\n Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=4,
                                        worker_init_fn=seed_worker, generator=generator)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4,
                                        worker_init_fn=seed_worker, generator=generator)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4,
                                        worker_init_fn=seed_worker, generator=generator)

            model = model_class(**net_parameters).to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                            momentum=momentum, 
                                            weight_decay=l2_norm)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                        factor=0.7, threshold=0.05, patience=10,
                                        min_lr=1e-6)
            for epoch in range(epochs):
                train_loss = train_one_epoch_fn(model, train_loader, optimizer, device)
                val_loss, val_rmse, val_r, val_mae = evaluate_fn(model, val_loader, device)

                scheduler.step(val_loss)

                mlflow.log_metric(f'Loss/Train_MSE_fold_{fold}', train_loss, epoch)
                mlflow.log_metric(f'Loss/Validation_MSE_fold_{fold}', val_loss, epoch)
                mlflow.log_metric(f'Validation/MAE_fold_{fold}', val_mae, epoch)
                mlflow.log_metric(f'Validation/RMSE_fold_{fold}', val_rmse, epoch)
                mlflow.log_metric(f'Validation/R2_fold_{fold}', val_r, epoch)
                mlflow.log_metric(f'LR/Learning Rate_fold_{fold}', optimizer.param_groups[0]['lr'], epoch)

                fold_key = f'fold_{fold}'
                cv_results[fold_key]['loss_train'].append(train_loss)
                cv_results[fold_key]['loss_val'].append(val_loss)
                cv_results[fold_key]['val_rmse'].append(val_rmse)
                cv_results[fold_key]['val_r'].append(val_r)
                cv_results[fold_key]['val_mae'].append(val_mae)
                cv_results[fold_key]['epoch'].append(epoch)

            mlflow.pytorch.log_model(model, name=f'model_fold_{fold}')

            _, test_rmse, test_r, test_mae = evaluate_fn(model, test_loader, device)

            cv_results[fold_key]['test_rmse'].append(test_rmse)
            cv_results[fold_key]['test_r'].append(test_r)            
            cv_results[fold_key]['test_mae'].append(test_mae)

        calculate_test_results(cv_results=cv_results, epochs=epochs, mlflow=mlflow, model=model)



def calculate_test_results(cv_results: dict, epochs: int, mlflow, model):
    test_rmse_all = []
    test_r_all = []
    test_mae_all = []
    for fold_key in cv_results.keys():
        test_rmse_all.append(cv_results[fold_key]['test_rmse'])
        test_r_all.append(cv_results[fold_key]['test_r'])
        test_mae_all.append(cv_results[fold_key]['test_mae'])

    test_rmse_array = np.array(test_rmse_all).flatten()
    test_r_array = np.array(test_r_all).flatten() 
    test_mae_array = np.array(test_mae_all).flatten()


    mean_test_rmse = np.mean(test_rmse_array)
    std_test_rmse = np.std(test_rmse_array)

    mean_test_r = np.nanmean(test_r_array)
    std_test_r = np.nanstd(test_r_array)

    mean_test_mae = np.mean(test_mae_array)
    std_test_mae = np.std(test_mae_array)

    mlflow.log_metrics({
    "Test/RMSE_mean": mean_test_rmse,
    "Test/RMSE_std": std_test_rmse,
    "Test/MAE_mean": mean_test_mae,
    "Test/MAE_std": std_test_mae,
    "Test/R2_mean": mean_test_r,
    "Test/R2_std": std_test_r
    })


if __name__ == "__main__":

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)

    hidden_dims = [30, 64, 128]

    net_params = {
    'hidden_dims': hidden_dims,
    'dropout_rate': 0.2
    } 

    run_params = {
        'l2_norm': 1e-4,
        'batch_size': 16,
        'learning_rate': 1e-3, 
        'momentum': 0.98,
        "device": config.DEVICE, 
        "epochs": 300,
        "k_folds": 5,
    }

    run_description = """
    # Hidden dimensions:
    hidden_dims = [30, 64, 128]
    # Architecture details:
    class Net(torch.nn.Module):
        def __init__(self, hidden_dims: list[int], dropout_rate: float):
            super(Net, self).__init__()
            self.gcns = nn.ModuleList()
            self.hidden_dims = hidden_dims
            self.dropout_rate = dropout_rate

            for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
                gcn_block = GCNBlock(in_channels=in_dim,
                                    out_channels=out_dim,
                                    dropout_rate=self.dropout_rate)

                self.gcns.append(gcn_block)

            self.output_layer = nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for block in self.gcns:
                x = block(x, edge_index)

            x = global_mean_pool(x, batch)
            out = self.output_layer(x).view(-1)
            return out

    """

    tags = {
        'mlflow.note.content': run_description
    }

    cross_validation(
        dataset=MeltDataset(), 
        net_parameters=net_params,
        **run_params,
        config=config,
        model_class=Net, 
        train_one_epoch_fn=train_one_epoch, 
        evaluate_fn=evaluate,
        split_path=config.DATA_PATH / 'melt_split.json',
        generator=generator,
        seed_worker=seed_worker,
        tags=tags)



