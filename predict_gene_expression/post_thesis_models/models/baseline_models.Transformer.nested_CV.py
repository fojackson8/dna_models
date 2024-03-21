import pandas as pd
import numpy as np
import os
import re
import collections
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import optimize
import pickle
# import plotly.express as px
import datetime
from scipy import stats
# from scipy.stats import zscore
import time


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

import wandb

sys.path.append('/well/ludwig/users/dyp502/code/tissue_atlas_code_2023/')
from tissue_atlas_v3_functions import merge_filter_new_tissue_samples

sys.path.append('/well/ludwig/users/dyp502/code/taps_tissue_atlas_Nov2022/')
from taps_tissue_atlas_functions import make_tissue_average_df
from taps_tissue_atlas_functions import plot_region, plot_tissue_sample_atlas, visualise_bed



with open('/well/ludwig/users/dyp502/tissue_atlas_v3/metadata/Atlas_V3_Metadata_New.pickle', 'rb') as f:
    project_metadata = pickle.load(f)

# Load all the variables into memory    
project_dir = project_metadata['project_dir']
atlas_dir = project_metadata['atlas_dir'] 
metadata_dir = project_metadata['metadata_dir']
gene_exp_dir = project_metadata['gene_exp_dir']
intersections_dir = project_metadata['intersections_dir']
id_tissue_map = project_metadata["id_tissue_map"]
tissue_id_map = project_metadata["tissue_id_map"]
blood_cell_types = project_metadata["blood_cell_types"]
somatic_tissue_types = project_metadata["somatic_tissue_types"]
healthy_tissue_types = project_metadata["healthy_tissue_types"]
cancer_tissue_types = project_metadata["cancer_tissue_types"]
diseased_tissue_types = project_metadata["diseased_tissue_types"]
tissues_with_tumour = project_metadata["tissues_with_tumour"]
tissue_order = project_metadata["tissue_order"]
genomic_features = project_metadata["genomic_features"]
sample_outliers = project_metadata["sample_outliers"]
healthy_high_cov_samples = project_metadata["healthy_high_cov_samples"]
genomic_feature_colnames = project_metadata["genomic_feature_colnames"]
genomic_regions_dir = project_metadata['genomic_regions_dir']



use_wandb=False


# data_load_name = "1kb_bins.leave_one_out.test11.remove_zeros."
# model_name = "1kb_bins.new_data.leave_one_out.exp20." # - strand genes are not switched in initial data processing, this data was then overwritten
# model_name = "1kb_bins.new_data.leave_one_out.exp22." # after switching order for - strand genes
model_name = "1kb.hybrid_model.nested_cv.exp2." # after switching order for - strand genes
data_load_name = "20kb_upstream_downstream_gene_equal_bins."
# save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/dl_model/"
save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/data/"
model_save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/results/"
save=False
load=True
normalise=True
remove_brain =True
use_normalised_TPM = False
# save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/"




### Load data
data_tensor_filtered = np.load(save_dir +  data_load_name +  "data_tensor_filtered.npy")
rna_seq_values = np.load(save_dir + data_load_name +  "rna_seq_values.npy")
rna_seq_values_df = pd.read_csv(save_dir +  data_load_name + f"gene_tissue_combinations_rnaseq.csv",sep='\t')

if remove_brain:
    rna_seq_values = rna_seq_values[rna_seq_values_df['tissue_x']!='Brain']
    data_tensor_filtered = data_tensor_filtered[rna_seq_values_df['tissue_x']!='Brain']
    rna_seq_values_df = rna_seq_values_df.loc[rna_seq_values_df['tissue_x']!='Brain']
    rna_seq_values_df = rna_seq_values_df.reset_index()








## Save a section of the data to examine
examine_data_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/examine_data/"

n = 100000
data_tensor_filtered_subset = data_tensor_filtered[:n]
rna_seq_values_subset = rna_seq_values[:n]
rna_seq_values_df_subset = rna_seq_values_df.iloc[:n]

# np.save(examine_data_dir + data_load_name + "data_tensor_filtered.npy", data_tensor_filtered_subset)
# np.save(examine_data_dir + data_load_name + "rna_seq_values_subset.npy", rna_seq_values_subset)
# rna_seq_values_df_subset.to_csv(examine_data_dir + data_load_name + "gene_tissue_combinations_rnaseq_subset.csv", sep='\t', index=False)













n_epigenetic_cols = 3
n_1kb_bins = 60

batch_size = 32
predictions_dict = {}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Linear(input_dim, d_model)
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
#             num_layers
#         )
#         self.fc = nn.Linear(d_model, num_classes)

#     def forward(self, x):
#         x = x.permute(2, 0, 1)  # Reshape to (sequence_length, batch_size, input_dim)
#         x = self.embedding(x)
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=0) 
#         x = self.fc(x)
#         return x




class HybridModel(nn.Module):
    def __init__(self, input_channels, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.embedding = nn.Linear(64, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, input_channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(2, 0, 1)  # Reshape to (sequence_length, batch_size, num_channels)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  
        x = self.fc(x)
        return x

input_dim = 3
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 256
num_classes = 1

# transformer_model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes).to(device)
hybrid_model = HybridModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes).to(device)





config = {"epochs":50,
"learning_rate":0.001}

if use_wandb:
    wandb.login() 
    run = wandb.init(project=f'predict_ge.DL_1kb.test9', config=config,entity="fojackson")

# wandb.init(config=args)


model = ConvNet()
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.optim.lr_scheduler import StepLR


# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by a factor of 0.1 every 10 epochs


# Training function
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)


        current_batch_size = targets.shape[0]
        if current_batch_size != batch_size:
            print(f"Warning: Batch has a different size ({current_batch_size}) than the specified batch size ({batch_size}).")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            current_batch_size = targets.shape[0]
            if current_batch_size != batch_size:
                print(f"Warning: Batch has a different size ({current_batch_size}) than the specified batch size ({batch_size}).")
                print(f"Inputs size: {inputs.shape}, Outputs size: {outputs.shape}")
            # Debug print to check the shape of your output
            # print(f"Shape of outputs: {outputs.shape}")

            # Check if outputs is a 0-d array (a scalar)
            if outputs.dim() == 0:
                print("Warning: outputs is a scalar")
                all_preds.append(outputs.item())
            else:
                all_preds.extend(outputs.squeeze().cpu().numpy())
                
            # Similar shape checking for targets
            # print(f"Shape of targets: {targets.shape}")
            if targets.dim() == 0:
                print("Warning: targets is a scalar")
                all_labels.append(targets.item())
            else:
                all_labels.extend(targets.cpu().numpy())
    return all_preds, all_labels







results = []
all_tissue_predictions_df = pd.DataFrame()
counter = 0
for tissue in rna_seq_values_df['rna_seq_tissue'].unique():
# for tissue in ['lung']:
    start_time = time.time()
    print(f"Holding out {tissue}")
    leave_out_tissue = tissue
    train_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue'] != tissue]
    leave_out_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue'] == tissue]
    train_indices = list(train_df.index)
    leave_out_indices = list(leave_out_df.index)
    print(f"Training on {train_df.shape}")
    print(f"Testing on {leave_out_df.shape}")

    train_array = data_tensor_filtered[train_indices]
    test_array = data_tensor_filtered[leave_out_indices]
    train_target = rna_seq_values[train_indices]
    test_target = rna_seq_values[leave_out_indices]

    # Split the training data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_array, train_target, test_size=0.2, random_state=42
    )

    # Convert arrays to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1), torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32).unsqueeze(1), torch.tensor(val_labels, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_array, dtype=torch.float32).unsqueeze(1), torch.tensor(test_target, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = ConvNet().to(device)
    model = HybridModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes).to(device)
    model.apply(weights_init)

    if use_wandb:
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop with Early Stopping
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    best_model = None

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Load the best model
    model.load_state_dict(best_model)

    # Retrain the model on the combined training and validation data
    combined_data = np.concatenate((train_data, val_data), axis=0)
    combined_labels = np.concatenate((train_labels, val_labels), axis=0)

    combined_dataset = TensorDataset(torch.tensor(combined_data, dtype=torch.float32).unsqueeze(1), torch.tensor(combined_labels, dtype=torch.float32))
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Reset the model weights and train on the combined data
    model = ConvNet().to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        train_loss = train(model, combined_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        scheduler.step()

    # Evaluation
    train_preds, train_labels = evaluate(model, combined_loader, device)
    test_preds, test_labels = evaluate(model, test_loader, device)

    leave_out_df['model_preds'] = test_preds
    all_tissue_predictions_df = pd.concat([all_tissue_predictions_df, leave_out_df])

    ### Store model predictions
    predictions_dict[tissue] = {
        'train_preds': train_preds,
        'train_labels': train_labels,
        'test_preds': test_preds,
        'test_labels': test_labels
    }

    train_mae = mean_absolute_error(train_labels, train_preds)
    test_mae = mean_absolute_error(test_labels, test_preds)

    train_r2 = r2_score(train_labels, train_preds)
    test_r2 = r2_score(test_labels, test_preds)

    print(f"[{tissue}] Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"[{tissue}] Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for {tissue}: {elapsed_time:.2f} seconds")

    results.append({
        'train_samples': combined_data.shape[0],
        'test_samples': test_array.shape[0],
        'Tissue': tissue,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'Test R2': test_r2
    })

    if use_wandb:
        wandb.log({
            'train_data': train_array[10,:,:],
            'test_data': test_array[10,:,:],
            'Tissue': tissue,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R2': train_r2,
            'Test R2': test_r2
        })

        # "Examples": example_images,
        # "Test Accuracy": 100. * correct / len(test_loader.dataset),
        # "Test Loss": test_loss})


    
    model_save_path = model_save_dir + f"{model_name}{tissue}.trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")


    counter+=1

results_df = pd.DataFrame(results)
results_df.to_csv(model_save_dir + model_name + 'tissue_results.csv', sep='\t')



with open(model_save_dir + model_name + "predictions_dict.pkl", "wb") as f:
    pickle.dump(predictions_dict, f)


all_tissue_predictions_df.to_csv(model_save_dir + model_name + "model_predictions_df.csv")















 