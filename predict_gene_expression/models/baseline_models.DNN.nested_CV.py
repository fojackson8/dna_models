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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import torch
import torch.nn as nn
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
model_name = "1kb.DNN.nested_cv.exp1." # after switching order for - strand genes
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
# examine_data_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/examine_data/"

# n = 100000
# data_tensor_filtered_subset = data_tensor_filtered[:n]
# rna_seq_values_subset = rna_seq_values[:n]
# rna_seq_values_df_subset = rna_seq_values_df.iloc[:n]

# np.save(examine_data_dir + data_load_name + "data_tensor_filtered.npy", data_tensor_filtered_subset)
# np.save(examine_data_dir + data_load_name + "rna_seq_values_subset.npy", rna_seq_values_subset)
# rna_seq_values_df_subset.to_csv(examine_data_dir + data_load_name + "gene_tissue_combinations_rnaseq_subset.csv", sep='\t', index=False)













n_epigenetic_cols = 3
n_1kb_bins = 60
test_run = False
batch_size = 32
### Store model predictions
predictions_dict = {}





class SimpleDNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleDNN, self).__init__()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, 512)  # First layer with input_size neurons
        self.fc2 = nn.Linear(512, 256)  # Second layer
        self.fc3 = nn.Linear(256, 128)  # Third layer
        self.fc4 = nn.Linear(128, output_size)  # Output layer with 1 neuron for regression

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function for the output layer in regression tasks
        return x


# Adjustments for model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_features = n_epigenetic_cols * n_1kb_bins  # Calculate the total number of input features
model = SimpleDNN(input_size=n_features).to(device)
print(model)







# config = {"epochs":50,
# "learning_rate":0.001}






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



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()






results = []
all_tissue_predictions_df = pd.DataFrame()
counter = 0
for tissue in rna_seq_values_df['rna_seq_tissue'].unique():
# for tissue in list(rna_seq_values_df['rna_seq_tissue'].unique())[4:5]:
# for tissue in ['lung']:
    start_time = time.time() 
    print(f"Holding out {tissue}")
    leave_out_tissue=tissue
    train_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue']!=tissue]
    leave_out_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue']==tissue]
    train_indices = list(train_df.index)
    leave_out_indices = list(leave_out_df.index)
    print(f"Training on {train_df.shape}")
    print(f"Testing on {leave_out_df.shape}")

    train_array = data_tensor_filtered[train_indices]
    test_array = data_tensor_filtered[leave_out_indices]
    train_target = rna_seq_values[train_indices]
    test_target = rna_seq_values[leave_out_indices]


    if test_run:
        topn = 10000
        train_array = train_array[0:topn,:,:]
#         test_array = test_array[0:topn,:,:]
        
        train_target = train_target[0:topn]
#         test_target = test_target[0:topn]

    ## Reshape train_array and test_array into 2D for regression / feedforward
    train_array = train_array.reshape((train_array.shape[0],  n_epigenetic_cols*n_1kb_bins))
    test_array = test_array.reshape((test_array.shape[0], n_epigenetic_cols*n_1kb_bins))

    # Print the shapes of the training and testing arrays
    print(f"Shape of train_array: {train_array.shape}")
    print(f"Shape of test_array: {test_array.shape}")

    # If you're also interested in the target shapes
    print(f"Shape of train_target: {train_target.shape}")
    print(f"Shape of test_target: {test_target.shape}")


    # np.save(examine_data_dir + data_load_name + "train_array.npy", train_array[:n,:])
    # np.save(examine_data_dir + data_load_name + "test_array.npy", rna_seq_values_subset)

    # np.save(examine_data_dir + data_load_name + "train_target.npy", data_tensor_filtered_subset)
    # np.save(examine_data_dir + data_load_name + "test_target.npy", rna_seq_values_subset)

    # Convert arrays to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(torch.tensor(train_array, dtype=torch.float32).unsqueeze(1), torch.tensor(train_target, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_array, dtype=torch.float32).unsqueeze(1), torch.tensor(test_target, dtype=torch.float32))
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   # model = ConvNet()
    model = SimpleDNN(input_size=n_features).to(device)
    model.apply(weights_init)

    # wandb.init(config=args)

    if use_wandb:
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Reinitialize the scheduler if you're creating a new optimizer instance
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
#     epochs = 100
    if test_run:
        epochs = 1
    else:
        epochs = 20
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
        scheduler.step()  # Step the learning rate scheduler

    # Evaluation
    train_preds, train_labels = evaluate(model, train_loader, device)
    test_preds, test_labels = evaluate(model, test_loader, device)

    print(f"Shape of train_preds: {len(train_preds)}")
    print(f"Shape of train_labels: {len(train_labels)}")
    print(f"Shape of test_preds: {len(test_preds)}")
    print(f"Shape of test_labels: {len(test_labels)}")

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


    end_time = time.time()  # End time measurement here
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Time taken for {tissue}: {elapsed_time:.2f} seconds")  # Print the time taken

    results.append({
            'train_samples': train_array.shape[0],
            'test_samples': test_array.shape[0],
            'Tissue': tissue,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R2': train_r2,
            'Test R2': test_r2
        })


    if use_wandb:
        wandb.log({'train_data': train_array[10,:,:],
                'test_data': test_array[10,:,:],
                'Tissue': tissue,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R2': train_r2,
                'Test R2': test_r2
                })




    model_save_path = model_save_dir + f"{model_name}{tissue}.trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")


    counter+=1

results_df = pd.DataFrame(results)
results_df.to_csv(model_save_dir + model_name + 'tissue_results.csv', sep='\t')



with open(model_save_dir + model_name + "predictions_dict.pkl", "wb") as f:
    pickle.dump(predictions_dict, f)


all_tissue_predictions_df.to_csv(model_save_dir + model_name + "model_predictions_df.csv")

















 