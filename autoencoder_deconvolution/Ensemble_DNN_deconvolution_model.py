import json
import os
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

''''
More diverse architectures for the two models in the ensemble.
Experimentation with variable dropout rates.
Implementation of cyclical learning rates for the optimizers.
Introduction of Stochastic Weight Averaging (SWA).

'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats

from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
warnings.filterwarnings("ignore")


def lin_concordance_correlation_coefficient(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    covar = np.mean((y_true - mean_y_true) * (y_pred - mean_y_pred))
    var_y_true = np.var(y_true)
    var_y_pred = np.var(y_pred)

    pearson_r, _ = stats.pearsonr(y_true.flatten(), y_pred.flatten())
    ccc = (2 * covar) / (var_y_true + var_y_pred + (mean_y_true - mean_y_pred)**2)

    return ccc




def row_wise_spearman_correlation(test_y, pred):
    correlations = []
    for i in range(test_y.shape[0]):
        corr, _ = stats.spearmanr(test_y[i], pred[i])
#         corr, _ = stats.pearsonr(test_y[i], pred[i])
        correlations.append(corr)
    
    correlations_array = np.array(correlations)
    return correlations_array,np.mean(correlations_array)


def row_wise_pearson_correlation(test_y, pred):
    correlations = []
    for i in range(test_y.shape[0]):
#         corr, _ = stats.spearmanr(test_y[i], pred[i])
        corr, _ = stats.pearsonr(test_y[i], pred[i])
        correlations.append(corr)
    
    correlations_array = np.array(correlations)
    return correlations_array,np.mean(correlations_array)









from torch.optim.swa_utils import AveragedModel, SWALR
class SimDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(device)

        print(f"\n\n shape in SimDataset")
        print(self.train_x.shape)
        print(self.train_y.shape)

    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        x = self.train_x[idx]
        y = self.train_y[idx]
        return x, y

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes):
        super(DNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_dim, layer_sizes[i]))
            else:
                layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())
            if i < len(layer_sizes) - 1:  # Apply dropout to all but last layer
                dropout_rate = 0.05 if i % 2 != 0 else 0
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DNN3(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes):
        super(DNN3, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_dim, layer_sizes[i]))
            else:
                layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())
            # Apply dropout differently to introduce more diversity
            dropout_rate = 0.1 if i % 2 == 0 else 0.05
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_ensemble(model1, model2, model3, train_loader, val_loader, optimizer1, optimizer2, optimizer3, epochs=100, early_stopping_patience=10):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model1.train()
        model2.train()
        model3.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Training step for model 1
            optimizer1.zero_grad()
            outputs1 = model1(inputs)
            loss1 = F.l1_loss(outputs1, labels)
            loss1.backward()
            optimizer1.step()

            # Training step for model 2
            optimizer2.zero_grad()
            outputs2 = model2(inputs)
            loss2 = F.l1_loss(outputs2, labels)
            loss2.backward()
            optimizer2.step()

            # Training step for model 3
            optimizer3.zero_grad()
            outputs3 = model3(inputs)
            loss3 = F.l1_loss(outputs3, labels)
            loss3.backward()
            optimizer3.step()

        # Validation step
        model1.eval()
        model2.eval()
        model3.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                outputs3 = model3(inputs)
                averaged_outputs = (outputs1 + outputs2 + outputs3) / 3
                val_loss = F.l1_loss(averaged_outputs, labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

def plot_loss(loss, title='Loss'):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


# simulated_dir = "/well/ludwig/users/dyp502/deconvolution/simulated_data/exp3/"
simulated_dir = "/well/ludwig/users/dyp502/deconvolution/taps_atlas_deconv/autoencoder/exp6/"

suffix = "TAPS_atlas_dmbs.all_samples.top300_pairwise_100.hypo_1_atlas_df_12774_32"
save_suffix = "TAPS_atlas_dmbs.all_samples.top300_pairwise_100.hypo_1_atlas_df_12774_32.ensemble_dnn_15"

train_y = np.load(simulated_dir + f"train_y_{suffix}.npy")
train_x = np.load(simulated_dir + f"train_x_{suffix}.npy")

train_y = torch.from_numpy(train_y).float().to(device)
train_x = torch.from_numpy(train_x).float().to(device)

test_y = np.load(simulated_dir + f"test_y_{suffix}.npy")
test_x = np.load(simulated_dir + f"test_x_{suffix}.npy")

# test_y = np.load(simulated_dir + f"test_y_{suffix}.npy")
# test_x = np.load(simulated_dir + f"test_x_{suffix}.npy")

models_dir = "/well/ludwig/users/dyp502/deconvolution/models/"
# model_name=f"TAPS_Atlas_DMBs.{suffix}.test1"
model_name=f"TAPS_Atlas_DMBs.{save_suffix}.test5"
model_path = models_dir + model_name + '.pth'



batch_size=128 
epochs=200 
seed=1




n_val = 5000

val_x = train_x[-n_val:]
val_y = train_y[-n_val:]

train_x = train_x[0:-n_val]
train_y = train_y[0:-n_val]





train_loader = DataLoader(SimDataset(train_x, train_y), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(SimDataset(val_x, val_y), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_x.shape[1]
output_dim = train_y.shape[1]
layer_sizes_1 = [1024, 512, 128]
layer_sizes_2 = [256, 128, 32]

model1 = DNN(input_dim, output_dim, layer_sizes_1).to(device)
model2 = DNN(input_dim, output_dim, layer_sizes_2).to(device)
# model3 = DNN3(input_dim, output_dim).to(device)
layer_sizes_3 = [512, 256, 64]  # Example layer sizes for model3
model3 = DNN3(input_dim, output_dim, layer_sizes_3).to(device)

optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

train_the_model=True
# Modify the training call to include the third model and optimizer
if train_the_model:
    train_loader = DataLoader(SimDataset(train_x, train_y), batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(SimDataset(val_x, val_y), batch_size=32)
    train_ensemble(model1, model2, model3, train_loader, val_loader, optimizer1, optimizer2, optimizer3, epochs=100, early_stopping_patience=30)

    model1_path = model_path.replace('.pth', '_model1.pth')
    model2_path = model_path.replace('.pth', '_model2.pth')
    model3_path = model_path.replace('.pth', '_model3.pth')

    torch.save(model1.state_dict(), model1_path)
    torch.save(model2.state_dict(), model2_path)
    torch.save(model3.state_dict(), model3_path)
    print(f"Saving models to {model1_path}, {model2_path}, {model3_path}")
else:
    print('Loading existing models from:', model_path)
    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))
    model3.load_state_dict(torch.load(model3_path))

# Prediction section, averaging predictions from all three models
print('Predict cell fractions without adaptive training')
model1.eval()
model2.eval()
model3.eval()
data = torch.from_numpy(test_x).float().to(device)
with torch.no_grad():
    pred1 = model1(data)
    pred2 = model2(data)
    pred3 = model3(data)
pred = (pred1 + pred2 + pred3) / 3
pred = pred.cpu().detach().numpy()


celltypes = [f"tissue_{i}" for i in range(test_y.shape[1])]
# celltypes = list(atlas_df.columns)
samplename = np.arange(0,len(test_y),1)
# pred_df = pd.DataFrame(pred, columns=celltypes, index=samplename)
print('Prediction is done')





mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y, pred)
linC = lin_concordance_correlation_coefficient(test_y, pred)
correlations_array,mean_corr = row_wise_spearman_correlation(test_y,pred)
pearson_correlations_array,pearson_mean_corr = row_wise_pearson_correlation(test_y,pred)


print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Concordance Correlation Coefficient: {linC:.4f}")
print(f"Mean Spearman's Correlation: {mean_corr:.4f}")
print(f"Mean Spearman's Correlation: {pearson_mean_corr:.4f}")



metrics = {
    "Mean Absolute Error": mae,
    "Mean Squared Error": mse,
    "Concordance Correlation Coefficient": linC,
    "Mean Spearman's Correlation": mean_corr,
    "Mean Pearson's Correlation": pearson_mean_corr
}



import json

with open(simulated_dir + f'Metrics.TAPS_atlas.{save_suffix}.json', 'w') as f:
    json.dump(metrics, f)




save_dir = "/well/ludwig/users/dyp502/deconvolution/taps_atlas_deconv/autoencoder/exp6/"
samples_file = "samples_df.TAPS_atlas_dmbs.all_samples.top300_pairwise_100.hypo_1_atlas_df_12774_32.csv"
samples_df = pd.read_csv(save_dir + samples_file,
               sep='\t').set_index('Unnamed: 0')

samples_df.index.name = 'chr_start_end'


atlas_df = pd.read_csv(save_dir+ "atlas_df.TAPS_atlas_dmbs.all_samples.top300_pairwise_100.hypo_1_atlas_df_12774_32.csv",
                      sep='\t').set_index('Unnamed: 0')

atlas_df.index.name = 'chr_start_end'


# samples_df.to_csv(save_dir + f"samples_df.{feature}_{counter}_{suffix}.csv",
#                sep='\t',
#                  index=True)



### Now apply to real data samples_df

real_test_x = np.array(samples_df.T)





print('Predict cell fractions without adaptive training')
model1.eval()
model2.eval()
model3.eval()
data = torch.from_numpy(real_test_x).float().to(device)
with torch.no_grad():
    pred1 = model1(data)
    pred2 = model2(data)
    pred3 = model3(data)
pred = (pred1 + pred2 + pred3) / 3
pred = pred.cpu().detach().numpy()


# pred = pred.cpu().detach().numpy()

# celltypes = [f"tissue_{i}" for i in range(5)]
celltypes = list(atlas_df.columns)
samplename = np.arange(0,len(real_test_x),1)
# pred_cpu = pred.cpu().detach().numpy()
pred_df = pd.DataFrame(pred, columns=celltypes, index=samples_df.columns).T
print('Prediction is done')


pred_df.to_csv(simulated_dir + f'Predicted_sample_mixtures.TAPS_atlas.{save_suffix}.csv',
                    sep='\t')




print(f"Saving predicted sample_mixtures to {save_suffix}")




