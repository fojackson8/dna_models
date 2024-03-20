import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import json
import os


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




class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     
                                     nn.Linear(64, output_dim),
                                     )
        

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    
    def sigmatrix(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.relu(w04)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
        if self.state == 'train':
            pass
        elif self.state == 'test':
            z = F.relu(z)
            z = self.refraction(z)
            
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


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
        # print(idx)
        x = self.train_x[idx]
        y = self.train_y[idx]
        return x, y


def train_model(train_x, train_y,
                model_name=None,
                batch_size=128, epochs=128):
    
    # train_loader = DataLoader(SimDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(SimDataset(train_x, train_y), batch_size=batch_size, shuffle=True, drop_last=True)
    model = AutoEncoder(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print('Start training')
    model, loss, reconloss = training_stage(model, train_loader, optimizer, epochs=epochs)
    print('Training is done')
    print('Prediction loss is:')
    plot_loss(loss, title='Prediction Loss')
    print('Reconstruction loss is:')
    plot_loss(reconloss, title='Reconstruction Loss')
    if model_name is not None:
        print('Model is saved')
        torch.save(model, model_name+".pth")
    return model

def training_stage(model, train_loader, optimizer, epochs=128):
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []

    for i in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_recon_loss = 0
        for k, (data, label) in enumerate(train_loader):
            assert not torch.any(torch.isnan(data)) # make sure no NaNs in training data
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data)
            
            ## Stop gradients exploding during backprop
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            batch_loss.backward()
            optimizer.step()
            loss_value = F.l1_loss(cell_prop, label).cpu().detach().numpy()
            recon_loss_value = F.l1_loss(x_recon, data).cpu().detach().numpy()
            epoch_loss += loss_value
            epoch_recon_loss += recon_loss_value

        loss.append(epoch_loss / (k + 1))
        recon_loss.append(epoch_recon_loss / (k + 1))
        if i % 10 ==0:
            
            print(f"Epoch: {i+1}, Prediction Loss: {loss[-1]:.6f}, Reconstruction Loss: {recon_loss[-1]:.6f}")

    return model, loss, recon_loss

def plot_loss(loss, title='Loss'):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
    






# simulated_dir = "/well/ludwig/users/dyp502/deconvolution/simulated_data/exp3/"
simulated_dir = "/well/ludwig/users/dyp502/deconvolution/taps_atlas_deconv/autoencoder/exp6/"

### Grail cfDNA from exp3
suffix = "TAPS_atlas_dmbs.all_samples.top300_pairwise_100.hypo_1_atlas_df_12774_32"
train_y = np.load(simulated_dir + f"train_y_{suffix}.npy")
train_x = np.load(simulated_dir + f"train_x_{suffix}.npy")

train_y = torch.from_numpy(train_y).float().to(device)
train_x = torch.from_numpy(train_x).float().to(device)

test_y = np.load(simulated_dir + f"test_y_{suffix}.npy")
test_x = np.load(simulated_dir + f"test_x_{suffix}.npy")

# test_y = np.load(simulated_dir + f"test_y_{suffix}.npy")
# test_x = np.load(simulated_dir + f"test_x_{suffix}.npy")

models_dir = "/well/ludwig/users/dyp502/deconvolution/models/"
model_name=f"TAPS_Atlas_DMBs.{suffix}.test1"
model_path = models_dir + model_name + '.pth'
train_model=False

batch_size=128 
epochs=200 
seed=1

if train_model:

    model = train_model(train_x, train_y, batch_size=batch_size, epochs=epochs)


    # models_dir = "/well/ludwig/users/dyp502/deconvolution/models/"
    # # model_name=f"Grail_DMBs_mod2.{suffix}.test2"
    # model_name=f"TAPS_Atlas_DMBs.{suffix}.test1"
    torch.save(model, model_path)




else:
    print('Loading existing model from:', model_path)
    model = torch.load(model_path)


# if model is not None and model_name is None:
#     print('Model is saved without defined name')
#     torch.save(model, 'model.pth')






print('Predict cell fractions without adaptive training')
model.eval()
model.state = 'test'
data = torch.from_numpy(test_x).float().to(device)
_, pred, _ = model(data)
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




with open(simulated_dir + f'Metrics.TAPS_atlas.{suffix}.json', 'w') as f:
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




# ### Now apply to real data samples_df

real_test_x = np.array(samples_df.T)


# model_name=None
# if model_name is not None and model is None:
#     model = torch.load(model_name+".pth")
# elif model is not None and model_name is None:
#     model = model
# print('Predict cell fractions without adaptive training')
# model.eval()
# model.state = 'test'
data = torch.from_numpy(real_test_x).float().to(device)
_, pred, _ = model(data)
# pred = pred.cpu().detach().numpy()

# celltypes = [f"tissue_{i}" for i in range(5)]
celltypes = list(atlas_df.columns)
samplename = np.arange(0,len(real_test_x),1)
pred_cpu = pred.cpu().detach().numpy()
pred_df = pd.DataFrame(pred_cpu, columns=celltypes, index=samples_df.columns).T
print('Prediction is done')


pred_df.to_csv(simulated_dir + f'Predicted_sample_mixtures.TAPS_atlas.{suffix}.csv',
                    sep='\t')








