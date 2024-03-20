import numpy as np
import os
import os.path
import pandas as pd
import sys
import pysam
import gc
from pyfaidx import Fasta
from collections import Counter
import random
import re
import codecs
import collections
import pickle
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import tensorflow as tf
import time
import datetime
import pickle
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

import wandb
from wandb.keras import WandbCallback






''''
Just TAPS:
- Train on healthy TAPS vs Lo + CD tumour 
- Test on everything else
'''


exp_no = 6
suffix = "cfTAPS_new_DMBs"
dmr_setting = "Liver_tumour_vs_healthy_dmbs.top1000.hyper_hypo"
# dmr_setting = "all_samples_dmbs.Liver_tumour_and_Liver.300.hyper_hypo"

reads_dir="/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/read_files/"
experiments_dir = '/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/experiments/'
exp_dir = experiments_dir + f"exp{exp_no}/"
train_dir = exp_dir + 'train_dir/'
train_results_dir = exp_dir + 'train_dir/results/'
test_dir = exp_dir + "test_dir/"
figures_dir = exp_dir + 'figures/'
encoded_samples_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/encoded_samples/"


config_dict = {}
config_dict['reads_dir'] = reads_dir
config_dict['exp_dir'] = exp_dir
config_dict['train_dir'] = train_dir
config_dict['train_results_dir'] = train_results_dir
config_dict['test_dir'] = test_dir
config_dict['encoded_samples_dir'] = encoded_samples_dir
config_dict['dmr_setting'] = dmr_setting

for directory in [train_dir,train_results_dir,test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


config_dict['exp_no'] = exp_no
config_dict['trimmed'] = False
config_dict['width'] = 180
config_dict['train_fraction'] = 0.95
config_dict['split_no'] = 1
config_dict['experiment_description'] = "Training on just TAPS liver tumour vs healthy cfDNA"
config_dict['epochs'] = 20
config_dict['learning_rate'] = 0.001
# samples from split 1 of exp11
# config_dict['Lo_Healthy_training_set'] = ['CTR110', 'CTR111', 'CTR147', 'CTR148', 'CTR149', 'CTR108','CTR98', 'CTR152', 'CTR103', 'CTR117', 'CTR153', 'CTR154', 'CTR129', 'CTR113', 'CTR151', 'CTR114', 'CTR126', 'CTR85']
# config_dict['Lo_tumour_training_set'] = ['HOT170T', 'HOT215T', 'HOT236T', 'HOT172T', 'HOT222T', 'HOT207T', 'HOT229T', 'HOT197T', 'HOT151T']

lstm_encoding = {'A':1, 'T':2, 'C':3, 'G':4, 'm':5}
lstm_encoding_reverse = {u:v for v,u in lstm_encoding.items()}
config_dict['lstm_encoding'] = lstm_encoding
min_length_filter = False
config_dict['min_length_filter'] = min_length_filter

same_model=False
use_trimmed_data = config_dict['trimmed'] # Whether or not to use trimmed reads 
width = config_dict['width'] # Set width of arrays that represent each read (width = read length)
train_fraction = config_dict['train_fraction'] # fraction of data to be used for model training. Remaining data is used as internal test set
split_no = config_dict['split_no']

train_new_model = False






sample_list = list(set([u.split('.')[0] for u in os.listdir(reads_dir)]))


remove_samples = ['Healthy_NHCCJORO037']
# samples_dict = {}
categories = []
for sample in sample_list:
    if sample in remove_samples:
        continue
    category=None
    if sample.startswith('Healthy'):
        category = 'Healthy_cfDNA'
    elif sample.startswith('HCC_'):
        category = 'HCC_cfDNA'
    elif sample.startswith('PDAC_'):
        category = 'PDAC_cfDNA'
    elif 'Liver-Tumour' in sample:
        category = 'Liver_Tumour'
    elif 'Pancreas-Tumour' in sample:
        category = 'Pancreas_Tumour'
    if category:
        categories.append({'Sample': sample, 'Category': category})

samples_df = pd.DataFrame(categories)

config_df = samples_df.sort_values(by='Category')
training_healthy_idx = config_df[config_df['Category'] == 'Healthy_cfDNA'].sample(n=10, random_state=42).index

# Mark all selected healthy cfDNA and all liver tumour samples as training
config_df['Set'] = 'Test'  # Default value for all rows
config_df.loc[training_healthy_idx, 'Set'] = 'Train'
config_df.loc[config_df['Category'] == 'Liver_Tumour', 'Set'] = 'Train'
config_df.loc[config_df['Category'] == 'PDAC_cfDNA', 'Set'] = 'Neither'
config_df.loc[config_df['Category'] == 'Pancreas_Tumour', 'Set'] = 'Neither'

config_df['Label'] = config_df['Sample'].apply(lambda x: 1 if 'HCC' in x or 'Liver-Tumour' in x else 0)
config_df = config_df.reset_index()

config_dict['config_df'] = config_df

# with open(exp_dir+f'exp{exp_no}_config_dict.pickle', 'wb') as handle:
#     pickle.dump(config_dict, handle)

# Adapted code to work with the updated config_df
healthy_train = config_df[(config_df['Set'] == 'Train') & (config_df['Label'] == 0)]['Sample']
cancer_train = config_df[(config_df['Set'] == 'Train') & (config_df['Label'] == 1)]['Sample']
healthy_test = config_df[(config_df['Set'] == 'Test') & (config_df['Label'] == 0)]['Sample']
cancer_test = config_df[(config_df['Set'] == 'Test') & (config_df['Label'] == 1)]['Sample']


print(f"Training model on: \n\
      \n Healthy\n {', '.join(healthy_train)}\n \
      \n Cancer\n {', '.join(cancer_train)}\n\nTesting model on: \n\
      \n Healthy\n {', '.join(healthy_test)}\n \
      \n Cancer\n {', '.join(cancer_test)}\
      ")






# External model testing
print(f'\n\nBeginning external model testing \n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
all_testing_samples = list(healthy_test) + list(cancer_test)
print(f"\nEvaluating on trained model on following test samples:\n{all_testing_samples}")

# Initialize a list to hold all predictions from each model
ensemble_predictions = [[] for _ in range(len(models))]

# Load weights for each model and predict
for i, model in enumerate(models):
    model.load_weights(f'model_{i}.h5')
    for sample in all_testing_samples:
        seq_3one_hot = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.conv_onehot.{suffix}.npy')
        seq_3one_hot = seq_3one_hot[:, 0:180, :]  # Adjust the slicing based on new input shape
        prediction = model.predict(seq_3one_hot, verbose=0)
        ensemble_predictions[i].append(prediction)
        print(f"Prediction has shape {prediction.shape}")

# Calculate the mean prediction across all models
final_predictions = [np.mean([pred[i] for pred in ensemble_predictions], axis=0) for i in range(len(all_testing_samples))]

# Save final predictions for each sample
for i, sample in enumerate(all_testing_samples):
    np.savetxt(test_dir + 'result_' + f'{sample}.{dmr_setting}.txt', final_predictions[i])

print("External model testing completed.")
