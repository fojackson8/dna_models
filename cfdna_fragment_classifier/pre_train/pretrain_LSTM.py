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




from model_training_functions import data_prepare_from_config, make_chrom_region, conv_onehot_longer_custom_encoding, lstm_seq_longer_custom_encoding, LSTM_deep_longer




''''
Just TAPS:
- Train on healthy TAPS vs Lo + CD tumour 
- Test on everything else
'''


exp_no = 3
suffix = "cfTAPS_new_DMBs"
# dmr_setting = "Liver_tumour_vs_healthy_dmbs.top1000.hyper_hypo"
dmr_setting = "Pancreas_tumour_vs_healthy_dmbs.top1000.hyper_hypo"

reads_dir="/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/read_files/"
experiments_dir = '/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/experiments/'
exp_dir = experiments_dir + f"exp{exp_no}/"
train_dir = exp_dir + 'train_dir/'
train_results_dir = exp_dir + 'train_dir/results/'
test_dir = exp_dir + "test_dir/"
figures_dir = exp_dir + 'figures/'
encoded_samples_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/encoded_samples/"

for directory in [train_dir,train_results_dir,test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

config_dict = {}
config_dict['reads_dir'] = reads_dir
config_dict['exp_dir'] = exp_dir
config_dict['train_dir'] = train_dir
config_dict['train_results_dir'] = train_results_dir
config_dict['test_dir'] = test_dir
config_dict['encoded_samples_dir'] = encoded_samples_dir
config_dict['dmr_setting'] = dmr_setting

# for directory in [train_dir,train_results_dir,test_dir]:
#     if not os.path.exists(directory):
#         os.makedirs(directory)


config_dict['exp_no'] = exp_no
config_dict['trimmed'] = False
config_dict['width'] = 130
config_dict['train_fraction'] = 0.95
config_dict['split_no'] = 1
config_dict['experiment_description'] = "Training on just TAPS Pancreas tumour vs healthy cfDNA"
config_dict['epochs'] = 20
config_dict['learning_rate'] = 0.01
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
config_df.loc[config_df['Category'] == 'Pancreas_Tumour', 'Set'] = 'Train'
config_df.loc[config_df['Category'] == 'HCC_cfDNA', 'Set'] = 'Neither'
config_df.loc[config_df['Category'] == 'Liver_Tumour', 'Set'] = 'Neither'


config_df['Label'] = config_df['Sample'].apply(lambda x: 1 if 'PDAC' in x or 'Pancreas-Tumour' in x else 0)
config_df = config_df.reset_index()

config_dict['config_df'] = config_df

with open(exp_dir+f'exp{exp_no}_config_dict.pickle', 'wb') as handle:
    pickle.dump(config_dict, handle)

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





train_new_model = True

load_experiment = True # if True, don't recreate training data, just load what we've previously saved
### ------------------------------------              Load arrays for training       ------------------------------------------------------------------------
if load_experiment:
    print(f"Loading training data because load_experiment=True")
    # Load training data
    with open(train_dir + f'train_data.{suffix}.npy', 'rb') as f:
        train_data = np.load(f, allow_pickle=True)
    train_label = np.loadtxt(train_dir + f'train_label.{suffix}.txt')

    # Load testing data
    with open(train_dir + f'test_data.{suffix}.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)
    test_label = np.loadtxt(train_dir + f'test_label.{suffix}.txt')

    print(f'\n\nTraining data loaded from {train_dir}')

    print(f"Shape of loaded training data: {train_data.shape}")
    print(f"Shape of loaded training labels: {train_label.shape}")

    print(f"Shape of loaded testing data: {test_data.shape}")
    print(f"Shape of loaded testing labels: {test_label.shape}")



else:
    print(f"Making new training data")
    for k,sample in enumerate(healthy_train):
        label=0
        
        seq_lstm = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.lstm_seq.{suffix}.npy')
        seq_3one_hot = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.conv_onehot.{suffix}.npy')
        sample_read_length = len(seq_3one_hot)
        origin_array = np.array([sample]*sample_read_length)
    #     label = int(config_df.loc[config_df['samples']==sample][['train','test']].sum(axis=1)) # May be better set manually, for each list of healthy train, cancer train etc
        label_array = np.full(len(seq_3one_hot),label)

        
        if k==0:
            healthy_lstm_all = seq_lstm
            healthy_one_hot_all = seq_3one_hot
            healthy_label_all = label_array
            healthy_origin_all = origin_array
        else:
            healthy_lstm_all = np.vstack((healthy_lstm_all,seq_lstm))
            healthy_one_hot_all = np.vstack((healthy_one_hot_all,seq_3one_hot))
            healthy_label_all = np.append(healthy_label_all,label_array)
            healthy_origin_all = np.append(healthy_origin_all,origin_array)

    len_healthy_reads = healthy_origin_all.shape[0]

            
    for k,sample in enumerate(cancer_train):
        label = 1
        seq_lstm = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.lstm_seq.{suffix}.npy')
        seq_3one_hot = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.conv_onehot.{suffix}.npy')
        sample_read_length = len(seq_3one_hot)
        origin_array = np.array([sample]*sample_read_length)
    #     label = int(config_df.loc[config_df['samples']==sample][['train','test']].sum(axis=1)) # May be better set manually, for each list of healthy train, cancer train etc
        label_array = np.full(len(seq_3one_hot),label)

        if k==0:
            cancer_lstm_all = seq_lstm
            cancer_one_hot_all = seq_3one_hot
            cancer_label_all = label_array
            cancer_origin_all = origin_array
        else:
            cancer_lstm_all = np.vstack((cancer_lstm_all,seq_lstm))
            healthy_one_hot_all = np.vstack((healthy_one_hot_all,seq_3one_hot))
            cancer_label_all = np.append(cancer_label_all,label_array)
            cancer_origin_all = np.append(cancer_origin_all,origin_array)
        
    len_cancer_reads = cancer_origin_all.shape[0]
















    ## Combine healthy and cancer reads by stacking vertically
    if min_length_filter: # if True, then we healthy and cancer training arrays to have same length r, equal to smallest of the two arrays
        r = min(len_healthy_reads,len_cancer_reads)
        lstm_all = np.vstack((healthy_lstm_all[0:r],cancer_lstm_all[0:r]))
        label_all = np.append(healthy_label_all[0:r],cancer_label_all[0:r])
        origin_all = np.append(healthy_origin_all[0:r],cancer_origin_all[0:r])
        one_hot_all = np.vstack((healthy_one_hot_all[0:r],cancer_one_hot_all[0:r]))
        data = one_hot_all
    else:
        lstm_all = np.vstack((healthy_lstm_all,cancer_lstm_all))
        label_all = np.append(healthy_label_all,cancer_label_all)
        origin_all = np.append(healthy_origin_all,cancer_origin_all)
        one_hot_all = np.vstack((healthy_one_hot_all,cancer_one_hot_all))
        data = one_hot_all
        
        
    perm = random.sample(range(len_healthy_reads+len_cancer_reads), len_healthy_reads+len_cancer_reads) # randomise order of reads
    data = one_hot_all[:,0:width,:]
    data = data[perm]
    lstm_all = lstm_all[perm]
    label_all = label_all[perm]
    origin_all = origin_all[perm]






    train_num = int(len(data) * train_fraction) # 80% as training set, 75% among them for training, 25% among them for validation
    test_num = int(len(data) * (1-train_fraction))
    train_data = data[0:train_num] # take 0th to 60th percentile for training
    train_label = label_all[0:train_num]
    test_data = data[train_num:(train_num + test_num)] # take 60th to 80th percentile for test
    test_label = label_all[train_num:(train_num + test_num)]

    print(f"\n\nUsing {train_fraction} of dataset for training so final input data size for model training is:\
    \n {train_data.shape} \n Test data shape is: \n {test_data.shape}" )


    if train_new_model:

        ## Save training data
        with open(train_dir + f'train_data.{suffix}.npy', 'wb') as f:
            np.save(f, train_data,allow_pickle=True)
        np.savetxt(train_dir + f'train_label.{suffix}.txt', train_label)

        # Save testing data
        with open(train_dir + f'test_data.{suffix}.npy', 'wb') as f:
            np.save(f, test_data,allow_pickle=True)
        np.savetxt(train_dir + f'test_label.{suffix}.txt', test_label)

        print(f'\n\nTraining data saved in {train_dir}')





# Calculate the index for 10% of the training data
ten_percent_index = int(len(train_data) * 0.1)

# Save 10% of training data for quick testing
with open(train_dir + f'train_10_percent_x.{suffix}.npy', 'wb') as f:
    np.save(f, train_data[:ten_percent_index], allow_pickle=True)
np.savetxt(train_dir + f'train_10_percent_y.{suffix}.txt', train_label[:ten_percent_index])












## -------------------------          Initialise WandB logging            -------------------------
config = {
            "epochs": config_dict['epochs'],
            "batch_size": 128,
            "log_step": 200,
            "val_log_step": 50,
        }
run = wandb.init(project=f'exp{exp_no}', config=config,entity="fojackson")
config = wandb.config


print(f"\n\nUsing {train_fraction} of dataset for training so final input data size for model training is:\
\n {train_data.shape} \n Test data shape is: \n {test_data.shape}" )



# building and training the model
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
model = LSTM_deep_longer(config_dict['width'])
sgd = optimizers.SGD(lr=config_dict['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

if train_new_model:
    print("Not same model so training new model from scratch")
    ## Add in Tensorboard save
    log_dir = f"/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/logs/exp_{exp_no}/fit/" + f'{suffix}_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f'\n\n Beginning Model Training \n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    start = time.time()
    # history = model.fit(train_data, train_label, epochs=300, batch_size=128, validation_split=0.1,
    #                     callbacks=[EarlyStopping(patience=10), ModelCheckpoint(filepath=train_results_dir + f'weight.{suffix}.h5', save_best_only=True),tensorboard_callback],
    #                     shuffle=True, verbose=2)
    history = model.fit(train_data, train_label, epochs=config_dict['epochs'], batch_size=128, validation_split=0.1,
                        callbacks=[EarlyStopping(patience=5), ModelCheckpoint(filepath=train_results_dir + f'weight.{suffix}.h5', save_best_only=True),WandbCallback()],
                        shuffle=True, verbose=2)

    end = time.time()
    elapsed = end-start
    print(f'\nModel training for {suffix} complete in {elapsed/60} minutes')


    ## -------------------------------------------           Internal held-out test set          -------------------------------------------

    model.load_weights(train_results_dir + f'weight.{suffix}.h5')
    result = model.predict(test_data, verbose=2) # test data is a mix of tumour reads and healthy cfDNA reads

    np.savetxt(train_results_dir + f'test_label.{suffix}.txt', test_label)
    np.savetxt(train_results_dir + f'predict_result.{suffix}.txt', result)
    # np.savetxt(train_results_dir + f'test_chrom.{suffix}.txt', test_chrom)
    # np.savetxt(train_results_dir + f'test_region.{suffix}.txt', test_region)
    # result_all = model.predict(data[0:(train_num + test_num)], verbose=0)
    # np.savetxt(train_results_dir + f'data_lstm.{suffix}.txt', data_lstm_all)
    # np.savetxt(train_results_dir + f'label_all.{suffix}.txt', label_all)
    # np.savetxt(train_results_dir + f'chrom_all.{suffix}.txt', chrom_all)
    # np.savetxt(train_results_dir + f'region_all.{suffix}.txt', region_all)
    # np.savetxt(train_results_dir + f'result_all.{suffix}.txt', result_all)
    print(f'\n\nResults saved in {train_results_dir}')



else:
    print(f"Loading model weights from {train_results_dir}")
    model.load_weights(train_results_dir + f'weight.{suffix}.h5')
















## -------------------------------           Evaluating model on transfer learning task          -------------------------------------------

all_testing_samples = list(healthy_test) + list(cancer_test)

print(f'\n\nBeginning external model testing \n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# healthy_testing_samples = list(config_df.loc[config_df['test']==0]['samples'])
# cancer_testing_samples = list(config_df.loc[config_df['test']==1]['samples'])
# healthy_testing_files = list(config_df.loc[config_df['test']==0]['file'])
# cancer_testing_files = list(config_df.loc[config_df['test']==1]['file'])
model.load_weights(train_results_dir + f'weight.{suffix}.h5')
print(f"\nEvaluating on trained model on following test samples:\n{all_testing_samples}")


for sample in all_testing_samples:
# for sample in healthy_test + cancer_test:
    seq_3one_hot = np.load(encoded_samples_dir + f'{sample}.{dmr_setting}.conv_onehot.{suffix}.npy')
    seq_3one_hot = seq_3one_hot[:,0:width,:] 
    result = model.predict(seq_3one_hot, verbose=0)
    np.savetxt(test_dir + 'result_' +  f'{sample}.{dmr_setting}' + '.txt', result)


















## Calculate risk scores in test set - summarising read likelihoods across each patient to generate a single risk score


# config_df['risk_score'] = ''
# sample_risks = defaultdict(list)
# sample_all_scores = {}
# cancer_sample_scores = {}
# healthy_sample_scores = {}
# taps_hcc = {}
# taps_healthy = {}

# for file in os.listdir(test_dir):
# # cal risk by maximize posterior probability
#     gaps = np.linspace(0, 1, 1001)
#     score = np.vstack((gaps, 1 - gaps))
#     score = score.T
#     likelihood_1 = np.loadtxt(test_dir + file)
#     likelihood_2 = 1 - likelihood_1
#     likelihood = np.vstack((likelihood_1, likelihood_2))
#     val = np.log10(np.dot(score, likelihood))
#     sum = np.sum(val, axis=1)
#     result = gaps[np.argmax(sum)]

#     sample_name = file.split('.')[0][7:]
#     # config_df.loc[config_df['samples']==sample_name,'risk_score'] = result
#     config_df.loc[config_df['samples'].str.contains(sample_name),'risk_score'] = result
# config_df.to_csv(f"{exp_dir}risk_scores.csv",index=False)







print(f'\n\nScript finished running at \n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')