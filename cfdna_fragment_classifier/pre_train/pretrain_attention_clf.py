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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM, Permute, Reshape, Multiply, Lambda, BatchNormalization

import wandb
from wandb.keras import WandbCallback




from model_training_functions import data_prepare_from_config, make_chrom_region, conv_onehot_longer_custom_encoding, lstm_seq_longer_custom_encoding




''''
Just TAPS:
- Train on healthy TAPS vs Lo + CD tumour 
- Test on everything else
'''


exp_no = 5
suffix = "cfTAPS_new_DMBs"
# dmr_setting = "Liver_tumour_vs_healthy_dmbs.top1000.hyper_hypo"
dmr_setting = "all_samples_dmbs.Liver_tumour_and_Liver.300.hyper_hypo"

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
config_dict['width'] = 150
config_dict['train_fraction'] = 0.95
config_dict['split_no'] = 1
config_dict['experiment_description'] = "Training on just TAPS liver tumour vs healthy cfDNA"
config_dict['epochs'] = 20
config_dict['learning_rate'] = 0.001

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
load_experiment = False # if True, don't recreate training data, just load what we've previously saved
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






###   NEW MODEL OPTIONs

def attention_block(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_probs')(a)
    output_attention = Multiply()([inputs, a_probs])
    return output_attention

def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Attention layer
    x = attention_block(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

input_shape = (150, 5)  
model = build_attention_model(input_shape)

sgd = tf.keras.optimizers.SGD(lr=config_dict['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])











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

    print(f'\n\nResults saved in {train_results_dir}')


else:
    model.load_weights(train_results_dir + f'weight.{suffix}.h5')












print(f'\n\nScript finished running at \n {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')