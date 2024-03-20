import numpy as np
import os
import os.path
import pandas as pd
import sys
import random
import re
import codecs
import time
from datetime import datetime
import pickle
from collections import defaultdict

from model_training_functions import data_prepare_from_config, make_chrom_region, conv_onehot_longer_custom_encoding, lstm_seq_longer_custom_encoding, plot_chr_coverage, plot_read_methylation, plot_base_distributions, plot_seq_lengths





# experiments_dir = '/well/ludwig/users/dyp502/read_classifier/dismir_experiments/'
# new_dmr_dir = "/well/ludwig/users/dyp502/read_classifier/dismir_experiments/taps_dmrs/encoded_samples"
# with open(new_dmr_dir + 'cftaps_native_dmr_dict.pickle', 'rb') as handle:
#     new_dmr_dict = pickle.load(handle)
width = 200



# ## Save the below parameters as config
# experiments_dir = '/well/ludwig/users/dyp502/read_classifier/dismir_experiments/'
# new_dmr_dir = experiments_dir + 'cftaps_native_switching_regions/'
# with open(new_dmr_dir + 'cftaps_native_dmr_dict.pickle', 'rb') as handle:
#     new_dmr_dict = pickle.load(handle)

# config_dict = {}
# config_dict['exp_no'] = 59
# config_dict['trimmed'] = False
# config_dict['width'] = width
# config_dict['train_fraction'] = 0.95
# config_dict['split_no'] = 1
# # Can either supply an int (which if a subset of total will cause random sampling from total), or a list to specify exact samples
# config_dict['Lo_Healthy_training_set'] = 0 # 22 Lo Healthy cfDNA samples in training set, rest in test 
# config_dict['CD_Healthy_training_set'] = 0 # Can be list, or an int between 0-4
# config_dict['TAPS_Healthy_training_set'] = new_dmr_dict['healthy'] # Can be list, or an int between 0-23
# config_dict['Lo_tumour_training_set'] = list(new_dmr_dict['tumour']) # Can be list, or an int between 0-13
# config_dict['CD_tumour_training_set'] = 2 # Can be list, or an int between 0-2

# config_dict['experiment_description'] = "Training on just TAPS healthy cfDNA vs Lo + CD tumour"
# config_dict['epochs'] = 40

# samples from split 1 of exp11
# config_dict['Lo_Healthy_training_set'] = ['CTR110', 'CTR111', 'CTR147', 'CTR148', 'CTR149', 'CTR108','CTR98', 'CTR152', 'CTR103', 'CTR117', 'CTR153', 'CTR154', 'CTR129', 'CTR113', 'CTR151', 'CTR114', 'CTR126', 'CTR85']
# config_dict['Lo_tumour_training_set'] = ['HOT170T', 'HOT215T', 'HOT236T', 'HOT172T', 'HOT222T', 'HOT207T', 'HOT229T', 'HOT197T', 'HOT151T']

lstm_encoding = {'A':1, 'T':2, 'C':3, 'G':4, 'm':5}
lstm_encoding_reverse = {u:v for v,u in lstm_encoding.items()}
# config_dict['lstm_encoding'] = lstm_encoding
# min_length_filter = False
# config_dict['min_length_filter'] = min_length_filter

# same_model=False
# exp_no = config_dict['exp_no'] # Set experiment number
# use_trimmed_data = config_dict['trimmed'] # Whether or not to use trimmed reads 
# width = config_dict['width'] # Set width of arrays that represent each read (width = read length)
# train_fraction = config_dict['train_fraction'] # fraction of data to be used for model training. Remaining data is used as internal test set
# split_no = config_dict['split_no']
# config_dict['epochs'] = 20
# config_dict['learning_rate'] = 0.01

train_new_model = True
new_training_data = True
overwrite = True
suffix = "cfTAPS_new_DMBs"

# encoded_samples_dir = "/well/ludwig/users/dyp502/read_classifier/dismir_experiments/taps_dmrs/encoded_samples/"
encoded_samples_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/encoded_samples/"
# input_dir = "/well/ludwig/users/dyp502/read_classifier/dismir_experiments/taps_dmrs/filtered_reads/"
input_dir = "/well/ludwig/users/dyp502/read_classifier/read_classifier_2024/read_files/"


# if config_dict['trimmed']:
#     suffix = f'dismir_10x_switching_regions_split_{split_no}_trim'
# else:
#     suffix = f'dismir_10x_switching_regions_split_{split_no}_full_length'



# reads_dir = f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/filtered_reads/dmrs1/split_{split_no}/'
# # reads_dir = f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/filtered_reads/dismir_10x_switching_regions/split_{split_no}/'
# experiments_dir = '/well/ludwig/users/dyp502/read_classifier/dismir_experiments/'
# exp_dir = experiments_dir + f"exp{exp_no}/"
# train_dir = f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/exp{exp_no}/train_dir/'
# train_results_dir = f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/exp{exp_no}/train_dir/results/'
# test_dir = f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/exp{exp_no}/test_dir/'
# figures_dir = experiments_dir + 'figures/'
# encoded_samples_dir = '/well/ludwig/users/dyp502/read_classifier/dismir_experiments/samples_onehot/'


# config_dict['reads_dir'] = reads_dir
# config_dict['exp_dir'] = exp_dir
# config_dict['train_dir'] = train_dir
# config_dict['train_results_dir'] = train_results_dir
# config_dict['test_dir'] = test_dir
# config_dict['encoded_samples_dir'] = encoded_samples_dir

        
# if os.path.exists(exp_dir+f'exp{exp_no}_config_dict.pickle'):
#     with open(exp_dir+f'exp{exp_no}_config_dict.pickle', 'rb') as handle:
#         current_config = pickle.load(handle)
#     if current_config != config_dict: # If something has changed in the config (defined above) since last save, then need to make new training data and save this new config
#         new_config = True
#         print('Config defined above is different from the one saved in this directory, so saving new config dict for this experiment')
#         with open(exp_dir+f'exp{exp_no}_config_dict.pickle', 'wb') as handle:
#             pickle.dump(config_dict, handle)
#     elif current_config == config_dict:  # If nothing has changed in config, then experiment is the same and we can use old training data
#         new_config = False
#         print('Config specifications defined above are the same as saved version')
# else:
#     print('No config saved, so saving one now')
#     with open(exp_dir+f'exp{exp_no}_config_dict.pickle', 'wb') as handle:
#         pickle.dump(config_dict, handle)




# config_df = pd.read_csv(f'/well/ludwig/users/dyp502/read_classifier/dismir_experiments/exp{exp_no}/config_df.csv')


# healthy_train = config_df.loc[config_df['train']==0]
# cancer_train = config_df.loc[config_df['train']==1]
# healthy_test = config_df.loc[config_df['test']==0]
# cancer_test = config_df.loc[config_df['test']==1]

# print(f"Training model on: \n\
#       \n Healthy\n {', '.join(healthy_train['samples'])}\n \
#       \n Cancer\n {', '.join(cancer_train['samples'])}\n\nTesting model on: \n\
#       \n Healthy\n {', '.join(healthy_test['samples'])}\n \
#       \n Cancer\n {', '.join(cancer_test['samples'])}\
#       ")



min_cpg = 1

file_index = sys.argv[1]
# filename = os.listdir(input_dir)[int(file_index)-1]
filename = os.listdir(input_dir)[int(file_index)]
print(f"considering {filename}\n")
# sample = filename.split('.')[0]
sample = filename.replace(".reads","")
if (f"{sample}.lstm_seq.{suffix}.npy" not in os.listdir(encoded_samples_dir)) or overwrite: # only convert sample if not already converted
# filename = f"{sample}.filtered.{split}.reads"
    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print(f"encoding {sample} starting at: {now}")

    input = open(input_dir + filename,'r')
    seq = []
    methy = []
    for item in input:
        item = item.split()
        cpg_count = int(item[6])        
        if cpg_count >= min_cpg:
            # if len(item[2])>=66: ## TO-DO: Check what effect this additional constraint has - how many reads are we filtering from test set?
            seq.append(item[2])
            methy.append(item[3])
        else:
            pass
            # print(f'filtering out read {item[0]}:{item[1]} - {item[2]}')
    input.close()
    seq_lstm = lstm_seq_longer_custom_encoding(seq, methy,width,lstm_encoding)
    seq_3one_hot = conv_onehot_longer_custom_encoding(seq_lstm,width,lstm_encoding)
    seq_lengths = np.array([len(u) for u in seq])

    # sample_lstm[sample] = seq_lstm
    # seq_lengths_dict[sample] = seq_lengths


    with open(encoded_samples_dir + f'{sample}.lstm_seq.{suffix}.npy', 'wb') as f:
        np.save(f, seq_lstm,allow_pickle=True)
    with open(encoded_samples_dir + f'{sample}.conv_onehot.{suffix}.npy', 'wb') as f:
        np.save(f, seq_3one_hot,allow_pickle=True)
    with open(encoded_samples_dir + f'{sample}.seq_lengths.{suffix}.npy', 'wb') as f:
        np.save(f, seq_lengths,allow_pickle=True)

    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print(f"finished {sample} at: {now}")




















# if (f"{sample}.lstm_seq.{suffix}.npy" not in os.listdir(encoded_samples_dir)) or overwrite: # only convert sample if not already converted
# # filename = f"{sample}.filtered.{split}.reads"
#     now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"encoding {sample} starting at: {now}")

#     input = open(input_dir + filename,'r')
#     seq = []
#     methy = []
#     for item in input:
#         item = item.split()
#         cpg = 0
#         for i in range(len(item[2]) - 1):
#             if (item[2][i] == 'C') & (item[2][i + 1] == 'G'): # Presumably this checks REFERENCE sequence
#                 cpg = cpg + 1
#         if cpg > 2:
#             # if len(item[2])>=66: ## TO-DO: Check what effect this additional constraint has - how many reads are we filtering from test set?
#             seq.append(item[2])
#             methy.append(item[3])
#         else:
#             pass
#             # print(f'filtering out read {item[0]}:{item[1]} - {item[2]}')
#     input.close()
#     seq_lstm = lstm_seq_longer_custom_encoding(seq, methy,width,lstm_encoding)
#     seq_3one_hot = conv_onehot_longer_custom_encoding(seq_lstm,width,lstm_encoding)
#     seq_lengths = np.array([len(u) for u in seq])

#     # sample_lstm[sample] = seq_lstm
#     # seq_lengths_dict[sample] = seq_lengths


#     with open(encoded_samples_dir + f'{sample}.lstm_seq.{suffix}.npy', 'wb') as f:
#         np.save(f, seq_lstm,allow_pickle=True)
#     with open(encoded_samples_dir + f'{sample}.conv_onehot.{suffix}.npy', 'wb') as f:
#         np.save(f, seq_3one_hot,allow_pickle=True)
#     with open(encoded_samples_dir + f'{sample}.seq_lengths.{suffix}.npy', 'wb') as f:
#         np.save(f, seq_lengths,allow_pickle=True)

#     now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"finished {sample} at: {now}")














