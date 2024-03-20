# ## This script contains functions to convert .read files into the final one-hot encoded arrays ready for model input
## Current - 09/05/2022

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



def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def data_prepare_from_config(file_dir,config_df,trimmed=True,output_read_origin=False):
    '''
    For each group from {healthy, tumour}, extracts and combines all reads into 4 lists.
    If a read:
    - Has length 66
    - Has >2 CpGs
    
    Makes 4 lists of chr, read_position, read_sequence, binary_meth state. 
    NOTE: reads are combined across samples, so end up with 4 lists of length: n_reads(s1) + n_reads(s2) + ...


    Outputs two sets of lists: 'Healthy' and 'Tumour'. The samples designated as each of these are specified
    in the config_df.
    
    If trimmed=False, will output full length reads, otherwise will trim reads to 66bp.
    
    If output_read_origin=True, will pass a 5th object in each set, giving the origin of each read, 
    eg Lo, 

    TO-DO: figure out why CG filter is unnecessary filtering out most TAPS reads. Do we need this second CG filter?
    '''
    
    healthy_training_samples = list(config_df.loc[config_df['train']==0]['samples'])
    tumour_training_samples = list(config_df.loc[config_df['train']==1]['samples'])

    healthy_training_files = list(config_df.loc[config_df['train']==0]['file'])
    tumour_training_files = list(config_df.loc[config_df['train']==1]['file'])

    # prepare reads from normal plasma
    normal_seq = []
    normal_methy = []
    normal_chrom = []
    normal_region = []
    normal_read_origin = []
    if trimmed: # include 66bp length filter
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in healthy_training_samples:  # keyword for normal plasma samples for training
                input_norm = open(file_dir + file, 'r')
                for item in input_norm:
                    item = item.split()
                    cpg = 0
                    if len(item[2]) == 66:  # check the length
                        for i in range(len(item[2]) - 1):
                            if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                                cpg = cpg + 1
                        if cpg > 2: # filter out reads with less than 3 CpG sites
                            normal_chrom.append(item[0])
                            normal_region.append(item[1])
                            normal_seq.append(item[2])
                            normal_methy.append(item[3])
                            normal_read_origin.append(sample)
                input_norm.close()
    else: # don't include 66bp length filter
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in healthy_training_samples:  # keyword for normal plasma samples for training
                input_norm = open(file_dir + file, 'r')
                for item in input_norm:
                    item = item.split()
                    cpg = 0
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        normal_chrom.append(item[0])
                        normal_region.append(item[1])
                        normal_seq.append(item[2])
                        normal_methy.append(item[3])
                        normal_read_origin.append(sample)
                input_norm.close()

    # prepare reads from cancer tissue
    tumor_seq = []
    tumor_methy = []
    tumor_chrom = []
    tumor_region = []
    tumor_read_origin = []
    files = file_name(file_dir)
    flag = 0
    if trimmed:
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in tumour_training_samples: # keyword for cancer tissue samples for training
                input_tumor = open(file_dir + file, 'r')
                for item in input_tumor:
                    item = item.split()
                    cpg = 0
                    if len(item[2]) == 66:
                        for i in range(len(item[2]) - 1):
                            if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                                cpg = cpg + 1
                        if cpg > 2: # filter out reads with less than 3 CpG sites
                            tumor_chrom.append(item[0])
                            tumor_region.append(item[1])
                            tumor_seq.append(item[2])
                            tumor_methy.append(item[3])
                            tumor_read_origin.append(sample)
                input_tumor.close()
            flag = flag + 1
    else:
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in tumour_training_samples: # keyword for cancer tissue samples for training
                input_tumor = open(file_dir + file, 'r')
                for item in input_tumor:
                    item = item.split()
                    cpg = 0
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        tumor_chrom.append(item[0])
                        tumor_region.append(item[1])
                        tumor_seq.append(item[2])
                        tumor_methy.append(item[3])
                        tumor_read_origin.append(sample)
                input_tumor.close()
            flag = flag + 1
        
    if output_read_origin:
        return normal_seq, normal_methy, normal_chrom, normal_region, normal_read_origin, tumor_seq, tumor_methy, tumor_chrom, tumor_region, tumor_read_origin
    else:
        return normal_seq, normal_methy, normal_chrom, normal_region, tumor_seq, tumor_methy, tumor_chrom, tumor_region






def data_prepare_from_config_no_filter(file_dir,config_df,trimmed=True,output_read_origin=False):
    '''
    This version has no CpG filter
    
    For each group from {healthy, tumour}, extracts and combines all reads into 4 lists.
    If a read:
    - Has length 66
    - Has >2 CpGs
    
    Makes 4 lists of chr, read_position, read_sequence, binary_meth state. 
    NOTE: reads are combined across samples, so end up with 4 lists of length: n_reads(s1) + n_reads(s2) + ...


    Outputs two sets of lists: 'Healthy' and 'Tumour'. The samples designated as each of these are specified
    in the config_df.
    
    If trimmed=False, will output full length reads, otherwise will trim reads to 66bp.
    
    If output_read_origin=True, will pass a 5th object in each set, giving the origin of each read, 
    eg Lo, 

    TO-DO: figure out why CG filter is unnecessary filtering out most TAPS reads. Do we need this second CG filter?
    '''
    
    healthy_training_samples = list(config_df.loc[config_df['train']==0]['samples'])
    tumour_training_samples = list(config_df.loc[config_df['train']==1]['samples'])

    healthy_training_files = list(config_df.loc[config_df['train']==0]['file'])
    tumour_training_files = list(config_df.loc[config_df['train']==1]['file'])

    # prepare reads from normal plasma
    normal_seq = []
    normal_methy = []
    normal_chrom = []
    normal_region = []
    normal_read_origin = []
    if trimmed: # include 66bp length filter
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in healthy_training_samples:  # keyword for normal plasma samples for training
                input_norm = open(file_dir + file, 'r')
                for item in input_norm:
                    item = item.split()
                    cpg = 0
                    if len(item[2]) == 66:  # check the length
                        for i in range(len(item[2]) - 1):
                            if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                                cpg = cpg + 1
                        if cpg > 2: # filter out reads with less than 3 CpG sites
                            normal_chrom.append(item[0])
                            normal_region.append(item[1])
                            normal_seq.append(item[2])
                            normal_methy.append(item[3])
                            normal_read_origin.append(sample)
                input_norm.close()
    else: # don't include 66bp length filter
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in healthy_training_samples:  # keyword for normal plasma samples for training
                input_norm = open(file_dir + file, 'r')
                for item in input_norm:
                    item = item.split()
                    cpg = 0
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        normal_chrom.append(item[0])
                        normal_region.append(item[1])
                        normal_seq.append(item[2])
                        normal_methy.append(item[3])
                        normal_read_origin.append(sample)
                input_norm.close()

    # prepare reads from cancer tissue
    tumor_seq = []
    tumor_methy = []
    tumor_chrom = []
    tumor_region = []
    tumor_read_origin = []
    files = file_name(file_dir)
    flag = 0
    if trimmed:
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in tumour_training_samples: # keyword for cancer tissue samples for training
                input_tumor = open(file_dir + file, 'r')
                for item in input_tumor:
                    item = item.split()
                    cpg = 0
                    if len(item[2]) == 66:
                        for i in range(len(item[2]) - 1):
                            if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                                cpg = cpg + 1
                        if cpg > 2: # filter out reads with less than 3 CpG sites
                            tumor_chrom.append(item[0])
                            tumor_region.append(item[1])
                            tumor_seq.append(item[2])
                            tumor_methy.append(item[3])
                            tumor_read_origin.append(sample)
                input_tumor.close()
            flag = flag + 1
    else:
        for file in healthy_training_files+tumour_training_files:
            sample = file.split('.')[0]
            if sample in tumour_training_samples: # keyword for cancer tissue samples for training
                input_tumor = open(file_dir + file, 'r')
                for item in input_tumor:
                    item = item.split()
                    cpg = 0
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        tumor_chrom.append(item[0])
                        tumor_region.append(item[1])
                        tumor_seq.append(item[2])
                        tumor_methy.append(item[3])
                        tumor_read_origin.append(sample)
                input_tumor.close()
            flag = flag + 1
        
    if output_read_origin:
        return normal_seq, normal_methy, normal_chrom, normal_region, normal_read_origin, tumor_seq, tumor_methy, tumor_chrom, tumor_region, tumor_read_origin
    else:
        return normal_seq, normal_methy, normal_chrom, normal_region, tumor_seq, tumor_methy, tumor_chrom, tumor_region








# transform sequence into number for storage (A/T/C/G to 0/1/2/3, methylated C to 4)
def lstm_seq(seq, methy):
    '''
    Takes as input sequence and methylation_state lists
    Outputs a numpy array of size n_rows = n_reads, n_cols = 66
    Replaces each letter with a number 
    If methylation_state == 1, assigns letter 4
    '''
    i = 0
    lstmseq = np.zeros((len(seq), 66), dtype='int')
    while i < len(seq):
        tmp = seq[i]
        j = 0
        while j < len(tmp):
            if tmp[j] == 'A':
                lstmseq[i, j] = 0
            elif tmp[j] == 'T':
                lstmseq[i, j] = 1
            elif tmp[j] == 'C':
                lstmseq[i, j] = 2
            else:
                lstmseq[i, j] = 3
            if int(methy[i][j]) == 1:
                lstmseq[i, j] = 4
            j = j + 1
        i = i + 1
    return lstmseq


def data_prepare_include_cancerdetector(file_dir,split,splits_dict):
    '''
    
    For each group from {healthy, tumour}, extracts and combines all reads into 4 lists.
    If a read:
    - Has length 66
    - Has >2 CpGs
    
    Makes 4 lists of chr, read_position, read_sequence, binary_meth state. 
    NOTE: reads are combined across samples, so end up with 4 lists of length: n_reads(s1) + n_reads(s2) + ...

    Addition (12/11/2021) - Also filters by split
    
    Modified (21/02/2022) to add CancerDetector HCC and healthy cfDNA to training set
    '''
    
    cancer_detector_healthy_cfdna = [ 'N1L', 'N2L','N3L', 'N4L'] 
    cancer_detector_cancer_tumour = ['HCC1', 'HCC2']
    
    healthy_training_samples = splits[f"{split}"]['training_set']['healthy_cfdna']
    healthy_training_samples += cancer_detector_healthy_cfdna
    tumour_training_samples = splits[f"{split}"]['training_set']['tumour']
    tumour_training_samples += cancer_detector_cancer_tumour
    
    # prepare reads from normal plasma
    normal_seq = []
    normal_methy = []
    normal_chrom = []
    normal_region = []
    file_dir = file_dir + split + '/'
    files = file_name(file_dir)
    for file in files:
        sample = file.split('.')[0]
        if sample in healthy_training_samples:  # keyword for normal plasma samples for training
            input_norm = open(file_dir + file, 'r')
            for item in input_norm:
                item = item.split()
                cpg = 0
                if len(item[2]) == 66:  # check the length
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        normal_chrom.append(item[0])
                        normal_region.append(item[1])
                        normal_seq.append(item[2])
                        normal_methy.append(item[3])
            input_norm.close()

    # prepare reads from cancer tissue
    tumor_seq = []
    tumor_methy = []
    tumor_chrom = []
    tumor_region = []
    files = file_name(file_dir)
    flag = 0
    for file in files:
        sample = file.split('.')[0]
        if sample in tumour_training_samples: # keyword for cancer tissue samples for training
            input_tumor = open(file_dir + file, 'r')
            for item in input_tumor:
                item = item.split()
                cpg = 0
                if len(item[2]) == 66:
                    for i in range(len(item[2]) - 1):
                        if (item[2][i] == 'C') & (item[2][i + 1] == 'G'):
                            cpg = cpg + 1
                    if cpg > 2: # filter out reads with less than 3 CpG sites
                        tumor_chrom.append(item[0])
                        tumor_region.append(item[1])
                        tumor_seq.append(item[2])
                        tumor_methy.append(item[3])
            input_tumor.close()
        flag = flag + 1
    return normal_seq, normal_methy, normal_chrom, normal_region, tumor_seq, tumor_methy, tumor_chrom, tumor_region




def lstm_seq_longer(seq, methy,width):
    '''
    Takes as input sequence and methylation_state lists
    Outputs a numpy array of size n_rows = n_reads, n_cols = 66
    Replaces each letter with a number 
    If methylation_state == 1, assigns letter 4
    '''
    i = 0
    lstmseq = np.zeros((len(seq), width), dtype='int') # let the width of the array be defined in variable width
    while i < len(seq):
        tmp = seq[i]
        j = 0
        while j < len(tmp):
            if tmp[j] == 'A':
                lstmseq[i, j] = 0
            elif tmp[j] == 'T':
                lstmseq[i, j] = 1
            elif tmp[j] == 'C':
                lstmseq[i, j] = 2
            else:
                lstmseq[i, j] = 3
            if int(methy[i][j]) == 1:
                lstmseq[i, j] = 4
            j = j + 1
        i = i + 1
    return lstmseq


def lstm_seq_longer_custom_encoding(seq, methy,width,encoding):
    '''
    Takes as input sequence and methylation_state lists
    Outputs a numpy array of size n_rows = n_reads, n_cols = 66
    Replaces each letter with a number 
    If methylation_state == 1, assigns letter 4

    Change (26/05/2022): Now have to specify encoding in a dictionary mapping bases to numbers.
    eg to encoding +1 to allow for padding, would need to pass {'A':1, 'T':2, 'C':3, 'G':4, 'm':5} Now empty space=0, A=1, T=2 etc 
    '''
    i = 0
    lstmseq = np.zeros((len(seq), width), dtype='int') # let the width of the array be defined in variable width
    while i < len(seq):
        tmp = seq[i]
        j = 0
        while (j < len(tmp)) & (j<width):
            lstmseq[i, j] = encoding[tmp[j]]
            if int(methy[i][j]) == 1:
                lstmseq[i, j] = encoding['m']
            # if tmp[j] == 'A':
            #     lstmseq[i, j] = 1
            # elif tmp[j] == 'T':
            #     lstmseq[i, j] = 2
            # elif tmp[j] == 'C':
            #     lstmseq[i, j] = 3
            # else:
            #     lstmseq[i, j] = 4
            # if int(methy[i][j]) == 1:
            #     lstmseq[i, j] = 5
            j = j + 1
        i = i + 1
    return lstmseq

# transform sequence into one-hot code (0/1/2/3 to one-hot) and add methylation state channel
def conv_onehot(seq):
    
    '''
    Loops through the sequence + methylation state numpy array, replacing each state {0:4} with a one hot encoding
    vector of length 5
    So each read is now represented as 66x5 array
    Then appends each of these arrays into a list of length n_reads
    '''
    module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
    onehot = np.zeros((len(seq), 66, 5), dtype='int')
    for i in range(len(seq)):
        tmp = seq[i]
        tmp_onehot = np.zeros((66, 5), dtype='int')
        for j in range(len(tmp)):
            if tmp[j] == 0:
                tmp_onehot[j] = module[0]
            elif tmp[j] == 1:
                tmp_onehot[j] = module[1]
            elif tmp[j] == 2:
                tmp_onehot[j] = module[2]
            elif tmp[j] == 3:
                tmp_onehot[j] = module[3]
            else:
                tmp_onehot[j] = module[4]
        onehot[i] = tmp_onehot
    return onehot


def conv_onehot_longer(seq,width):
    
    '''
    Loops through the sequence + methylation state numpy array, replacing each state {0:4} with a one hot encoding
    vector of length 5
    So each read is now represented as 66x5 array
    Then appends each of these arrays into a list of length n_reads
    '''
    module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
    onehot = np.zeros((len(seq), width, 5), dtype='int')
    for i in range(len(seq)):
        tmp = seq[i]
        tmp_onehot = np.zeros((width, 5), dtype='int')
        for j in range(len(tmp)):
            if tmp[j] == 0:
                tmp_onehot[j] = module[0]
            elif tmp[j] == 1:
                tmp_onehot[j] = module[1]
            elif tmp[j] == 2:
                tmp_onehot[j] = module[2]
            elif tmp[j] == 3:
                tmp_onehot[j] = module[3]
            else:
                tmp_onehot[j] = module[4]
        onehot[i] = tmp_onehot
    return onehot





# def conv_onehot_longer_custom_encoding(seq,width,encoding):
    
#     '''
#     Loops through the sequence + methylation state numpy array, replacing each state {0:4} with a one hot encoding
#     vector of length 5
#     So each read is now represented as 66x5 array
#     Then appends each of these arrays into a list of length n_reads
#     '''
#     lstm_encoding_values = list(encoding.values())
#     module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
#     onehot = np.zeros((len(seq), width, 5), dtype='int')
#     for i in range(len(seq)):
#         tmp = seq[i]
#         tmp_onehot = np.zeros((width, 5), dtype='int')
#         for j in range(len(tmp)):
#             if tmp[j] == lstm_encoding_values[0]:
#                 tmp_onehot[j] = module[0]
#             elif tmp[j] == lstm_encoding_values[1]:
#                 tmp_onehot[j] = module[1]
#             elif tmp[j] == lstm_encoding_values[2]:
#                 tmp_onehot[j] = module[2]
#             elif tmp[j] == lstm_encoding_values[3]:
#                 tmp_onehot[j] = module[3]
#             else:
#                 tmp_onehot[j] = module[4]
#         onehot[i] = tmp_onehot
#     return onehot




def conv_onehot_longer_custom_encoding(seq,width,encoding):
    
    '''
    Loops through the sequence + methylation state numpy array, replacing each state {0:4} with a one hot encoding
    vector of length 5
    So each read is now represented as 66x5 array
    Then appends each of these arrays into a list of length n_reads
    '''
    lstm_encoding_values = list(encoding.values())
    module = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])
    padding = np.array([0, 0, 0, 0, 0])
    onehot = np.zeros((len(seq), width, 5), dtype='int')
    for i in range(len(seq)):
        tmp = seq[i]
        tmp_onehot = np.zeros((width, 5), dtype='int')
        for j in range(len(tmp)):
            if tmp[j] == lstm_encoding_values[0]:
                tmp_onehot[j] = module[0]
            elif tmp[j] == lstm_encoding_values[1]:
                tmp_onehot[j] = module[1]
            elif tmp[j] == lstm_encoding_values[2]:
                tmp_onehot[j] = module[2]
            elif tmp[j] == lstm_encoding_values[3]:
                tmp_onehot[j] = module[3]
            elif tmp[j] == lstm_encoding_values[4]:
                tmp_onehot[j] = module[4]
            else:
                tmp_onehot[j] = padding #all values outside of possible lstm_encoding values are assigned as padding 
        onehot[i] = tmp_onehot
    return onehot






# get the chromosome information and position of each read
def make_chrom_region(chrom_0, region_0):
    i = 0
    chrom = np.zeros(len(chrom_0), dtype='int')
    region = np.zeros(len(region_0), dtype='int')
    while i < len(chrom_0):
        chrom[i] = int(re.findall('(\d+)', chrom_0[i])[0])
        region[i] = region_0[i]
        i = i + 1
    return chrom, region



def LSTM_deep_longer(width):
    model = Sequential()
    model.add(layers.Conv1D(filters=100,
                                   input_shape=(width, 5),
                                   kernel_size=10,
                                   padding="same",
                                   activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(layers.LSTM(int(width/2), return_sequences=True)))
    model.add(layers.Conv1D(input_shape=(int(width/2), 132),
                                   filters=100,
                                   kernel_size=3,
                                   padding="same",
                                   activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(750, activation='relu', kernel_regularizer=None, bias_regularizer=None))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None))
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None))
    return model













## Visualising data conversion

def plot_chr_coverage(chroms,origins_array,sample_metadata_df,status='healthy_cfdna',savename=False):
    fig,ax = plt.subplots(1,figsize=(22,12),sharex=True,sharey=True)
    colors = ['red','green','blue']
    for k,origin in enumerate(['Lo','cancer_detector','TAPS']):
        sample_list = list(sample_metadata_df.loc[(sample_metadata_df['status'].isin([status])) & (sample_metadata_df['origin'] == origin)]['samples'])
        for j, sample in enumerate(sample_list):
            sample_chroms = np.array(chroms)[np.isin(origins_array,sample)]
            chr_counts = np.unique(sample_chroms,return_counts=True)
            chr_counts_df = pd.DataFrame(chr_counts[1],index=chr_counts[0],columns=[sample]).reindex(index=[f'chr{u}' for u in np.arange(1,23)])
            if j==0:
                df = chr_counts_df
            else:
                df[sample] = chr_counts_df[sample]
        if len(sample_list)>0:
            sns.scatterplot(y='value',x='index',
                        data=pd.melt(df.reset_index(),id_vars='index'),
                        ax=ax,color=colors[k],label=origin)

            ax.grid(visible=True)
            ax.set_title(origin)

    fig.suptitle(f'Comparing chromosome read coverage across {status} ',fontsize=24)
    if savename:
        plt.savefig(savename)


def plot_read_methylation(seq_lstm,origins_array,sample_metadata_df,lstm_encoding,status='healthy_cfdna',savename=False,normalise=False,sharey=False):
    if sharey:
        fig,axes = plt.subplots(3,figsize=(20,16),sharex=True,sharey=True)
    else:
        fig,axes = plt.subplots(3,figsize=(20,16),sharex=True,sharey=False)
    for k,origin in enumerate(['Lo','cancer_detector','TAPS']):
        sample_list = list(sample_metadata_df.loc[(sample_metadata_df['status'].isin([status])) & (sample_metadata_df['origin'] == origin)]['samples'])
        ax = axes[k]
        sample_lstm_reads = seq_lstm[np.isin(origins_array,sample_list)]
        if normalise:
            fragment_lengths = np.sum(sample_lstm_reads==(lstm_encoding['A']-1),axis=1)
            sns.distplot(np.sum(sample_lstm_reads==lstm_encoding['m'],axis=1)/fragment_lengths,ax=ax,hist=True,kde=False)
        else:
            sns.distplot(np.sum(sample_lstm_reads==lstm_encoding['m'],axis=1),ax=ax,hist=True,kde=False)
        ax.grid(visible=True)
        ax.set_title(origin)

    fig.suptitle(f'Comparing read methylated cytosine count across {status}',fontsize=24)
    if savename:
        plt.savefig(savename)


def plot_base_distributions(seq_lstm,origins_array,sample_metadata_df,lstm_encoding_reverse,status='healthy_cfdna',savename=False):

    fig,axes = plt.subplots(3,figsize=(20,16),sharex=True,sharey=True)
    for k,origin in enumerate(['Lo','cancer_detector','TAPS']):
        sample_list = list(sample_metadata_df.loc[(sample_metadata_df['status'].isin([status])) & (sample_metadata_df['origin'] == origin)]['samples'])
        ax = axes[k]
        sample_lstm_reads = seq_lstm[np.isin(origins_array,sample_list)]
        for number in list(lstm_encoding_reverse.keys()):
            sns.distplot(np.sum(sample_lstm_reads==number,axis=1),label=lstm_encoding_reverse[number],ax=ax,hist=False,kde=True)

        ax.grid(visible=True)
        ax.set_title(origin)
        ax.legend()
    fig.suptitle(f'Comparing base and mC distributions across {status}',fontsize=24)
    if savename:
        plt.savefig(savename)
        
        
        
def plot_seq_lengths(seq_lengths,origins_array,sample_metadata_df,status='healthy_cfdna',savename=False,hist=False,log=False):
    if hist:
        fig,axes = plt.subplots(3,figsize=(20,16),sharex=True,sharey=False)
    else:
        fig,axes = plt.subplots(3,figsize=(20,16),sharex=True,sharey=True)
        
    for k,origin in enumerate(['Lo','cancer_detector','TAPS']):
        sample_list = list(sample_metadata_df.loc[(sample_metadata_df['status'].isin([status])) & (sample_metadata_df['origin'] == origin)]['samples'])
        ax = axes[k]
        sample_seq_lengths = seq_lengths[np.isin(origins_array,sample_list)]
        
        if log:
            ax.hist(sample_seq_lengths,log=True,density=True,label=origin,bins=100)
        else:
            if hist:
                sns.distplot(sample_seq_lengths,ax=ax,hist=True,kde=False,label=origin,bins=100)
            else:
                sns.distplot(sample_seq_lengths,ax=ax,hist=False,kde=True,label=origin,bins=100)

        ax.grid(visible=True)
        ax.set_title(origin)
        ax.legend()
    fig.suptitle(f'Comparing sequence length across {status}',fontsize=24)
    if savename:
        plt.savefig(savename)