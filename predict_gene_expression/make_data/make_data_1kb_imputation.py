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


# import torch
# from torch import nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from torch.utils.data import TensorDataset, DataLoader

# import wandb

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
# model_name = "20kb_upstream_downstream_gene_equal_bins."
# data_load_name = model_name 
# data_load_name = "1kb_bins.leave_one_out.test2.no_brain."
save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/dl_model/"
save=False
load=False
normalise=True
remove_brain =False
use_normalised_TPM = False
# save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/"





rna_seq_num1 = 2
rna_seq_num2 = 5
rna_seq_filepaths = ["/well/ludwig/users/qwb368/public_data/GTEx/TPM/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct",
                    "/well/ludwig/users/qwb368/public_data/HPA/rna_tissue_hpa_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/HPA_GTEx_consensus/rna_tissue_consensus_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/FANTOM/rna_tissue_fantom_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/hmC-Seal/GSE144530_RNAseq_RPKM_RefSeq_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/HPA_immune/rna_immune_cell_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/Monaco_immune/rna_immune_cell_monaco_formatted.tsv",
                    "/well/ludwig/users/qwb368/public_data/Schmiedel_immune/rna_immune_cell_schmiedel_formatted.tsv"]
rna_seq_filepath1 = rna_seq_filepaths[rna_seq_num1]
rna_seq_filepath2 = rna_seq_filepaths[rna_seq_num2]
rna_seq_filename1 = re.sub('\..*','', re.sub('/.*/', "", rna_seq_filepath1))
rna_seq_filename2 = re.sub('\..*','', re.sub('/.*/', "", rna_seq_filepath2))
rna_seq1 = pd.read_csv(rna_seq_filepath1,sep='\t')
rna_seq2 = pd.read_csv(rna_seq_filepath2,sep='\t')
rna_seq = pd.merge(rna_seq1, rna_seq2, on=['Name','Description']) ### bc those 2 datasets don't have version numbers



if "." in rna_seq.loc[0,"Name"]:
    rna_seq[['core_gene','version']] = rna_seq['Name'].str.split('.',expand=True)
else:
    rna_seq['core_gene'] = rna_seq['Name']
    rna_seq['version'] = 0
rna_seq = rna_seq.set_index('core_gene')





# tissue_names = [u.lower() for u in tissue_id_map.keys()]
tissue_names = list(tissue_id_map.keys())
rna_seq_df = rna_seq
rna_seq_tissues = list(rna_seq_df.columns)


def clean_string(s):
    return s.split('-')[0]  # Split on hyphens and take the first part

# Create a lowercased version of match_tissue_list for comparison
match_tissue_list_lower = [clean_string(tissue.lower()) for tissue in rna_seq_tissues]

mapping = {}

for tissue in tissue_names:
    tissue_clean = clean_string(tissue.lower())
    for match in match_tissue_list_lower:
        if tissue_clean in match:
            mapping[tissue] = match
            break  # Found a match, go to the next tissue
        else:
            mapping[tissue] = ""  # Default case if no match is found

# print(mapping)



mapping['Neutrophils'] = 'neutrophil'
mapping['B-cells'] =  'naive B-cell' # 'memory B-cell'
mapping['Eosinophils'] = 'eosinophil'
mapping['Monocytes'] = 'classical monocyte'
mapping['NK-cells'] = 'NK-cell'
mapping['CD4-T-cells'] = 'memory CD4 T-cell'
mapping['CD8-T-cells'] = 'memory CD8 T-cell'


# Melt the RNA-seq data to long format
rna_seq_data_long = rna_seq.reset_index().melt(id_vars=['core_gene', 'Name', 'Description'], var_name='tissue', value_name='TPM')

# Inspect the first few rows of the long format data
# rna_seq_data_long.head()

rna_seq_data_long['tissue'].value_counts()

rna_seq_data_long['logTPM'] = np.log1p(rna_seq_data_long['TPM'])


# Calculate the mean TPM for each tissue
tissue_means = rna_seq_data_long.groupby('tissue')['TPM'].mean()

# Normalize the TPM values by dividing by the tissue mean
rna_seq_data_long['Normalized_TPM'] = rna_seq_data_long.apply(lambda row: row['TPM'] / tissue_means[row['tissue']], axis=1)
rna_seq_data_long['log_Normalized_TPM'] = np.log1p(rna_seq_data_long['Normalized_TPM'])


## SAVE TO input_data dir
# rna_seq_data_long.to_csv(f"/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/input_data/rna_seq_data_long_post_processing.csv",
#                         sep='\t',
#                         index=False)









### -----------------    Read in MLML


original_mlml_file = "/gpfs3/well/ludwig/users/cfo155/tissueMap/methcalls/aggregate_tissue/tissue_merged_common_cpg.mincov3.all_meth.txt"
mlml_colnames = list(pd.read_csv(original_mlml_file,sep="\t",nrows=1).columns)
mlml_colnames.insert(2, 'END')
# bedfile_colnames = ["chr", "start", "end", "gene_id", "gene_name", "level", "strand"]
bedfile_colnames = ["chr", "start", "end", "gene_id", "gene_name", "strand"]


# mlml_file = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/data/MLML.tissue_merged_common_cpg.mincov3.all_meth.gencode.protein_coding_genes.10kb_upstream_downstream.with_header.bed"
# mlml_file = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/data/MLML.tissue_merged_common_cpg.mincov3.all_meth.gencode.protein_coding_genes.20kb_upstream_downstream.bed"
# names = bedfile_colnames + mlml_colnames
# mlml = pd.read_csv(mlml_file,
#                 sep='\t',
#                   nrows=100000,
#                 names=names
#                 )




filedir = "/gpfs3/well/ludwig/users/cfo155/tissueMap/methcalls/meth_around_gene/"
filename = "MANE.GRCh38.v1.0.refseq_genomic.gene.flank20000.nwin20.all_meth.txt"


tmp = pd.read_csv(filedir + filename,
                 sep='\t',
#                  nrows=100000,
                 )


# tmp.to_csv(f"/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/input_data/{filename}",
#                         sep='\t',
#                         index=False)

## Load new data from Jingfei
# Function to calculate mean across tissue samples for each epigenetic mark
def calculate_tissue_means(df, tissue_id_map):
    tissue_means = {}

    for tissue, samples in tissue_id_map.items():
        for mark in ['mC', 'hmC', 'umC']:
            relevant_columns = [f"{sample}_{mark}" for sample in samples if f"{sample}_{mark}" in df.columns]
            if relevant_columns:
                tissue_means[f"{tissue}_{mark}"] = df[relevant_columns].mean(axis=1)

    return pd.DataFrame(tissue_means)

# Calculate the means
tissue_mean_df = calculate_tissue_means(tmp, tissue_id_map)
mlml = pd.concat([tmp.iloc[:,0:6],tissue_mean_df],axis=1)




if remove_brain:
    mlml = mlml.drop(columns=[u for u in mlml.columns if 'Brain' in u])

# epigenetic_cols = list(mlml.columns)[10:]
epigenetic_cols = [u for u in mlml.columns if 'mC' in u]


### Bin into 1kb bins
# mlml['gene_id'] = mlml['gene_id'].str.split('.').str[0]





# # Calculate the bin for each methylation site
# # data_sample=mlml
# mlml['bin'] = ((mlml['START'] - mlml['start']) // 1000).astype(int)

if normalise:
    mlml[epigenetic_cols] = mlml[epigenetic_cols]/ mlml[epigenetic_cols].mean()
    

grouped_data = mlml.reset_index(drop=True)

# Group the data by gene_id and bin, then calculate the mean for each group
# grouped_data = data_sample.groupby(['gene_id', 'bin'])[epigenetic_cols].mean().reset_index()
# grouped_data = mlml.groupby(['gene_id', 'bin']).agg({
#     'chr': 'first',
#     'start': 'first',
#     'end': 'first',
#     'gene_name': 'first',
# #     'level': 'first',
#     'strand': 'first',
#     'CHR': 'first',
#     'START': 'first',
#     'END': 'first',
#     **{col: 'mean' for col in epigenetic_cols}
# }).reset_index()


# Calculate the start and end coordinates of each bin
# grouped_data['bin_start'] = grouped_data['start'] + grouped_data['bin'] * 1000
# grouped_data['bin_end'] = grouped_data['bin_start'] + 1000



# grouped_data.to_csv(save_dir +  model_name + f"mlml_grouped_data.csv",sep='\t')
# grouped_data.idx.hist(bins=60)








##    --------------------------------       Make different data formats      ----------------------------------------------------------------

'''
Making different versions of the input data, taking wider or narrower region around gene_body. Wider region means higher context, but also more features,
and potentially more missing genes (due to missing values)
'''

# Contains coordinates of start and end of each data format. Start and End are INCLUSIVE
data_formats_indices = {}

# data_formats_indices['info'] = "Contains coordinates of start and end of each data format. Start and End are INCLUSIVE."
data_formats_indices['gene_body'] = [21,40]
data_formats_indices['5kb'] = [16,45]
data_formats_indices['10kb'] = [11,50]
data_formats_indices['20kb'] = [1,60]





### Add imputation / interpolation functionality

def impute_missing_rows_only_middle(df):
    if 'imputed' not in df.columns:
        df['imputed'] = False
    numerical_columns = df.select_dtypes(include=np.number).columns.drop('idx')
    
    all_idx = range(df['idx'].min(), df['idx'].max() + 1)
    missing_idx = sorted(set(all_idx) - set(df['idx']))
    imputed_rows = []
    
    for idx in missing_idx:
        if (idx - 1) in df['idx'].values and (idx + 1) in df['idx'].values:
            before_row = df[df['idx'] == idx - 1].iloc[0]
            after_row = df[df['idx'] == idx + 1].iloc[0]
            
            imputed_row = {'chr': before_row['chr'], 'start': before_row['end'], 'end': after_row['start'],
                           'gene': before_row['gene'], 'feature': before_row['feature'], 'idx': idx, 'imputed': True}
            
            for col in numerical_columns:
                imputed_row[col] = (before_row[col] + after_row[col]) / 2
            imputed_rows.append(imputed_row)
    
    if imputed_rows:
        df_imputed = pd.concat([df, pd.DataFrame(imputed_rows)], ignore_index=True)
        df_imputed.sort_values(by='idx', inplace=True)
    else:
        df_imputed = df.copy()
        
    return df_imputed












impute = True
total_length = 60
### --------------------        Encode into tensors       --------------------------------

# Initialize the list for storing the 2D arrays for each gene x tissue combination
data_list_new = []

# Initialize the list for storing the gene-tissue combinations
gene_tissue_combinations = []


# tissues = list(np.unique([u.split('_')[0] for u in epigenetic_cols if 'mC' in u]))
tissues = [u for u in healthy_tissue_types if u not in ['Bone-Marrow','Thymus']]

replace_new_blood_cell_names = {"Erythroblasts":"CD34-erythroblasts",
                           "Megakaryocytes":"CD34-megakaryocytes"}
tissues = [replace_new_blood_cell_names.get(u) if u in list(replace_new_blood_cell_names.keys()) else u for u in tissues  ]

if remove_brain:
    tissues = [u for u in tissues if u not in ['Brain']]
marks = ['mC','hmC','umC']

# Initialize the list for storing the 2D arrays for each gene x tissue combination
data_list_new = []

# Initialize the list for storing the gene-tissue combinations
gene_tissue_combinations = []


data_save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/data/"
model_save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/results/"

skip_counts = {}

for impute in [True,False]:
    for data_format in data_formats_indices.keys():
        
        start = data_formats_indices[data_format][0]
        end = data_formats_indices[data_format][1]
        required_length = 1+ end - start
        
        
        print(f"Data format is {data_format}. Checking that gene_data between {start} and {end} has length {required_length}")
        skip_counts[data_format] = 0
        
        # Loop over each gene
        gene_shapes = collections.defaultdict(list)
        for gene_id in grouped_data['gene'].unique():
            # Filter the data for the current gene
            gene_data = grouped_data[grouped_data['gene'] == gene_id]
            gene_data = gene_data.sort_values(by="idx")
        #     assert len(gene_data['strand'].unique())==1
        #     strand = gene_data['strand'].unique()[0]





            # Impute if necessary 
            if gene_data.shape[0]<total_length and impute:
                # print(f"Imputing missing values for gene_data with shape {gene_data.shape}")
                gene_data = impute_missing_rows_only_middle(gene_data)

            # Reduce gene_data to necessary size for the specific data format
            gene_data = gene_data.loc[(gene_data['idx']>=start)&(gene_data['idx']<=end)]

            # If still missing values after imputation, then don't include this gene
            if gene_data.shape[0]<required_length:
                skip_counts[data_format] += 1
                # print(f"For gene {gene_id}, gene_data shape is {gene_data.shape[0]}; needs to be {required_length}. Skipping this gene")
                continue


            gene_shapes['gene_id'].append(gene_id)
            gene_shapes['rows'].append(gene_data.shape[0])
            gene_shapes['cols'].append(gene_data.shape[1])
            gene_shapes['genebody_rows'].append(gene_data.loc[gene_data['feature']=="genebody"].shape[0])
            # Loop over each tissue
            for tissue in tissues:
                if tissue in mapping.keys():
                    # Get the data for all marks at once and convert it to a 2D array
                    tissue_data = gene_data[[f'{tissue}_{mark}' for mark in marks]].T.values

                    # print(tissue_data.shape)
        #             if strand =='-':
        #                 tissue_data = np.flip(tissue_data,axis=1)
                    # If the data is not complete (i.e., the gene does not have 20 bins), skip this gene x tissue combination

        #             if tissue_data.shape[1] < 60:
        #                 continue

                    # if tissue_data.shape[1] < required_length:
                    #     print(f"For gene {gene_id} and tissue {tissue}, tissue_data shape is {tissue_data.shape[1]}; needs to be {required_length}. Skipping this gene")
                    #     break



                    if tissue_data.shape[1] < required_length:
                        print(f"For gene {gene_id} and tissue {tissue}, tissue_data shape is {tissue_data.shape[1]}; needs to be {required_length}. Skipping this gene")
                        continue
                    # Add the 2D array to the list
                    data_list_new.append(tissue_data)

                    # Add the gene-tissue combination to the list
                    gene_tissue_combinations.append((gene_id, tissue, mapping[tissue]))

        # Convert the list of gene-tissue combinations to a dataframe
        gene_tissue_combinations_df = pd.DataFrame(gene_tissue_combinations, columns=['core_gene', 'tissue','rna_seq_tissue']) 


        gene_tissue_combinations_df = gene_tissue_combinations_df.reset_index().rename(columns={"index":"original_index"})
        print(f"shape of gene_tissue_combinations_df is {gene_tissue_combinations_df.shape}")

        print(f"shape of data_list_new is {len(data_list_new)}")
        assert len(gene_tissue_combinations_df) == len(data_list_new)



        # data_load_name = "1kb_bins.leave_one_out.test11.remove_zeros."
        # pd.DataFrame(gene_shapes).to_csv(save_dir +  model_name + f"gene_shapes.csv",sep='\t')



        gene_tissue_combinations_df.to_csv(data_save_dir + f"epigenetic_input.pre_merge.{data_format}.impute_{impute}.csv",sep='\t')
    





# ##    --------------------------------       Merge with RNA-seq      ----------------------------------------------------------------

        # Merge the RNA-seq data with the gene-tissue combinations dataframe to get the corresponding TPM values
        rna_seq_values_df = gene_tissue_combinations_df.merge(rna_seq_data_long, how='inner', left_on=['core_gene', 'rna_seq_tissue'],right_on=['Description', 'tissue'])

        # rna_seq_values_df.to_csv(save_dir +  model_name + f"rna_seq_values_df_after_merging.csv",sep='\t')

        # rna_seq_values_df = rna_seq_values_df.reset_index(drop=True)
        print(f"shape of rna_seq_values_df is {rna_seq_values_df.shape}")

        # rna_seq_values_df = rna_seq_values_df.drop_duplicates()
        # print(f"shape of rna_seq_values_df after dropping duplicates {rna_seq_values_df.shape}")

        # Get indices of genes with no RNA_seq TPM value
        ## If we remove NaNs, then this tends to remove whole genes, thus keeping the number tissue_gene sets intact. So each gene still has all 20 tissues associated.
        ## However, if we filter by logTPM, we remove certain tissues within genes, so some genes will only have eg 15 nonzero tissues associated with it. This causes batch_size errors

        rna_seq_notna_indices = rna_seq_values_df['TPM'].notna()
        # rna_seq_notna_indices = rna_seq_values_df.loc[(rna_seq_values_df['TPM'].notna()) & (rna_seq_values_df['logTPM']!=0)]
        # rna_seq_notna_indices = (rna_seq_values_df['TPM'].notna()) & (rna_seq_values_df['logTPM']!=0)

        print(f"After filtering out genes without RNA-seq data, keeping {rna_seq_notna_indices.sum()} of {rna_seq_values_df.shape} total")
        rna_seq_values_df = rna_seq_values_df[rna_seq_notna_indices]
        rna_seq_values_df = rna_seq_values_df.reset_index(drop=True)
        print(f"shape of rna_seq_values_df after filtering by rna_seq_notna_indices is {rna_seq_values_df.shape}")


        ### Also drop all rows where logTPM = 0
        # rna_seq_not_zero_indices = rna_seq_values_df['logTPM']!=0
        # print(f"After filtering out genes where logTPM > 0 , keeping {rna_seq_not_zero_indices.sum()} of {rna_seq_values_df.shape} total")
        # rna_seq_values_df = rna_seq_values_df[rna_seq_not_zero_indices]


        index_for_gene_tissue_combinations = list(rna_seq_values_df['original_index'])

        # gene_tissue_combinations_df = gene_tissue_combinations_df[rna_seq_notna_indices]
        gene_tissue_combinations_df = gene_tissue_combinations_df.loc[index_for_gene_tissue_combinations,:]

        print(f"shape of gene_tissue_combinations_df after filtering by rna_seq_notna_indices is {gene_tissue_combinations_df.shape}")

        # Convert the TPM values to a numpy array
        # rna_seq_values = rna_seq_values_df['TPM'].values
        if use_normalised_TPM:
            rna_seq_values = rna_seq_values_df['Normalized_TPM'].values
        else:
            rna_seq_values = rna_seq_values_df['logTPM'].values

        # Stack the 2D arrays vertically to get a 3D tensor
        data_tensor_new = np.stack(data_list_new)

        print(f"shape of data_tensor_new is {data_tensor_new.shape}")


        # Drop NaNs from rna_seq
        data_tensor_filtered = data_tensor_new[index_for_gene_tissue_combinations,:,:]

        print(f"shape of data_tensor_filtered after filtering by rna_seq_notna_indices is {data_tensor_filtered.shape}")
        print(f"shape of rna_seq_values is {rna_seq_values.shape}")


        # Print the shapes of the tensor and the target array
        # data_tensor_filtered.shape, rna_seq_values.shape



        np.save(data_save_dir +  f"epigenetic_rna_seq_merged.{data_format}.impute_{impute}.data_tensor_filtered.npy",data_tensor_filtered)
        np.save(data_save_dir +  f"epigenetic_rna_seq_merged.{data_format}.impute_{impute}.rna_seq_values.npy",rna_seq_values)
        rna_seq_values_df.to_csv(data_save_dir +  f"epigenetic_rna_seq_merged.{data_format}.impute_{impute}.gene_tissue_combinations_rnaseq.csv",sep='\t')

        print(f"Saving training array to {data_save_dir + f'epigenetic_rna_seq_merged.{data_format}.impute_{impute}.data_tensor_filtered.npy'}")
        print(f"Saving target array to {data_save_dir +  f'epigenetic_rna_seq_merged.{data_format}.impute_{impute}.rna_seq_values.npy'}")
        # print(f"Saving data array to {save_dir + model_name} rna_seq_values.npy")
        # data_tensor_filtered.save(save_dir + "data_tensor_filtered.npy")
        # rna_seq_filtered.save(save_dir + "rna_seq_filtered.npy")

        # if load:
        #     data_tensor_filtered = np.load(save_dir +  data_load_name +  "data_tensor_filtered.npy")
        #     rna_seq_values = np.load(save_dir + data_load_name +  "rna_seq_values.npy")
        #     rna_seq_values_df = pd.read_csv(save_dir +  data_load_name + f"gene_tissue_combinations_rnaseq.csv",sep='\t')







    skip_counts_df = pd.DataFrame(list(skip_counts.items()), columns=['Data Format', 'Skip Count'])
    skip_counts_df.to_csv(data_save_dir + f"epigenetic_input_STATS.pre_merge.{data_format}.impute_{impute}.csv",sep='\t')

