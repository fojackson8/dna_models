#!/bin/bash 

#SBATCH -J exp6_models
#SBATCH -A ludwig.prj
#SBATCH -p long 
#SBATCH -o %x.%j_output.out 
#SBATCH -e %x.%j_error.err
#SBATCH -c 10
# Parallel environment settings 

#  For more information on these please see the documentation 

#  Allowed parameters: 

#   -c, --cpus-per-task 

#   -N, --nodes 

#   -n, --ntasks 



  

# Some useful data about the job to help with debugging 

echo "------------------------------------------------" 

echo "Slurm Job ID: $SLURM_JOB_ID" 
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 


## Run with conda env
cd /well/ludwig/users/dyp502

source ~/.bashrc
conda activate cf_taps

# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp1.py
# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp2.py
# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp3.py
# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp4.py
# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp5.py
# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/taps_read_classifier_exp6.py
python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/exp6_get_predictions.py

#### Run this script 
#### sbatch -p gpu_short --gres gpu:1 /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/run_gpu_script1.sh
#### sbatch -p gpu_short --gres gpu:2 /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/run_gpu_script1.sh
#### sbatch -p gpu_long --gres gpu:2 /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/run_gpu_script1.sh

# squeue -u dyp502