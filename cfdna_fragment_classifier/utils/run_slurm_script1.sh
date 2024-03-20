#!/bin/bash 

#SBATCH -J bam2read
#SBATCH -A ludwig.prj
#SBATCH -p short 
#SBATCH -o my-job-%j_output.out 
#SBATCH -e my-job-%j_error.err
#SBATCH -c 4
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

python --version
# python /well/ludwig/users/dyp502/read_classifier/code/sample_processing/TAPS_get_methy_ratio_new_HCC_tumour.py 2
# python /well/ludwig/users/dyp502/read_classifier/code/TAPS_samples_November22/TAPS_get_read_stats_per_chromosome.py
python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/TAPS_bam2read_newdata_newfunction_nov22.py


### Run this job
### cd /well/ludwig/users/dyp502/logs
### sbatch --constraint="skl-compat" /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/run_slurm_script1.sh
### squeue -u dyp502