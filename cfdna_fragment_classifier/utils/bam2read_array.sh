#!/bin/bash 
 
# This script sets up a task array with a step size of one. 
 
#$ -J cram_intersect 
#$ -p test.qc 
#$ --array 0-120:1 
#$ --requeue 
#$ -o %x.%j_output.out 
#$ -e %x.%j_error.err
#$ -c 2

echo `date`: Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}, SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}, SLURM_ARRAY_TASK_STEP=${SLURM_ARRAY_TASK_STEP} 
 


## Run with conda env
cd /well/ludwig/users/dyp502

source ~/.bashrc
conda activate cf_taps

python --version


# python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/TAPS_bam2read_newdata_newfunction_nov22.py ${SLURM_ARRAY_TASK_ID} 
python /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/encode_new_TAPS_read_files.py ${SLURM_ARRAY_TASK_ID} 



## Run this job
##### sbatch --array 0-487:1 --constraint="skl-compat" /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/bam2read_array.sh
##### squeue -u dyp502
