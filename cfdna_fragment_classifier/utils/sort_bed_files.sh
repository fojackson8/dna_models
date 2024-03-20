#!/bin/bash 

#SBATCH -J sort_beds
#SBATCH -A ludwig.prj
#SBATCH -p short 
#SBATCH -o my-job-%j_output.out 
#SBATCH -e my-job-%j_error.err
#SBATCH -c 4
# Parallel environment settings 





  

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

cd /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bed_files

for bed_file in *.bed
do
bedname=${bed_file%.*}
echo $bedname
sort -k1,1 -k2,2n $bed_file > ${bedname}.sorted.bed
done

### Run this job
### cd /well/ludwig/users/dyp502/logs
### sbatch --constraint="skl-compat" /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/sort_bed_files.sh
### squeue -u dyp502