#!/bin/bash

#$ -N taps_intersections
#$ -P ludwig.prjc
#$ -q long.qc
#$ -cwd
# -o /well/ludwig/users/dyp502/logs/$JOB_ID.call.log
# -e /well/ludwig/users/dyp502/logs/$JOB_ID.call.err
#$ -pe shmem 8


# Some useful data about the job to help with debugging
echo "------------------------------------------------"
echo "SGE Job ID: $JOB_ID"
echo "SGE Task ID: $SGE_TASK_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "Argument: $@"
echo "------------------------------------------------"

PATH=$PATH:/users/ludwig/cfo155/miniconda2/bin
export PATH


## Run this script
## cd /well/ludwig/users/dyp502/logs
## qsub /well/ludwig/users/dyp502/read_classifier/code/sample_processing/cfTAPS_classifier_preprocessing_bams.sh





# ------------------------------------------------------------------------------------------------------------------------------


## Quick sample of TAPS and CancerDetector BAMs for inspection in IGV - 13/02/2022

# module load samtools/1.8-gcc5.4.0 
# samtools view -s 0.01 -b /well/ludwig/users/dyp502/read_classifier/data/taps_copy/HCC_NHCCLEHO053.sort.md.bam > /well/ludwig/users/dyp502/read_classifier/data/testing_data/TAPS.HCC_NHCCLEHO053.sort.md.SUBSAMPLE.01.bam 
# samtools view -s 0.01 -b /well/ludwig/users/dyp502/read_classifier/data/cancerdetector_copy/HCC1_trimmed.bs.md.bam > /well/ludwig/users/dyp502/read_classifier/data/testing_data/CancerDetector.HCC1_trimmed.bs.md.SUBSAMPLE.01.bam 




# ------------------------------------------------------------------------------------------------------------------------------





# ------------------------------------------------------------------------------------------------------------------------------

base_dir=/well/ludwig/users/dyp502/read_classifier/experiments

# HCC samples
for bed in ${base_dir}/exp1/.*.bed
do
for sample in /well/ludwig/users/dyp502/read_classifier/data/taps_copy/HCC_*.sort.md.bam
do
sample_full_name=${sample##*/}
sample_name=${sample_full_name%%.*}
bed_full_name=${bed##*/}
bed_name=${bed_full_name%.*}

echo $bed_name
echo $sample_name
samtools view -b -h $sample -L $bed > ${base_dir}/exp7/taps_filtered_bams/${sample_name}.${bed_name}.intersections.exp7.bam
done
done


# Healthy samples
for bed in ${base_dir}/exp1/.*.bed
do
for sample in /well/ludwig/users/dyp502/read_classifier/data/taps_copy/Healthy_*.sort.md.bam
do
sample_full_name=${sample##*/}
sample_name=${sample_full_name%%.*}
bed_full_name=${bed##*/}
bed_name=${bed_full_name%.*}

echo $bed_name
echo $sample_name
samtools view -b -h $sample -L $bed > ${base_dir}/exp7/taps_filtered_bams/${sample_name}.${bed_name}.intersections.exp7.bam
done
done
