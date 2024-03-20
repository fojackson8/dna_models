#!/bin/bash 
 
# This script sets up a task array with a step size of one. 
 
#$ -J cram_intersect 
#$ -p test.qc 
#$ --array 0-53:1 
#$ --requeue 
#$ -o %x.%j_output.out 
#$ -e %x.%j_error.err
#$ -c 2

echo `date`: Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}, SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}, SLURM_ARRAY_TASK_STEP=${SLURM_ARRAY_TASK_STEP} 
 


PATH=$PATH:/users/ludwig/cfo155/miniconda2/bin
export PATH

# cd /gpfs3/well/ludwig/users/cfo155/tissueMap/methcalls/meth/
# files=(*_CAPS_CG.merged.filtered.bed.gz) 
# files=$(ls /gpfs2/well/ludwig/users/cfo155/cfDNA/022020_cfDNA/align/merge/{HCC_*,Healthy_*}.sort.md.cram)


## cfDNA
files=(/gpfs2/well/ludwig/users/cfo155/cfDNA/022020_cfDNA/align/merge/{HCC_*,Healthy_*,PDAC_*}.sort.md.cram)
file=${files[$SLURM_ARRAY_TASK_ID]}
echo $SLURM_ARRAY_TASK_ID
echo $file
# file=${files[5]}
filename=$(basename "$file")
echo $filename
sample=${filename%.*.*.*}
echo $sample




## tumour samples
# files=($(ls /gpfs3/well/ludwig/users/cfo155/tissueMap/methcalls/*/Results/*/Alignments/*bam | grep -P "Liver|Pan*"))
# file=${files[$SLURM_ARRAY_TASK_ID]}
# echo $SLURM_ARRAY_TASK_ID
# echo $file
# # file=${files[5]}
# filename=$(basename "$file")
# echo $filename
# sample=${filename%.*}
# echo $sample



###   ------------------------      Intersect cfDNA .cram files            ------------------------------------------------

cd /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bed_files
for bed_file in *.sorted.bed
do
bedname=${bed_file%.*.*}

# bedname=${bed_file%.*}
echo $bedname

samtools view -b -h $file -L $bed_file -T /users/ludwig/cfo155/cfo155/cfDNA/012020_cfDNA/resource/GRCh38_spike_in.fasta > /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bam_intersections/${sample}.${bedname}.bam

done






# ###   ------------------------      Intersect cfDNA .cram files            ------------------------------------------------

# cd /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bed_files
# for bed_file in *.bed
# do
# bedname=${bed_file%.*}
# echo $bedname

# samtools view -b -h $file -L $bed_file -T /users/ludwig/cfo155/cfo155/cfDNA/012020_cfDNA/resource/GRCh38_spike_in.fasta > /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/bam_intersections/${sample}.${bedname}.bam

# done







## Run this job
##### sbatch --array 0-53:1 --constraint="skl-compat" /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/intersect_bam_files.sh
##### sbatch --array 0-77:1 --constraint="skl-compat" /well/ludwig/users/dyp502/read_classifier/read_classifier_2024/code/intersect_bam_files.sh
##### squeue -u dyp502
