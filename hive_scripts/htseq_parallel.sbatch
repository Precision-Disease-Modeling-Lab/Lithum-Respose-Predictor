#!/bin/bash
#################################################################################################################
# This file is a simple template of sbatch script that was made to help our Hive users create sbatch jobs easier.
# There are a lot of additional tricks and options that you can use(like array, checkpoints etc...)
# Please refer to Hive website or Slurm official guide for more advanced options.
#################################################################################################################
#SBATCH --job-name=HTseq-parallel ## Name of your job
#SBATCH --ntasks=1 ##number of cpu's to allocate for a job
#SBATCH --cpus-per-task=4
#SBATCH --array=0-127
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=hive7d
#SBATCH --error=log/htset-count.%J.errors ## stderr file name(The %J will print job ID number)
#SBATCH --output=log/htseq-count.%J.output ## stdout file name(The %J will print job ID number)
########### Job information #############
echo "================================"
echo "Start at `date`"
echo "Job id is $SLURM_JOBID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NTASKS processors."
echo "================================"
#########################################

######## Load required modules ##########
. /etc/profile.d/modules.sh # Required line for modules environment to work
#module load Biopython # Load modules that are required by your progr  

module purge
module load Biopython # Load modules that are required by your program
#module load Python/3.9.5-GCCcore-10.3.0
#pip install HTSeq

### Below you can enter your program job command ###

##DATABANK=/data/home/sstern/lmizrahi/RNA-seq/dataset/Homo_sapiens.GRCh38.104.gtf
DATABANK=/data/home/sstern/lmizrahi/RNA-seq/dataset/hg38.knownGene.gtf
INPUT=/data/home/sstern/lmizrahi/RNA-seq/data/BD-TAU/hisat2_hg38
OUTPUT=/data/home/sstern/lmizrahi/RNA-seq/data/BD-TAU/count_hg38


mkdir -p "$OUTPUT"
FILES=("$INPUT"/*.sorted.bam)
# counting number of datasets
echo "Number of samples:" ${#FILES[*]}

#Generate name of current sample
#[$SLURM_ARRAY_TTASK_ID] is the element index. Increases with one per job, i.e. one new sample for every job)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}


SHORTNAME1=$(basename $FILE)
SHORTNAME1=${SHORTNAME1%.sorted.bam}
OUTFILE="$OUTPUT"/"$SHORTNAME1".count


echo "Current input-file name is" $FILE
echo "Current output-file name is" "$OUTFILE"



python /data/home/sstern/lmizrahi/RNA-seq/software/htseq-master/scripts/htseq-count \
	-m union -i gene_id -f bam\
	 $FILE "$DATABANK" > "$OUTFILE"
	

#for file in $FILES; do
#  echo $file
#  SHORTNAME1=$(basename "$file")
#  SHORTNAME1=${SHORTNAME1%.sorted.bam}
#  OUTFILE="$OUTPUT"/"$SHORTNAME1".count
#  echo "$OUTFILE"
#  #./data/home/sstern/lmizrahi/.local/bin/htseq-count -m union -i Name -f bam "$file" "$DATABANK" > "$OUTFILE"
#  python /data/home/sstern/lmizrahi/RNA-seq/software/htseq-master/scripts/htseq-count -m union -i gene_id -f bam "$file" "$DATABANK" > "$OUTFILE"

#  echo "------------=================----------------"
#done

