#!/bin/bash
#################################################################################################################
# This file is a simple template of sbatch script that was made to help our Hive users create sbatch jobs easier.
# There are a lot of additional tricks and options that you can use(like array, checkpoints etc...)
# Please refer to Hive website or Slurm official guide for more advanced options.
#################################################################################################################
#SBATCH --job-name=HiSAT2 ## Name of your job
#SBATCH --ntasks=20 ## number of cpu's to allocate for a job
#SBATCH --nodes=20 ## number of nodes to allocate for a job
#SBATCH --mem=8G ## memory to allocate for your job in MB
#SBATCH --time=1-00:00:00 ## time to allocate for your job in format: DD-HH:MM:SS
#SBATCH --error=HiSAT2.%J.errors ## stderr file name(The %J will print job ID number)
#SBATCH --output=HiSAT2.%J.output ## stdout file name(The %J will print job ID number)
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
module load samtools # Load modules that are required by your program
module load libs/glibc-2.14.1
module load libsndfile/1.0.28-intel-2017.02
#########################################

### Below you can enter your program job command ###

HISAT2="$HOME"/RNA-seq/software/hisat2-2.2.1/hisat2
DATABANK="$HOME"/RNA-seq/index/grch38/genome
INPUT="$HOME"/RNA-seq/data/BD
OUTPUT="$HOME"/RNA-seq/data/BD/hisat2

mkdir -p "$OUTPUT"
FILES="$INPUT"/*.fastq.gz

for file in $FILES; do
  echo $file
  SHORTNAME1=$(basename "$file")
  SHORTNAME1=${SHORTNAME1%.fastq.gz}
  echo $SHORTNAME1
  echo "$OUTPUT"/${SHORTNAME1%.fastq.gz}.sam
  "$HISAT2" -p 4 -x "$DATABANK" -U "$file" -S "$OUTPUT"/"$SHORTNAME1".sam
  ##samtools view -b -S "$OUTPUT"/"$SHORTNAME1".sam > "$OUTPUT"/"$SHORTNAME1".bam
  ##samtools sort "$OUTPUT"/"$SHORTNAME1".bam -o "$OUTPUT"/"$SHORTNAME1".sorted.bam
  ##samtools index "$OUTPUT"/"$SHORTNAME1".sorted.bam
  echo " ----====----- "
done


  


