#!/bin/bash
#################################################################################################################
# This file is a simple template of sbatch script that was made to help our Hive users create sbatch jobs easier.
# There are a lot of additional tricks and options that you can use(like array, checkpoints etc...)
# Please refer to Hive website or Slurm official guide for more advanced options.
#################################################################################################################
#SBATCH --job-name=fastQC ## Name of your job
#SBATCH --ntasks=20 ## number of cpu's to allocate for a job
#SBATCH --ntasks-per-node=20 ## number of cpu's to allocate per each node
#SBATCH --nodes=1 ## number of nodes to allocate for a job
#SBATCH --mem=2800 ## memory to allocate for your job in MB
#SBATCH --time=1-00:00:00 ## time to allocate for your job in format: DD-HH:MM:SS
#SBATCH --error=fastQC.%J.errors ## stderr file name(The %J will print job ID number)
#SBATCH --output=fastQC.%J.output ## stdout file name(The %J will print job ID number)
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
module load FastQC # Load modules that are required by your program
module load gnu-parallel
#########################################

### Below you can enter your program job command ###

### BASEDIR="/data/home/sstern/hadwan/bioinformatics/Lymphoblast_BD/1st_Batch"
### POSTFIX="/fastQC_output/"
### OUTDIR=$BASEDIR$POSTFIX 
### echo $OUTDIR

## Changing directories to where the fastq files are located
cd ~/bioinformatics/Lymphoblast_BD/1st_batch

## Running FASTQC
fastqc -t 6 *trimmed.fastq.gz

## Moving files to our results directory
mv *fastqc* ../results/fastqc_trimmed/

### parallel "echo {} -o ${OUTDIR}" ::: $BASEDIR/*.fastq.gz
###########################################
## Changing directories to where the fastq files are located
cd ~/bioinformatics/Lymphoblast_BD/2nd_batch

## Running FASTQC
fastqc -t 6 *trimmed.fastq.gz

## Moving files to our results directory
mv *fastqc* ../results/fastqc_trimmed/

###########################################
## Changing directories to where the fastq files are located
cd ~/bioinformatics/Lymphoblast_BD/3rd_batch

## Running FASTQC
fastqc -t 6 *trimmed.fastq.gz

## Moving files to our results directory
mv *fastqc* ../results/fastqc_trimmed/

