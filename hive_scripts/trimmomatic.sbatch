#!/bin/bash
#################################################################################################################
# This file is a simple template of sbatch script that was made to help our Hive users create sbatch jobs easier.
# There are a lot of additional tricks and options that you can use(like array, checkpoints etc...)
# Please refer to Hive website or Slurm official guide for more advanced options.
#################################################################################################################
#SBATCH --job-name=trimmomatic ## Name of your job
#SBATCH --ntasks=1 ## number of cpu's to allocate for a job
#SBATCH --ntasks-per-node=1 ## number of cpu's to allocate per each node
#SBATCH --nodes=1 ## number of nodes to allocate for a job
#SBATCH --mem=2800 ## memory to allocate for your job in MB
#SBATCH --time=1-00:00:00 ## time to allocate for your job in format: DD-HH:MM:SS
#SBATCH --error=trimmomatic.%J.errors ## stderr file name(The %J will print job ID number)
#SBATCH --output=trimmomatic.%J.output ## stdout file name(The %J will print job ID number)
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
module load java # Load modules that are required by your program

#########################################

### Below you can enter your program job command ###

### cd ~/bioinformatics/Lymphoblast_BD/1st_Batch
srun runTRIMMOMATIC.sh
