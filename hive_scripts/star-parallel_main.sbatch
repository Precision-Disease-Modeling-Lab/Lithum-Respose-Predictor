#!/bin/bash
#################################################################################################################
# This file is a simple template of sbatch script that was made to help our Hive users create sbatch jobs easier.
# There are a lot of additional tricks and options that you can use(like array, checkpoints etc...)
# Please refer to Hive website or Slurm official guide for more advanced options.
#################################################################################################################
#SBATCH --job-name=STAR-parallel ## Name of your job
#SBATCH --ntasks=5 ##number of cpu's to allocate for a job
#SBATCH --array=0-64
#SBATCH --mem-per-cpu=64GB
#SBATCH --partition=hive7d
#SBATCH --time=1-00:00:00 ## time to allocate for your job in format: DD-HH:MM:SS
#SBATCH --error=log/STAR.%J.errors ## stderr file name(The %J will print job ID number)
#SBATCH --output=log/STAR.%J.output ## stdout file name(The %J will print job ID number)
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
 
module purge
module load GCC/10.3.0
module load samtools 

### Below you can enter your program job command ###

ANNOTATION=/data/home/sstern/lmizrahi/RNA-seq/dataset/Homo_sapiens.GRCh38.104.gtf
GENOME=/data/home/sstern/lmizrahi/RNA-seq/dataset/ENSEMBL-GENOME/star-index/
INPUT=/data/home/sstern/lmizrahi/RNA-seq/data/BD
OUTPUT=/data/home/sstern/lmizrahi/RNA-seq/data/BD/star-mapped/


mkdir -p "$OUTPUT"
FILES=("$INPUT"/*.fastq.gz)
# counting number of datasets
echo "Number of samples:" ${#FILES[*]}

#Generate name of current sample
#[$SLURM_ARRAY_TTASK_ID] is the element index. Increases with one per job, i.e. one new sample for every job)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

SHORTNAME1=$(basename $FILE)
#SHORTNAME1=${SHORTNAME1%.sorted.bam}
OUTFILE="$OUTPUT"/"$SHORTNAME1".count

echo "Current input-file name is" $FILE

cd /data/home/sstern/lmizrahi/RNA-seq/software/STAR

./STAR \
    --genomeLoad NoSharedMemory \
    --runThreadN 5 \
    --genomeDir "$GENOME" \
    --readFilesIn $FILE \
    --readFilesCommand zcat \
    --outFileNamePrefix "$OUTPUT"/$SHORTNAME1 \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMunmapped Within \
    --outSAMattributes Standard \
    --quantMode TranscriptomeSAM GeneCounts \
    --sjdbGTFfile "$ANNOTATION"


