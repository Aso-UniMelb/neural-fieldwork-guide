#!/bin/bash

#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:1
#SBATCH --qos=gpgpudeeplearn

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

#SBATCH --job-name="FieldworkGuide"

#SBATCH --account="punim0478"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Use this email address:
#SBATCH --mail-user=aso.mahmudi@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-5:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module purge
module load foss/2022a
module load PyTorch/1.12.1-CUDA-11.7.0
module load tqdm/4.64.0

# The job command(s):
bash guide/fieldwork_guide.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s