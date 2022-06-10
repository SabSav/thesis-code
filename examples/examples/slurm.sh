#!/bin/bash
#SBATCH -p htc-el8 # partiion
#SBATCH -J ising # Job name
#SBATCH -N 1 # Number of nodes
#SBATCH -n 2 # Number of cpus in node
#SBATCH --mem 8G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm/%j.log
#SBATCH -e slurm/%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sabrina.savino@embl.de

# Prologue
echo Working at `hostname`:`pwd` \(`date`\)
SECONDS=0
module purge
source /g/erzberger/savino/miniconda3/etc/profile.d/conda.sh # NB!
conda activate

python examples/3_spins/run.py

# Epilogue
duration=$SECONDS
days=$((duration / 86400))
hours=$((duration % 86400 / 3600))
minutes=$((duration % 3600 / 60))
seconds=$((duration % 60))
echo "Elapsed time: $days-$hours:$minutes:$seconds"
