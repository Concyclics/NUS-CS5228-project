#!/bin/sh
#SBATCH --time=1600
#SBATCH --job-name=ensemble6
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1124485@comp.nus.edu.sg
#SBATCH --partition=long

srun python ~/CS5228/trial6/ensemble6.py
