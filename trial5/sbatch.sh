#!/bin/sh
#SBATCH --time=1600
#SBATCH --job-name=ensemble5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1124485@comp.nus.edu.sg
#SBATCH --partition=long

srun python ~/CS5228/trial5/ensemble5.py
