#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ensemble8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1124485@comp.nus.edu.sg
#SBATCH --partition=long

srun python ~/CS5228/trial8/ensemble8.py
