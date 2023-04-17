#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:15:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

# git clone https://github.com/IP1102/refactoring-toy-example.git

# sh RefactoringMiner --help

# python play.py
# python db_test.py

for file in data/*.csv; do
    echo $refresearch/$file
done