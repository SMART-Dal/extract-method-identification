#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:0:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL


function handle_signal {
    echo 'Moving File'
    rsync -avz --remove-source-files $SLURM_TMPDIR/extract-method-identification/data/output/test.jsonl $refresearch/data/output
}

trap handle_signal SIGUSR1


cd $SLURM_TMPDIR
git clone git@github.com:SMART-Dal/extract-method-identification.git

cd ./extract-method-identification

git checkout ref-instance-fix

module purge

module load java/17.0.2
module load python/3.10

export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"
export PATH=$PATH:$SLURM_TMPDIR/extract-method-identification/executable/RefactoringMiner/bin

python -m venv res_venv
# cp $refresearch/res_venv .
source res_venv/bin/activate
pip install -r requirements.txt

python data_collector.py ./data/file_0000.csv file_0000.jsonl

