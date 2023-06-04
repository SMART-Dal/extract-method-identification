#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:15:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL
#SBATCH --signal=B:USR1@180

output_file_name=file_0002.jsonl
input_file_name=file_0002.csv
echo "Start"

handle_signal() 
{
    echo 'Trapped - Moving Files'
    rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/model_checkpoint.h5 $refresearch/data/output
    rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/loss_curves.csv $refresearch/data/output
    exit 0
}

trap 'handle_signal' SIGUSR1

echo 'Moving Data File'
rsync -axvH --no-g --no-p $refresearch/data/output/file_0001.jsonl $SLURM_TMPDIR/

cd $SLURM_TMPDIR
git clone git@github.com:SMART-Dal/extract-method-identification.git

cd ./extract-method-identification

git checkout ref-instance-fix

module purge

module load python/3.10

python -m venv res_venv
# cp $refresearch/res_venv .
source res_venv/bin/activate
pip install -r requirements.txt

# -u is for unbuffered output so the print statements print it to the slurm out file
# & at the end is to run the script in background. Unless it's running in background we can't trap the signal
python -u $SLURM_TMPDIR/extract-method-identification/deep-learning/autoencoder.py $SLURM_TMPDIR/ &  

PID=$!
wait ${PID}

echo "Python Script execution over. Attempting to copy the output file..."

rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/model_checkpoint.h5 $refresearch/data/output
rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/loss_curves.csv $refresearch/data/output

echo "Completed data collection process."


