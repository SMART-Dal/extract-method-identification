#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=7:30:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL
#SBATCH --signal=B:USR1@180
#SBATCH --gpus-per-node=v100l:1

# input_file_name=file_0001.jsonl
# echo "Start AE for Positive only cases with 5k Training Samples"
# echo "Start LR with AE"
echo "Start LR"

handle_signal() 
{
    echo 'Trapped - Moving Files'
    exit 0
}

trap 'handle_signal' SIGUSR1

# echo 'Moving Data File'
# rsync -axvH --no-g --no-p $refresearch/data/output/$input_file_name $SLURM_TMPDIR/

module purge
module load python/3.10

# python -m venv res_venv
# cp $refresearch/res_venv .
source res_venv/bin/activate
# pip install -r requirements.txt

# -u is for unbuffered output so the print statements print it to the slurm out file
# & at the end is to run the script in background. Unless it's running in background we can't trap the signal
cd $refresearch/deep-learning/
# python -u autoencoder_po.py &  
# python -u classify.py &  
python -u classify_baseline.py &  

PID=$!
wait ${PID}

# echo "Python Script execution over. Attempting to copy the output file..."

# rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/model_checkpoint.h5 $refresearch/data/output
# rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/deep-learning/loss_curves.csv $refresearch/data/output

echo "Completed LR process."


