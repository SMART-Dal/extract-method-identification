#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:0:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL
#SBATCH --signal=B:USR1@180

# output_file_name=file_0002.jsonl
# input_file_name=file_0002.csv

output_file_name=test.jsonl
input_file_name=test.csv

echo "Start"

handle_signal() 
{
    echo 'Trapped - Moving File'
    rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/data/output/$output_file_name $refresearch/data/output
    exit 0
}

trap 'handle_signal' SIGUSR1


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

# -u is for unbuffered output so the print statements print it to the slurm out file
# & at the end is to run the script in background. Unless it's running in background we can't trap the signal
python -u data_collector.py ./data/$input_file_name $output_file_name &  

PID=$!
wait ${PID}

echo "Python Script execution over. Attempting to copy the output file..."

rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-identification/data/output/$output_file_name $refresearch/data/output

echo "Completed data collection process."


