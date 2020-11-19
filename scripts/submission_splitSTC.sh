#!/bin/bash

#SBATCH --account=def-kjerbi
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=6

# Load the module:

module load freesurfer/5.3.0
module load python/3.8.0

# set the variables:

export SUBJECTS_DIR=/projects/def-kjerbi/hyruuk/saflow/saflow_anat
source $EBROOTFREESURFER/FreeSurferEnv.sh

echo "Starting run at: `date`"

$HOME/neuropycon/bin/python $HOME/projects/def-kjerbi/hyruuk/saflow/scripts/saflow_splitSTC.py

echo "Program finished with exit code $? at: `date`"
