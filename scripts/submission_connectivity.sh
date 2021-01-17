#!/bin/bash
for subid in 04 05 06 07 08 09 10 11 12 13 14
do
#SBATCH --account=def-kjerbi
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --job-name=s$subid_saflow_con
#SBATCH --ntasks-per-node=24

# Load the module:

module load freesurfer/5.3.0
module load python/3.8.0

# set the variables:

export SUBJECTS_DIR=/projects/def-kjerbi/hyruuk/saflow/saflow_anat
source $EBROOTFREESURFER/FreeSurferEnv.sh

echo "Starting run at: `date`"

$HOME/neuropycon/bin/python $HOME/projects/def-kjerbi/hyruuk/saflow/scripts/saflow_connectivity.py $subid

echo "Program finished with exit code $? at: `date`"
done
