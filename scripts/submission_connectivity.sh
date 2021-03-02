#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00


# Load the module:

module load freesurfer/5.3.0
module load python/3.8.0

# set the variables:

export SUBJECTS_DIR=/projects/def-kjerbi/hyruuk/saflow/saflow_anat
source $EBROOTFREESURFER/FreeSurferEnv.sh


for i in 04 05 06 07 08 09 10 11 12 13 14 15; do
  srun \
    -N1 \
    --mem=512G \
    --ntasks-per-node=24 \
    --nodes=2 \
    --time=12:00:00 \
    --job-name=saflow_con \
echo "Starting run at: `date`"
$HOME/neuropycon/bin/python $HOME/projects/def-kjerbi/hyruuk/saflow/scripts/saflow_connectivity.py $i

echo "Program finished with exit code $? at: `date`"
done

wait
