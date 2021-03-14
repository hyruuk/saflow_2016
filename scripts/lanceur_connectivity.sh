
for subid in 04 05 06 07 08 09 10 11 12 13 14 15
do
for cond in IN OUT
do
sbatch submission_connectivity_simple.sh $subid $cond
done
done
