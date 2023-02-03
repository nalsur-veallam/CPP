#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name="md-code"
#SBATCH --output=_slurm-out.txt
#SBATCH --error=_slurm-error.txt
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=16
#SBATCH --comment="md course 2022"

N=10000

for p in 1 2 4 8 16 48
do
./a.out $N 10 $p
done

N=20000

for p in 1 2 4 8 16 
do
./a.out $N 10 $p
done

N=30000

for p in 1 2 4 8 16 
do
./a.out $N 10 $p
done
