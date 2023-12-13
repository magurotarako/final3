#!/bin/bash
#PBS -l nodes=1:ppn=20

cd $PBS_O_WORKDIR
hostname
mpirun -np 20 ./cps/cps vsAI.sh