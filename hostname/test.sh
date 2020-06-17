#!/bin/bash
#PBS -l nodes=1:ppn=20

cd $PBS_O_WORKDIR

mpirun -np 20 ./a.out
