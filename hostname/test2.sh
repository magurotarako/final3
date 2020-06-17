#!/bin/bash
#PBS -l nodes=2:ppn=20

cd $PBS_O_WORKDIR

mpirun -np 40 ./a.out
