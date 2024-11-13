#!/bin/bash
#PBS -N impute_benchmarks_rewrite_IPD
#PBS -l select=1:ncpus=16:mem=100GB
#PBS -l walltime=80:00:00
#PBS -m bea
#PBS -M hugo.stackhouse@sydney.edu.au 
#PBS -V
cd "$PBS_O_WORKDIR"
julia --project=. --threads=auto instantiate.jl # ensure libraries according to manifest
julia --project=. --threads=auto --heap-size-hint=100G FinalBenchmarks/ItalyPower/Julia/IPD_Imputation_Benchmarks_save_trajectories.jl   # run your program, note the heap size limit

exit

# qsub -I -l select=1:ncpus=16:mpiprocs=2:ompthreads=1:mem=120GB -l walltime=60:00:00