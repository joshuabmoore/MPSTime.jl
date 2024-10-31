#!/bin/bash
#PBS -N impute_benchmarks_test
#PBS -l select=1:ncpus=16:mem=165GB
#PBS -l walltime=80:00:00
#PBS -m bea
#PBS -M hugo.stackhouse@sydney.edu.au 
#PBS -V
cd "$PBS_O_WORKDIR"
julia --project=. --threads=auto instantiate.jl # ensure libraries according to manifest
julia --project=. --threads=auto --heap-size-hint=160G FinalBenchmarks/ECG200/Julia/imputation_benchmarks_new_SL.jl   # run your program, note the heap size limit

exit
