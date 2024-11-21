#!/bin/bash
#PBS -N impute_benchmarks_time_dependent
#PBS -l select=1:ncpus=32:mem=100GB:ompthreads=1
#PBS -l walltime=16:00:00
#PBS -m bea
#PBS -M hugo.stackhouse@sydney.edu.au 
#PBS -V
cd "$PBS_O_WORKDIR"
julia --project=. -p 32 --heap-size-hint=185G Imputation/distributed_benchmark.jl   # run your program, note the heap size limit
# julia --project=. -p 32 --heap-size-hint=185G FinalBenchmarks/ItalyPower/Julia/IPD_Imputation_Bench_distributed.jl   # run your program, note the heap size limit

exit

# qsub -I -l select=1:ncpus=16:mpiprocs=2:ompthreads=1:mem=100GB -l walltime=35:00:00