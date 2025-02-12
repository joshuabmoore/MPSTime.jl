using MPSTime
using JLD2
using Distributed
using Optimization
using OptimizationBBO
using Random
# using OptimizationMetaheuristics
# using OptimizationOptimJL
# using OptimizationNLopt
# using OptimizationOptimisers

Random.seed!(1)
@load "test/Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

params = (
    eta=(-3,1), 
    d=(10,20), 
    chi_max=(20,50),
    nsweeps=(2,8)
,) 
e = copy(ENV)
e["OMP_NUM_THREADS"] = "1"
e["JULIA_NUM_THREADS"] = "1"

addprocs(5; env=e, exeflags="--heap-size-hint=2G", enable_threaded_blas=false)
@everywhere using MPSTime, Distributed, Optimization, OptimizationBBO

rs_f = jldopen("Folds/IPD/ipd_resample_folds_julia_idx.jld2", "r");
fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

@load "Folds/IPD/ipd_windows_julia_idx.jld2" windows_julia
folds = [(fold_idxs[i-1]["train"], fold_idxs[i-1]["test"]) for i in 1:30]

res = evaluate(
    vcat(X_train, X_test), 
    vcat(y_train, y_test), 
    params,
    BBO_random_search(); 
    objective=ImputationLoss(), 
    opts0=MPSOptions(; verbosity=-5, log_level=-1, nsweeps=5), 
    nfolds=5, 
    n_cvfolds=5,
    eval_windows=windows_julia,
    tuning_windows = nothing,
    tuning_pms=collect(5:20:95) ./100,
    tuning_abstol=1e-8, 
    tuning_maxiters=3,
    verbosity=2,
    foldmethod=folds,
    input_supertype=Float64,
    provide_x0=false,
    logspace_eta=true,
    distribute_folds=true)

@save "IPD_gen_opt.jld2" res
# 20 iter benchmarks 


# SA()
# t=697.29: training MPS with ((chi_max = 22, d = 2, eta = 1.7493084678516522))...  done
# t=697.97: Loss 0.17647058823529416

#t=2896, chim 30, d=2, eta=1.26
# 0.0303

# PSO()
# retcode: Default
# u: [20.456448963449063, 9.853361431316502, 3.6032277648473126]
# Final objective value:     0.02941176470588236

#  retcode: Default
# u: [20.5526603176487, 9.815468744903043, 3.5211517956867766]
# Final objective value:     0.02941176470588236


# BBO_adaptive_de_rand_1_bin()
# t=81.99: training MPS with ((chi_max = 29, d = 10, eta = 2.431916965383183))...  done
# t=88.85: Loss 0.02941176470588236

# u: [20.68733736874209, 7.282024263293762, 3.3227249898786315]
# Final objective value:     0.02941176470588236

#  retcode: MaxIters
# u: [21.794199781220755, 8.945887613181625, 0.8267693406424581]
# Final objective value:     0.030303030303030276
