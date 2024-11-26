using Pkg
Pkg.activate(".")
include("../../../LogLoss/RealRealHighDimension.jl")
include("../../../Imputation/imputation.jl");
using JLD2
using DelimitedFiles
using Plots

# load the original split
dloc = "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/FinalBenchmarks/NASA_KeplerV2/kepler_c0_all_folds.jld2"

f = jldopen(dloc, "r")
    folds = read(f, "folds");
close(f)

# load the window locations
f1 = jldopen("/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/FinalBenchmarks/NASA_KeplerV2/kepler_windows_julia_idx.jld2", "r")
    window_idxs = read(f1, "windows_per_percentage")
close(f1)

struct WindowScores
    mps_scores::Vector{Float64}
    nn_scores::Vector{Float64}
end

struct InstanceScores
    pm_scores::Vector{WindowScores}
end

struct TSResults
    instance_scores::Vector{InstanceScores}
end

struct FoldResults
    ts_scores::Vector{TSResults}
end

function run_folds(folds::Vector{Tuple{Vector{Any}, Vector{Any}}}, window_idxs::Dict{Any, Any}, which_folds::UnitRange=1:29)

    # Training parameters
    verbosity = 0
    test_run = false
    track_cost = false
    encoding = :legendre_no_norm
    encode_classes_separately = false
    train_classes_separately = false

    d = 15  # Originally 15
    chi_max = 35  # Originally 35

    opts = MPSOptions(; nsweeps=5, chi_max=chi_max, update_iters=1, verbosity=verbosity, loss_grad=:KLD,
        bbopt=:TSGO, track_cost=track_cost, eta=0.1, rescale=(false, true), d=d, aux_basis_dim=2, encoding=encoding,
        encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately,
        exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
    opts_safe, _... = safe_options(opts, nothing, nothing)

    dx = 5e-3
    mode_range = (-1, 1)
    xvals = collect(range(mode_range...; step=dx))
    mode_index = Index(opts_safe.d)
    xvals_enc = [get_state(x, opts_safe) for x in xvals]
    xvals_enc_it = [ITensor(s, mode_index) for s in xvals_enc]
    num_folds = length(which_folds)
    pms = 0.05:0.10:0.95
    fold_scores = Vector{FoldResults}(undef, num_folds)

    for (i, fold_idx) in enumerate(which_folds)
        ts_scores = Vector{TSResults}(undef, length(folds[1][1]))
        X_train_fold_all = folds[fold_idx][1]
        X_test_fold_all = folds[fold_idx][2]
        # Isolate the train and test set, single instance
        for ts in 1:length(folds[1][1])  # Or ts in 1:length(folds[1][1]) for generality
            # Extract the train and test windows to make a train/test set
            X_train_fold = vcat(X_train_fold_all[ts]'...)
            y_train_fold = zeros(Int64, size(X_train_fold, 1))
            X_test_fold = vcat(X_test_fold_all[ts]'...)
            y_test_fold = zeros(Int64, size(X_test_fold, 1))
            # Fit the MPS
            W, _, _, _ = fitMPS(X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts; chi_init=4,  test_run=false)
            # Begin imputation
            fc = init_imputation_problem(W, X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts_safe; verbosity=0)
            num_instances = size(X_test_fold, 1)
            instance_scores = Vector{InstanceScores}(undef, num_instances)
            for instance in 1:num_instances
                println("Evaluating instance $instance")
                pm_scores = Vector{WindowScores}(undef, length(pms))
                for (ipm, pm) in enumerate(pms)
                    num_wins = length(window_idxs[pm])
                    mps_scores = Vector{Float64}(undef, num_wins)
                    nn_scores = Vector{Float64}(undef, num_wins)
                    @threads for it in 1:num_wins
                        impute_sites = window_idxs[pm][it]
                        ts, pred_err, stats, _ = MPS_impute(fc, 0, instance, impute_sites, :directMedian;
                                invert_transform=true,
                                NN_baseline=true, X_train=X_train_fold, y_train=y_train_fold,
                                n_baselines=1, plot_fits=false, dx=dx, mode_range=mode_range, xvals=xvals,
                                mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
                        mps_scores[it] = stats[:MAE]
                        nn_scores[it] = stats[:NN_MAE]
                    end
                    pm_scores[ipm] = WindowScores(mps_scores, nn_scores)
                end
                instance_scores[instance] = InstanceScores(pm_scores)
            end
            ts_scores[ts] = TSResults(instance_scores)
        end
        fold_scores[i] = FoldResults(ts_scores)
    end
    return fold_scores, opts_safe
end

fscores, opts_safe = run_folds(folds, window_idxs, 1:2)

# first fold, first time series, first test window, 5% missing
mean(fscores[1].ts_scores[1].instance_scores[1].pm_scores[2].mps_scores)
mean(fscores[1].ts_scores[1].instance_scores[1].pm_scores[2].nn_scores)



