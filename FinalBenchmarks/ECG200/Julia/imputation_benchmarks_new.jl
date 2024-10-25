using Pkg
Pkg.activate(".")
include("../../../LogLoss/RealRealHighDimension.jl")
include("../../../Interpolation/imputation.jl");
using JLD2
using DelimitedFiles

# load the original ECG200 split
dloc = "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)

# recombine the original train/test splits
Xs = vcat(X_train, X_test)
ys = vcat(y_train, y_test)

# load the resample indices
rs_f = jldopen("FinalBenchmarks/ECG200/Julia/resample_folds_julia_idx.jld2", "r");
rs_fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

# load the window indices
windows_f = jldopen("FinalBenchmarks/ECG200/Julia/windows_julia_idx.jld2", "r");
window_idxs = read(windows_f, "windows_julia")
close(windows_f)

# define structs for the results
struct WindowScores
    mps_scores::Vector{Float64}
    nn_scores::Vector{Float64}
end

struct InstanceScores
    pm_scores::Dict{Int, WindowScores}
end

struct FoldResults 
    fold_scores::Vector{InstanceScores}
end

function run_folds(Xs::Matrix{Float64}, ys::Vector{Int64}, window_idxs::Dict,
        fold_idxs::Dict)

        setprecision(BigFloat, 128)
        Rdtype = Float64

        # training related stuff
        verbosity = 0
        test_run = false
        track_cost = false
        encoding = :legendre_no_norm
        encode_classes_separately = false
        train_classes_separately = false

        d = 10
        chi_max=20
        nsweeps = 5
        eta = 0.5
        opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
            bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
            encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
            exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0)
        opts_safe, _... = safe_options(opts, nothing, nothing)

        num_folds = length(fold_idxs)
        dx = 1e-4
        mode_range=(-1,1)
        xvals=collect(range(mode_range...; step=dx))
        mode_index=Index(opts_safe.d)
        pms = [5]
        
        max_num_wins = maximum([length(window_idxs[pm]) for pm in pms])
        per_pm_each_window_scores_mps = Vector{Float64}(undef, max_num_wins)
        per_pm_each_window_scores_nn = Vector{Float64}(undef, max_num_wins)

        # main loop
        fold_scores = Vector{FoldResults}(undef, num_folds)
        for fold_idx in 0:0
            fold_train_idxs = fold_idxs[fold_idx]["train"]
            fold_test_idxs = fold_idxs[fold_idx]["test"]
            X_train_fold = Xs[fold_train_idxs, :]
            y_train_fold = ys[fold_train_idxs]
            X_test_fold = Xs[fold_test_idxs, :]
            y_test_fold = ys[fold_test_idxs]

            W, _, _, _ = fitMPS(X_train_fold, y_train_fold, X_test_fold, y_test_fold; chi_init=4, opts=opts, test_run=false)
            fc = load_forecasting_info_variables(W, X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts_safe; verbosity=0)
            
            xvals_enc= [get_state(x, opts_safe, fc[1].enc_args) for x in xvals]
            xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];

            println("Finished training, beginning evaluation of imputed values...")
            samps_per_class = [size(f.test_samples, 1) for f in fc]
            all_instances = Vector{InstanceScores}()
            for (i, s) in enumerate(samps_per_class)
                # each class instances
                @threads for inst in 1:s
                    # loop over windows
                    pm_scores = Dict{Int, WindowScores}()
                    for pm in pms
                        num_wins = length(window_idxs[pm])
                        resize!(per_pm_each_window_scores_mps, num_wins)
                        resize!(per_pm_each_window_scores_nn, num_wins)
                        # loop over window iterations
                        for it in 1:num_wins
                            interp_sites = window_idxs[pm][it]
                            stats, _ = any_impute_single_timeseries(fc, (i-1), inst, interp_sites, :directMedian; invert_transform=true, 
                                NN_baseline=true, X_train=X_train_fold, y_train=y_train_fold, 
                                n_baselines=1, plot_fits=false, dx=dx, mode_range=mode_range, xvals=xvals, 
                                mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
                            per_pm_each_window_scores_mps[it] = stats[:MAE]
                            per_pm_each_window_scores_nn[it] = stats[:NN_MAE]
                        end
                        pm_scores[pm] = WindowScores(copy(per_pm_each_window_scores_mps), copy(per_pm_each_window_scores_nn))
                    end
                    instance_scores = InstanceScores(pm_scores)
                    push!(all_instances, instance_scores)
                end
            end
            fold_scores[fold_idx+1] = FoldResults(all_instances)
        end

        return fold_scores
end

out = run_folds(Xs, ys, window_idxs, rs_fold_idxs)
mean_over_all_instances_mps = mean([mean(out[1].fold_scores[inst].pm_scores[5].mps_scores) for inst in 1:100])
mean_over_all_instances_nn = mean([mean(out[1].fold_scores[inst].pm_scores[5].nn_scores) for inst in 1:100])
