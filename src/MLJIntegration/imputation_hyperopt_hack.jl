include("../Imputation/imputation.jl")
import MLJTuning 
import ProgressMeter



function revise_history!(Xs_train::Matrix, ys_train::Vector, impute_sites::Vector, max_ts_per_class::Integer, history)
    n_maes = length(history) * length(history[1].evaluation.fitted_params_per_fold)
    p = ProgressMeter.Progress(n_maes,
        dt = 0,
        desc = "Evaluating Interp error of $n_maes models:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :green
    )    
    ProgressMeter.update!(p,0)


    means = Vector{Float64}(undef, length(history))

    if !(history[1].evaluation isa PerformanceEvaluation)
        error("compact_history must be set to false to use the imputation_hyperopt hack!")
    end

    for (hi, h) in enumerate(history)
        e = h.evaluation
        nfolds = length(e.fitted_params_per_fold)
        f_per_fold = zeros(Float64, nfolds)
        for fold in 1:nfolds
            # retrieve model and train/test data
            dec, mopts, W = e.fitted_params_per_fold[fold][1]
            # @show hi, fold, W[1][1], W[1][2]
            # @show mopts.eta, mopts.chi_max
            opts, _... = safe_options(mopts, nothing, nothing) # make sure options is abstract

            train_idxs, val_idxs = e.train_test_rows[fold]
            Xs_train_fold, ys_train_fold = Xs_train[train_idxs, :], ys_train[train_idxs]
            Xs_val_fold, ys_val_fold = Xs_train[val_idxs, :], ys_train[val_idxs]


            # precompute encoded data for computational speedup
            fc = init_imputation_problem(W, Xs_train_fold, ys_train_fold, Xs_val_fold, ys_val_fold, opts; verbosity=-1);


            n_c1s = sum(ys_val_fold)
            n_c0s = length(ys_val_fold) - n_c1s

            n0max =  min(max_ts_per_class, n_c0s)
            n1max =  min(max_ts_per_class, n_c1s)

            classes = [zeros(Int,n0max); ones(Int,n1max)]
            sample_idxs = [shuffle(MersenneTwister(fold), 1:n_c0s)[1:n0max]; shuffle(MersenneTwister(fold), 1:n_c1s)[1:n1max]]
            #sample_idxs = [1:n0max; 1:n1max]

            n_ts = length(classes)
            # @show hi, fold, train_idxs[1:5], sample_idxs[1:5]
            @sync @simd for j in 1:n_ts
                Threads.@spawn begin 
                    class = classes[j]
                    sample = sample_idxs[j]

                    ts, pred_err, stats, _ = MPS_impute(fc, class, sample, impute_sites, :directMedian; NN_baseline=false, plot_fits=false);
                    f_per_fold[fold] += stats[:MAE]
                    # @show stats
                    # @show f_per_fold[fold]
                end
            end
            f_per_fold[fold] /= n_ts
            p.counter += 1
            ProgressMeter.updateProgress!(p)
        end
        e.per_fold[1] .= f_per_fold

        h.per_fold[1] .= f_per_fold
        for u in 2:length(e.per_fold)
            e.per_fold[u] .= zeros(nfolds)
            h.per_fold[u] .= zeros(nfolds)
        end
        
        mf = mean(f_per_fold)
        e.measurement[1] = mf
        h.measurement[1] = mf

        means[hi] = mf
    end
    return means
end
