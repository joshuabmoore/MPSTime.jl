function compute_missing_windows(pm::Real, num_samples::Int, num_windows::Int, seed::Int)
    rng = MersenneTwister(seed)
    num_missing = round(Int, pm * num_samples)

    max_index = num_samples - num_missing + 1
    starting_inds = randperm(rng, max_index)[1:min(num_windows, max_index)]

    return [collect(i:(i+num_missing-1)) for i in starting_inds]
end

function impute_instance_MLJ(instance::Integer, num_instances::Integer, fold_idx::Integer, pms::AbstractVector, max_windows::Int, imp::ImputationProblem)
    bt = time()
    # loop over windows
    pm_scores = Vector{WindowScores}(undef, length(pms))
    for pm in pms
        # loop over window iterations
        missing_vals = compute_missing_windows(pm, length(imp.y_train), max_windows, fold_idx)
        num_wins = length(missing_vals)
        mps_scores = Vector{Float64}(undef, num_wins)
        for it in 1:num_wins
            impute_sites = window_idxs[pm][it]
            ts_ecg, _, _, stats, _ = MPS_impute(imp, 0, instance, impute_sites, :directMedian; invert_transform=true, NN_baseline=false,plot_fits=false)
            mps_scores[it] = stats[1][:MAE]
        end
        pm_scores[ipm] = mean(mps_scores)
    end
    # println("Fold $(fold_idx): Evaluated instance $(instance)/$(num_instances) ($(round(time() - bt; digits=3))s)")

    return mean(pm_scores)
end


function revise_history!(Xs_train::Matrix, ys_train::Vector, pms::AbstractVector{<:Real}, max_windows::Int, max_ts_per_class::Integer, history)
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
        f_per_fold = pmap(i-> score_per_fold(i, e, Xs_train, ys_train, p), 1:nfolds)                    

        for fold in 1:nfolds
            # retrieve model and train/test data
            dec, mopts, W = e.fitted_params_per_fold[fold][1]
            # @show hi, fold, W[1][1], W[1][2]
            # @show mopts.eta, mopts.chi_max
            opts = safe_options(mopts) # make sure options isnt abstract

            train_idxs, val_idxs = e.train_test_rows[fold]
            Xs_train_fold, ys_train_fold = Xs_train[train_idxs, :], ys_train[train_idxs]
            Xs_val_fold, ys_val_fold = Xs_train[val_idxs, :], ys_train[val_idxs]


            # precompute encoded data for computational speedup
            fc = init_imputation_problem(W, Xs_train_fold, ys_train_fold, Xs_val_fold, ys_val_fold, opts; verbosity=-1);

            num_instances = length(ys_val_fold)
            n_c1s = sum(ys_val_fold)
            n_c0s = num_instances - n_c1s

            n0max =  min(max_ts_per_class, n_c0s)
            n1max =  min(max_ts_per_class, n_c1s)

            classes = [zeros(Int,n0max); ones(Int,n1max)]
            sample_idxs = [shuffle(MersenneTwister(fold), 1:n_c0s)[1:n0max]; shuffle(MersenneTwister(fold), 1:n_c1s)[1:n1max]]
            #sample_idxs = [1:n0max; 1:n1max]

            n_ts = length(classes)
            f_per_fold[fold] = pmap(i-> impute_instance_MLJ(i, num_instances, fold, pms, max_windows, imp), sample_idxs)                    

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
