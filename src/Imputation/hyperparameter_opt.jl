
function _make_folds(X::AbstractMatrix, k::Int; rng::Union{Nothing, AbstractRNG}=nothing)
    if isnothing(rng)
        rng = Xoshiro()
    end
    ninstances = size(X, 1)
    X_idxs = randperm(rng, ninstances)
    # split into k folds
    fold_size = ceil(Int, ninstances/k)
    all_folds = [X_idxs[(i-1)*fold_size+1 : min(i*fold_size, ninstances)] for i in 1:k]
    # build pairs
    X_train_idxs = Vector{Vector{Int}}(undef, k)
    X_val_idxs = Vector{Vector{Int}}(undef, k)
    for i in 1:k
        X_val_idxs[i] = all_folds[i]
        X_train_idxs[i] = vcat(all_folds[1:i-1]..., all_folds[i+1:end]...)
    end
    return X_train_idxs, X_val_idxs
end

function make_grid(grid_type::Symbol, sweep_range::Union{Vector{Int}, StepRange}, 
    d_range::Union{Vector{Int}, StepRange}, 
    chi_range::Union{Vector{Int}, StepRange}, 
    eta_range::Union{Vector{Float64}, StepRangeLen}; 
    num_evals::Int=10, 
    rng::Union{AbstractRNG, Nothing}=nothing)

    grid_iter = Iterators.product(sweep_range, d_range, chi_range, eta_range)

    if grid_type == :random
        # check that number of samples is less than exhuastive search
        total_combos = length(grid_iter)
        if num_evals > total_combos
            throw(ArgumentError("Number of evaluations ($num_evals) exceeds total possible hyperparameter combinations ($total_combos)."))
        end
        if rng === nothing
            rng = Xoshiro()
        end
        return sample(rng, collect(grid_iter), num_evals; replace=false)
    elseif grid_type == :exhaustive
        return vcat(grid_iter...)
    else
        throw(ArgumentError("grid type $(str(grid_type)) is not valid. Choose either :random or :exhaustive."))
    end
end

"""
K-fold cross validation for time-series imputation. 
"""
function search_cv_impute(X::Matrix, k::Int, grid_type::Symbol=:random; 
    sweep_range::Union{Vector{Int}, StepRange}, 
    d_range::Union{Vector{Int}, StepRange}, 
    chi_range::Union{Vector{Int}, StepRange}, 
    eta_range::Union{Vector{Float64}, StepRangeLen},
    rng::Union{AbstractRNG, Nothing}=nothing,
    num_models::Int=10)

    X_train_idxs, X_val_idxs = _make_folds(X, k; rng=rng)
    param_grid = make_grid(grid_type, sweep_range, d_range, chi_range, eta_range; num_evals=num_models, rng=rng)
    model_scores = Vector{Float64}(undef, length(param_grid)) # holds mean scores across k folds
    pms = collect(0.05:0.15:0.95)

    # loop over models (parameter combinations)
    for (ipg, (sw, d, chi, eta)) in enumerate(param_grid)
        printstyled("Evaluating model [$ipg/$(length(param_grid))]: $((sw, d, chi, eta))\n"; bold=true, color=:cyan)
        opts = MPSOptions(d=d, chi_max=chi, nsweeps=sw, eta=eta, sigmoid_transform=false, log_level=0);
        model_fold_scores = Vector{Float64}(undef, k) # score for each fold for each model
        for fold in 1:k
            printstyled("Evaluating fold $fold/$k...\n", color=:red)
            X_train_fold = X[X_train_idxs[fold], :]
            X_val_fold = X[X_val_idxs[fold], :]
            mps = fitMPS(X_train_fold, opts)[1];
            imp = init_imputation_problem(mps, X_val_fold);
            numval = size(X_val_fold, 1)
            instance_scores = Vector{Float64}(undef, numval) # score for each instance across all % missing
            for inst in eachindex(instance_scores)
                instance_pm_scores_mps = Vector{Float64}(undef, length(pms)) # score for each % missing for a given instance
                for (ipm, pm) in enumerate(pms)
                    impute_sites = mar(X_val_fold[inst, :], pm)[2]
                    stats = MPS_impute(imp, 0, inst, impute_sites, :median; NN_baseline=false, plot_fits=false)[4]
                    instance_pm_scores_mps[ipm] = stats[1][:MAE]
                end
                instance_scores[inst] = mean(instance_pm_scores_mps)
            end
            model_fold_scores[fold] = mean(instance_scores) # mean score across all instances in validation set
        end
        model_scores[ipg] = mean(model_fold_scores) # mean score for given model across all folds
    end
    # get best scoring parameters
    sweeps_best, d_best, chi_best, eta_best = param_grid[argmin(model_scores)]
    best_params = Dict(:nsweeps => sweeps_best, :d => d_best, :chi_max => chi_best, :eta => eta_best)
    return best_params, param_grid, model_scores
end
