include("../../MLJIntegration/MLJ_integration.jl")
using JLD2
using Plots
using Plots.PlotMeasures
using StatsBase
using Tables
using MLJParticleSwarmOptimization
using PrettyTables
using StableRNGs
import StatisticalMeasures.ConfusionMatrices as CM


f = jldopen("Data/epilepsy/datasets/Epilepsy2.jld2", "r");
    X_train_f = read(f, "X_train")
    y_train_f = read(f, "y_train");
    X_test_f = read(f, "X_test")
    y_test_f = read(f, "y_test");
close(f)

f2 = jldopen("FinalBenchmarks/Epilepsy2/python/epilepsy2_resample_folds_julia_idx.jld2", "r");
    resample_folds = read(f2, "julia_folds")
close(f2)

# load the resample fold indices (with Julia indexing)

# function class_distribution(y_train::Vector{Int}, y_test::Vector{Int})
#     train_counts = countmap(y_train)
#     test_counts = countmap(y_test)
#     tr_classes, tr_vals = collect(keys(train_counts)), collect(values(train_counts))
#     te_classes, te_vals = collect(keys(test_counts)), collect(values(test_counts))

#     # compute distribution stats
#     tr_dist = tr_vals ./ sum(tr_vals)
#     te_dist = te_vals ./ sum(te_vals)
#     # compute chance level acc
#     chance_acc = sum(te_dist.^2)
#     println("Distribution adjusted chance accuracy: $(round(chance_acc; digits=4))")

#     header = (
#         ["Class", "Train", "Test"]
#     )
#     t = pretty_table(hcat(tr_classes, tr_dist, te_dist); header=header)

#     p_train = bar(tr_classes, tr_vals, 
#         xlabel="Class", ylabel="Count", title="Train Set",
#         c=:lightsteelblue, label="")
#     p_test = bar(te_classes, te_vals, 
#         xlabel="Class", ylabel="Count", title="Test Set",
#         c=:red, label="")
#     p = plot(p_train, p_test, size=(1200, 300), bottom_margin=5mm, left_margin=5mm)
#     display(p)

# end

# function plot_examples(class::Int, X::Matrix{Float64}, y::Vector{Int};
#     nplot=10, seed=nothing)
#     if !isnothing(seed)
#         Random.seed!(seed)
#     end
#     pal = palette(:tab10)
#     c_idxs = findall(x -> x .== class, y)
#     p_idxs = sample(c_idxs, nplot; replace=false)
#     ps = [plot(X[idx, :], xlabel="t", ylabel="x", label="", c=pal[class+1]) for idx in p_idxs]
#     p = plot(ps..., size=(1000, 500), bottom_margin=5mm, left_margin=5mm, title="C$class")
#     display(p)
# end

# function plot_conf_mat(yhat, y, mach; normalise=false)
#     model = mach.model
#     eta = model.eta
#     chi = model.chi_max
#     seed = model.init_rng
#     d = model.d 
#     # infer the data length
#     T = size(mach.data[1], 1)
#     cm = CM.confmat(yhat, y);
#     confmat = Float64.(CM.matrix(cm));
#     if normalise
#         # divide each row by row sum to get proportions
#         confmat ./= sum(confmat, dims=2)[:, 1]
#     end
#     reversed_confmat = reverse(confmat, dims=1)
#     hmap = heatmap(reversed_confmat,
#         color=:Blues,
#         xticks=(1:size(confmat,2), ["$n" for n in 0:(size(confmat,2) - 1)]),
#         yticks=(1:size(confmat,1), reverse(["$n" for n in 0:(size(confmat,1) - 1)]) ),
#         xlabel="Predicted Class",
#         ylabel="Actual Class",
#         title="Confusion Matrix, η=$(round(eta; digits=3)), χ=$chi, \nd=$d, T=$T, seed=$seed")
        
#     for (i, row) in enumerate(eachrow(reversed_confmat))
#         for (j, value) in enumerate(row)
            
#             annotate!(j, i, text(string(round(value; digits=3)), :center, 10))
#         end
#     end

#     display(hmap)
# end

# function plot_incorrectly_labelled(X_test, y_test, y_preds; zero_indexing::Bool=false)
#     """Function to plot the time-series that were incorrectly labelled"""
#     # get idxs of incorrectly classified instances
#     # use zero indexing to match python outputs for direct comparison
#     incorrect = y_test .!= y_preds
#     incorrect_idxs = findall(x -> x .== 1, incorrect)
#     X_test_mat = Tables.matrix(X_test)
#     incorrect_ts = X_test_mat[incorrect_idxs, :]
#     ps = []
#     for i in 1:(size(incorrect_ts, 1))
#         color = y_test[incorrect_idxs[i]] == 0 ? :orange : :blue
#         pi = plot(incorrect_ts[i, :], 
#             xlabel="t", ylabel="x", 
#             title="Actual: $(y_test[incorrect_idxs[i]]), Pred: $(y_preds[incorrect_idxs[i]]), idx: $(incorrect_idxs[i])",
#             c=color, label="")
#         push!(ps, pi)
#     end
#     p = plot(ps..., size=(2000, 1200), bottom_margin=5mm, left_margin=5mm)
#     display(p)
# end

X_train = MLJ.table(X_train_f)
X_test = MLJ.table(X_test_f)
y_train = coerce(y_train_f, OrderedFactor)
y_test = coerce(y_test_f, OrderedFactor)

Xs = MLJ.table([X_train_f; X_test_f])
ys = coerce([y_train_f; y_test_f], OrderedFactor)

exit_early=false

nsweeps=3
chi_max=10
eta=0.1
d=5

############### Do some hyperparameter optimisation ###############
base_mps = MPSClassifier(nsweeps=3, chi_max=35, eta=0.1, d=4, encoding=:Legendre_No_Norm, 
    exit_early=false, init_rng = 4567)

r_d = MLJ.range(base_mps, :d, values=[4, 5, 6])
r_chi = MLJ.range(base_mps, :chi_max, values=[8, 10, 15, 20, 25, 30, 35])
    
swarm = AdaptiveParticleSwarm(rng=MersenneTwister(42)) 
self_tuning_mps = TunedModel(
        model=base_mps,
        resampling=StratifiedCV(nfolds=5, rng=MersenneTwister(42)),
        tuning=swarm,
        range=[r_chi, r_d],
        measure=MLJ.misclassification_rate,
        n=18,
        acceleration=CPUThreads()
    );

function run_folds(Xs, ys, resample_folds, self_tuning_model)
    per_fold_accs = Vector{Float64}(undef, 30);
    per_fold_best_model = Vector{Dict}(undef, 30); 
    for i in 1:30
        println("Running fold $(i)")
        train_idxs = resample_folds[i]["train"]
        test_idxs = resample_folds[i]["test"]
        X_train_fold = MLJ.table(Tables.matrix(Xs)[train_idxs, :])
        y_train_fold = ys[train_idxs]
        X_test_fold = MLJ.table(Tables.matrix(Xs)[test_idxs, :])
        y_test_fold = ys[test_idxs]
        mach = machine(self_tuning_model, X_train_fold, y_train_fold)
        MLJ.fit!(mach)

        best_model = report(mach).best_model
        mach_best = machine(best_model, X_train_fold, y_train_fold)
        MLJ.fit!(mach_best)
        y_preds = MLJ.predict(mach_best, X_test_fold)
        acc = MLJ.accuracy(y_preds, y_test_fold)
        println("FOLD $i ACC: $acc")
        # extract info
        m = mach_best.model
        info = Dict(
            "d" => m.d,
            "chi_max" => m.chi_max,
            "eta" => m.eta
        )
        per_fold_accs[i] = acc 
        per_fold_best_model[i] = info
    end
    return per_fold_accs, per_fold_best_model
end

# run main loop 
fold_accs, fold_models = run_folds(Xs, ys, resample_folds, self_tuning_mps)
#JLD2.@save "epilepsy2_30fold_mps.jld2" fold_accs fold_models
