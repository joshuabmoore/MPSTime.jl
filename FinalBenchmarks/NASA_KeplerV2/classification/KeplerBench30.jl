include("../../../MLJIntegration/MLJ_integration.jl");
using MLJParticleSwarmOptimization
using MLJ
using StableRNGs
using JLD2
using Tables 

dloc = "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/Data/NASA_KeplerV2/datasets/classification/KeplerBinaryOrigUnbal.jld2";
w = 1:50 # SET WINDOW SIZE
f = jldopen(dloc, "r")
    X_train_f = read(f, "X_train")[:, w]
    y_train_f = read(f, "y_train")
    X_test_f = read(f, "X_test")[:, w]
    y_test_f = read(f, "y_test")
close(f)

X_train = MLJ.table(X_train_f)
y_train = coerce(y_train_f, OrderedFactor)
X_test = MLJ.table(X_test_f)
y_test = coerce(y_test_f, OrderedFactor)

Xs = MLJ.table([X_train_f; X_test_f])
ys = coerce([y_train_f; y_test_f], OrderedFactor)

chi_max = 30 #30
nsweeps = 10 # 2
d = 4 # 4
eta = 1.0
init_rng = 4567 # 4567
mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=:Legendre_No_Norm, 
    exit_early=false, init_rng=init_rng);
mach = MLJ.machine(mps, X_train, y_train)
MLJ.fit!(mach)
yhat = MLJ.predict(mach, X_test)
@show MLJ.accuracy(yhat, y_test)

# set the hyperparameter search ranges
r_eta = 0.1
r_d = MLJ.range(mps, :d, values=[3, 4, 5, 6])
r_chi = MLJ.range(mps, :chi_max, values=[20, 30, 40])
# setup the search algorithm
swarm = AdaptiveParticleSwarm(rng=MersenneTwister(0))
self_tuning_mps = TunedModel(
        model=mps,
        resampling=StratifiedCV(nfolds=5, rng=MersenneTwister(0)),
        tuning=swarm,
        range=[r_chi, r_d],
        measure=MLJ.misclassification_rate,
        n=12,
        acceleration=CPUThreads()
    );
# make the splits
train_ratio = length(y_train)/length(ys)
num_resamps = 29
splits = [
    if i == 0
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        MLJ.partition(1:length(ys), train_ratio, rng=StableRNG(i), stratify=ys) 
    end 
    for i in 0:num_resamps]

per_fold_accs = zeros(Float64, length(splits))
best_models = Vector{Any}(undef, length(splits))
for i in eachindex(splits)

    train_idxs = splits[i][1]
    X_train_fold = MLJ.table(Tables.matrix(Xs)[train_idxs, :])
    y_train_fold = ys[train_idxs]
    test_idxs = splits[i][2]
    X_test_fold = MLJ.table(Tables.matrix(Xs)[test_idxs, :])
    y_test_fold = ys[test_idxs]
    mach = machine(self_tuning_mps, X_train_fold, y_train_fold)
    MLJ.fit!(mach)
    best = report(mach).best_model
    mach_best = machine(best, X_train_fold, y_train_fold)
    MLJ.fit!(mach_best)
    yhat = MLJ.predict(mach_best, X_test_fold)
    acc = MLJ.accuracy(yhta, y_test_fold)
    per_fold_accs[i] = acc
    best_models[i] = mach_best
    println("Fold $i, Acc: $acc")
end


mach = machine(self_tuning_mps, X_train_fold, y_train_fold)
MLJ.fit!(mach)
best = report(mach).best_model
mach_best = machine(best, X_train_fold, y_train_fold)
MLJ.fit!(mach_best)
yhat = MLJ.predict(mach_best, X_test_fold)
acc = MLJ.accuracy(yhat, y_test_fold)
# mach = machine(mps, X_train, y_train )
# MLJ.fit!(mach)
# yhat = MLJ.predict(mach, X_test)
# acc = MLJ.accuracy(yhat, y_test)
# #bal_acc = MLJ.balanced_accuracy(yhat, y_test)