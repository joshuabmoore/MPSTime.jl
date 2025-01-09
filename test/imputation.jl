using JLD2
using Random

# ecg200 dataset
@load "Data/ecg200/mps_saves/test_dataset.jld2" mps X_train y_train X_test y_test

imp = init_imputation_problem(mps, X_test, y_test; verbosity=-10)
imp_rtol = 0.0001

# median test
class = 1
pm = 0.8 # 80% missing data
instance_idx = 20 # time series instance in test set
_, impute_sites_pm80 = mar(X_test[instance_idx, :], pm; state=42) # simulate MAR mechanism
method = :median

_,_,_, stats_pm80, plots_pm80 = MPS_impute(
    imp,
    class, 
    instance_idx, 
    impute_sites_pm80, 
    method; 
    NN_baseline=true, # whether to also do a baseline imputation using 1-NN
    plot_fits=true, # whether to plot the fits
)

# we don't watnt to be _that_ precise here becase the fperror can really add up depending on what architecture this is running on
@test isapprox(stats_pm80[1][:MAPE], 0.5290891017253843; rtol=imp_rtol)
@test isapprox(stats_pm80[1][:NN_MAPE], 0.5214376879212099; rtol=imp_rtol)

pm = 0.2 # a quick version

rng = Xoshiro(1)
imp_methods = [:median, :mean, :mode, :ITS, :kNearestNeighbour]

# ecg200 has two class, 0 and 1
nc1s = sum(y_test)
ncs = [length(y_test) - nc1s, nc1s]

expected_maes = [
    0.30185545893738613 0.24989831461426074;
    0.39411112608690896 0.2295878147712654;
    0.46757431058025445 0.5939123034832571;
    1.1344937668575197 0.960981000190942;
    0.47411534752886075 0.5176290208509069;
]
for (i, method) in enumerate(imp_methods)
    # println("method = $(string(method))")
   
    for (ci, class) in enumerate([0,1])
        # println("class = $class")
        if method == :mean && class == 1
            println("Expecting exactly one warning:")
        end
        ns = ncs[ci]
        idxs = randperm(rng, ns)[1:10]
        mae = 0.
        for instance_idx in idxs
            # println("idx=$(instance_idx)")
            _, impute_sites_pm20 = mar(X_test[instance_idx, :], pm; state=1000*i + 100*ci + instance_idx)
            _, _, _, stats_pm20, plots_pm20 = MPS_impute(
                imp,
                class, 
                instance_idx, 
                impute_sites_pm20, 
                method; 
                NN_baseline=false, # whether to also do a baseline imputation using 1-NNI
                plot_fits=false, # whether to plot the fits
            )
            mae += stats_pm20[1][:MAE]
        end
        @test isapprox(expected_maes[i, ci], mae / length(idxs); rtol=imp_rtol)
    end
end
