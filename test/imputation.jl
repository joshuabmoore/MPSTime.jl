using JLD2
using Random

# ecg200 dataset
@load "Data/ecg200/mps_saves/test_dataset.jld2" mps X_train y_train X_test y_test

imp = init_imputation_problem(mps, X_test, y_test; verbosity=-10);
imp_rtol = 0.0001;

# median test
class = 1
pm = 0.8 # 80% missing data
instance_idx = 20 # time series instance in test set
_, impute_sites_pm80 = mar(X_test[instance_idx, :], pm; rng=Xoshiro(123)) # simulate MAR mechanism
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
@test isapprox(stats_pm80[1][:MAPE], 0.4932038621699139; rtol=imp_rtol)
@test isapprox(stats_pm80[1][:NN_MAPE], 0.5319691385738694; rtol=imp_rtol)

pm = 0.2 # a quick version

rng = Xoshiro(1)
imp_methods = [:median, :mean, :mode, :ITS, :kNearestNeighbour]

# ecg200 has two class, 0 and 1
nc1s = sum(y_test)
ncs = [length(y_test) - nc1s, nc1s]

expected_maes = [
    0.3939908566614806 0.2471168103635673;
    0.29751811868210404 0.19130498934765275;
    0.672379450643265 0.4675452519447549;
    0.9523681478324807 0.9022735207072483;
    0.6496682469680876 0.43737665707665724;
]
for (i, method) in enumerate(imp_methods)
    # println("method = $(string(method))")
   
    for (ci, class) in enumerate([0,1])
        # println("class = $class")
        if method == :mean && class == 1
            # println("Expecting exactly one warning:")
        end
        ns = ncs[ci]
        idxs = randperm(rng, ns)[1:10]
        mae = 0.
        for instance_idx in idxs
            # println("idx=$(instance_idx)")
            _, impute_sites_pm20 = mar(X_test[instance_idx, :], pm; rng=rng) # state=1000*i + 100*ci + instance_idx
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
        #@show mae/length(idxs)
        @test isapprox(expected_maes[i, ci], mae / length(idxs); rtol=imp_rtol)
    end
end
