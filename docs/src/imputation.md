# [Imputation](@id Imputation_top)

## Setup

The first step is to train an MPS (see [Tutorial](@ref)). Here, we'll train an unsupervised MPS (no class labels) using the the noisy trendy sine from the tutorial.

```Julia
# Fix rng seed
using Random
rng = Xoshiro(1)

# dataset size
ntimepoints = 100
ntrain_instances = 300
ntest_instances = 200

# generate the train and test datasets
X_train = trendy_sine(ntimepoints, ntrain_instances, 0.1, rng);
X_test = trendy_sine(ntimepoints, ntest_instances , 0.1, rng);

# hyper parameters and training
opts = MPSOptions(d=10, chi_max=40, sigmoid_transform=false);
mps, info, test_states= fitMPS(X_train, opts);
```

Next, we initialise an imputation problem. This does a lot of necessary precomputation
```Julia
julia> imp = init_imputation_problem(mps, X_test);
imp = init_imputation_problem(mps, X_test);
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                         Summary:

 - Dataset has 300 training samples and 200 testing samples.
Slicing MPS into individual states...
 - 1 class(es) were detected.
 - Time independent encoding - Legendre - detected.
 - d = 10, chi_max = 40
Re-encoding the training data to get the encoding arguments...

 Created 1 ImputationProblem struct(s) containing class-wise mps and test samples.
```
For multiclass data, you can pass in `y_test` to exploit the labels / class information while doing imputation.

## Imputation with the median
Now decide what you want to impute. The necessary options are:
- `class::Integer`: The class of the timeseries instance we are going to impute, leave as zero for "unlabelled" data
- `impute_sites`: The sites that are missing (inclusive). In this example we'll consider 81% of the data to be missing values
- `instance_idx`: The instance from the chosen class in the test set.
- `method`: The method to use  trajectories, or median, mode, mean etc...


```Julia
class = 0
impute_sites = collect(10:90)
instance_idx = 59
method = :median

imputed_ts, pred_err, target_ts, stats, plots = MPS_impute(
    imp,
    class, 
    instance_idx, 
    impute_sites, 
    method; 
    NN_baseline=true, # whether to also do a baseline imputation using 1-NN
    plot_fits=true, # whether to plot the fits
)
```

```Julia
julia> using PrettyTables, Plots
julia> pretty_table(stats[1]; header=["Metric", "Value"], header_crayon= crayon"yellow bold", tf = tf_unicode_rounded);
╭────────┬───────────╮
│ Metric │     Value │
├────────┼───────────┤
│    MAE │ 0.0817192 │
│ NN_MAE │  0.127104 │
╰────────┴───────────╯

plot(plots...)
```
![](./figures/median_impute.svg)


There are a lot of other options, and many more impution methods to choose from! See[`MPS_impute`](@ref) for more details.



## Plotting Trajectories
To plot trajectories, use `method=:ITS`. Here, we'll plot 10 randomly selected trajectories by setting the `num_trajectories` keyword. 
```Julia
class = 0
impute_sites = collect(10:90)
instance_idx = 59
method = :ITS

imputed_ts, pred_err, target_ts, stats, plots = MPS_impute(
    imp,
    class, 
    instance_idx, 
    impute_sites, 
    method; 
    NN_baseline=false, # whether to also do a baseline imputation using 1-NN
    plot_fits=true, # whether to plot the fits
    num_trajectories=10, # number of trajectories to plot
    rejection_threshold=2.5 # limits how unlikely we allow the random trajectories to be.
    # there are more options! see [`MPS_impute`](@ref)
)

plot(plots...)
```
![](./figures/ITS_impute.svg)



## Plotting cumulative distribution functions

It can be interesting to inspect the probability distribution being sampled from at each missing time point. To enable this, we provide the [`get_cdfs`](@ref) function, which works very similarlary to [`MPS_impute`](@ref), only it returns the CDF at each missing time point.

```Julia
cdfs, ts, pred_err, target = get_cdfs(
    imp, 
    class, 
    instance_idx, 
    impute_sites
    );

xvals = imp.x_guess_range.xvals[1:10:end]

plot(xvals, cdfs[1][1:10:end]; legend=:none)
p = last([plot!(xvals, cdfs[i][1:10:end]) for i in eachindex(cdfs)])
ylabel!("cdf(x)")
xlabel!("x_t")
title!("CDF at each time point.")
```
![](./figures/cdfs.svg)


## Docstrings 
```@docs
init_imputation_problem(::TrainedMPS, ::Matrix)
MPS_impute
get_cdfs
```

Internal imputation methods:

## Internal imputation methods

```@docs
MPSTime.impute_median
MPSTime.impute_ITS
MPSTime.kNN_impute
```
