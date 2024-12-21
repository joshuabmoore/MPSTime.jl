# [Imputation](@id Imputation_top)

## Setup

The first step is to train an MPS (see [Tutorial](@ref)). 
Here, we'll train an MPS in an unsupervised manner (no class labels) using a noisy trendy sine.

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

Next, we initialise an imputation problem. This does a lot of necessary precomputation:
```Julia
julia> imp = init_imputation_problem(mps, X_test);

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
A summary of the imputation problem setup is printed to verify the model parameters and dataset information.
For __multi-class__ data, you can pass `y_test` to `init_imputation_problem` in order to exploit the labels / class information while doing imputation.

## Imputing missing values
Now, decide how you want to impute the missing data.
The necessary options are:
- `class::Integer`: The class of the time-series instance we are going to impute, leave as zero for "unlabelled" data (i.e., all data belong to the same class).
- `impute_sites`: The sites that are missing (__inclusive__).
- `instance_idx`: The time-series instance from the chosen class in the test set.
- `method`: The imputation method to use. Can be either trajectories (ITS), median, mode, mean etc...

In this example, we will consider a single block of contiguous missing values, starting at $t = 10$ through to $t = 90$ 

encompassing __81\%__ of the original time-series (i.e., 81\% data missingness).


```Julia
class = 0
impute_sites = collect(10:90) # impute time pts. 10-90 inclusive
instance_idx = 59 # time series instance in test set
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
