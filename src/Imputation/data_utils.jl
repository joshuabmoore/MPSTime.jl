
remove_values(X::AbstractVector{Float64}, idxs::Vector{Int64}) = (Xc = deepcopy(X); Xc[idxs] .= NaN; Xc)
percentage_missing_values(X::AbstractVector) = ((sum([isnan(x) for x in X]))/length(X)) * 100.0

"""
```Julia
mcar(X::AbstractVector{Float64}, p_miss::Float64, seed::Int=42) -> Vector{Int}
```
Missing completely at random (MCAR). 
Assumption is that the probability of a particular time point being missing is independent of the 
time series value at that point:
``\\P(r \\| \\Y_obs, \\Y_missing) = \\P(r)``.
Missing values are generated using a random number generator to select deletion locations.
MCAR is a strong assumption that is often unrealistic in real-world scenarios. 
"""
function mcar(X::AbstractVector{Float64}, fraction_missing::Float64; state::Union{Nothing, Int}=nothing, verbose::Bool=true)
    if !isnothing(state)
        Random.seed!(state)
    end
    if !(0 <= fraction_missing <= 1.0)
        throw(ArgumentError("fraction of missing points must be in the range [0, 1]."))
    end
    mask = rand(length(X)) .< fraction_missing
    missing_idxs = collect(1:length(X))[mask]
    X_corrupted = remove_values(X, missing_idxs)
    if verbose
        println("Percentage of newly generated missing values: $(round(percentage_missing_values(X_corrupted); digits=2))%")
    end
    return (X_corrupted, missing_idxs)
end

"""
Determine missing value indices by sampling from an exponential distribution with rate determined by the target 
percentage missing. Adapted from [Moritz et al.](https://arxiv.org/pdf/1510.03924).
"""
function mcar_exp(X::AbstractVector, fraction_missing::Float64; state::Union{Nothing, Int}=nothing, verbose::Bool=true)
    if !isnothing(state)
        Random.seed!(state)
    end
    if !(0 <= fraction_missing <= 1.0)
        throw(ArgumentError("fraction of missing points must be in the range [0, 1]."))
    end
    a = 1 # initialise index
    expon = Exponential(1/fraction_missing)
    missing_idxs = []
    while a < length(X)
        a = ceil(Int, a + rand(expon))
        if a <= length(X)
            push!(missing_idxs, a)
        end
    end
    X_corrupted = remove_values(X, Int.(missing_idxs))
    if verbose
        println("Percentage of newly generated missing values: $(round(percentage_missing_values(X_corrupted); digits=2))%")
    end
    return (X_corrupted, missing_idxs)
end

"""
```Julia
mbov(X::AbstractVector{Float64}, fraction_missing::Float64, remove_smallest::Bool=true; verbose::Bool=true)
```
Missing based on own values by removing the largest or smallest N points where N is determined by the target fraction missing.
"""
function mbov(X::AbstractVector{Float64}, fraction_missing::Float64, remove_smallest::Bool=true)
    npts = round(Int, length(X) * fraction_missing) # determine num of pts to remove
    # select the first npts as the points to remove
    missing_idxs = (remove_smallest == true) ? X |> x -> sortperm(x) |> x -> sort(x[1:npts-1]) : X |> x -> sortperm(x; rev=true) |> x -> sort(x[1:npts-1]) # lol
    X_corrupted = remove_values(X, missing_idxs)
    return (X_corrupted, missing_idxs)
end

"""
```Julia

```
Missing not-at-random (MNAR) is when the probability of a data point being missing is related to the value of the missing data itself and/or 
values at othe time points:
``\\P(r | \\Y_obs, \\Y_missing) = \\P(r | \\Y_obs, \\Y_missing).``
Strategies are informed by this recent [paper](https://www.sciencedirect.com/science/article/pii/S0957417424005207).
Several mechanisms are available:
1. ``:MBOVHigh``: Missing based on own values. Remove largest values such that fraction missing is obtained. 
1. ``:MBOVLow``: Missing based on own values. Remove smallest values such that fraction missing is obtained. 
"""
function mnar(X::AbstractVector{Float64}, fraction_missing::Float64, mechanism::Symbol=:MBOVHigh; verbose::Bool=true)
    if !(0 <= fraction_missing <= 1.0)
        throw(ArgumentError("fraction of missing points must be in the range [0, 1]."))
    end
    if mechanism == :MBOVHigh
        X_corrupted, missing_idxs = mbov(X, fraction_missing, false)
    elseif mechanism == :MBOVLow
        X_corrupted, missing_idxs = mbov(X, fraction_missing, true)
    else
        throw(ArgumentError("Invalid mechanism `$(string(mechanism))`. Choose either :MBOVHigh or :MBOVLow."))
    end
    if verbose
        println("Percentage of newly generated missing values: $(round(percentage_missing_values(X_corrupted); digits=2))%")
    end
    return (X_corrupted, missing_idxs)
end

"""
```Julia
block_missing(X::AbstractVector{Float64}, fraction_missing::Float64, start_idx::Union{Int, Nothing}=nothing)
    -> Tuple{Vector{Float64}, Vector{Int}}
```
Introduces a contiguous block of missing values starting at either a user-specified index or a randomly
selected location. 
In some cases, block missing data can be considered MAR if the underlying source of the missingness is from real-world events 
such as sensor downtime as the missingness is related to some external (unmeasured) factor and is not related to the time-series values themselves. 
If the block is randomly selected (and unrelated to the values themselves), then it would be considered MCAR. 
"""
function block_missing(X::AbstractVector{Float64}, fraction_missing::Float64, start_idx::Union{Int, Nothing}=nothing)
    npts_miss = round(Int, length(X) * fraction_missing)
    possible_start_idxs = collect(1:length(X)-npts_miss+1)
    if !isnothing(start_idx)
        if start_idx > possible_start_idxs[end]
            throw(ArgumentError("Starting index exceeds maximum possible starting location for $fraction_missing missing."))
        end
    else
        start_idx = rand(possible_start_idxs)
    end
    missing_idxs = collect(start_idx:(start_idx+npts_miss - 1))
    X_corrupted = remove_values(X, missing_idxs)
    return (X_corrupted, missing_idxs)
end

"""
The probability of a value being missing is unrelated to the probability of missing data at other time points but may be related to the observed values
at the current and other time points. 
"""
function mar(X::AbstractVector{Float64}, fraction_missing::Float64)
    base_prob = fraction_missing * 0.5
    slope = fraction_missing * 0.5
    p_t = base_prob .- slope .* (collect(1:length(X))/length(X))
    mask = rand(length(X)) .< p_t
    missing_idxs = collect(1:length(X))[mask]
    X_corrupted = remove_values(X, missing_idxs)
    return (X_corrupted, missing_idxs)
end
