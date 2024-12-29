
# abstract type for all missing data mechanisms
abstract type MissingMechanism end

# various missing data mechanisms according to Rubin
abstract type MCARMechanism <: MissingMechanism end
abstract type MARMechanism <: MissingMechanism end
abstract type MNARMechanism <: MissingMechanism end

# concrete types for each mechanism 
# MCAR - missing completely at random
struct BernoulliMCAR <: MCARMechanism end
struct ExponentialMCAR <: MCARMechanism end

# MAR - missing at random 
struct BlockMissingMAR <: MARMechanism end

# MNAR - missing not at random
struct LowestMNAR <: MNARMechanism end
struct HighestMNAR <: MNARMechanism end

remove_values(X::AbstractVector{Float64}, idxs::Vector{Int64}) = (Xc = deepcopy(X); Xc[idxs] .= NaN; Xc)
percentage_missing_values(X::AbstractVector) = 100.0 * count(isnan, X) / length(X)

"""
Missing completely at random (MCAR).
"""
function mcar(X::AbstractVector, fraction_missing::Float64, mechanism::MCARMechanism=BernoulliMCAR();
    state::Union{Int, Nothing}=nothing, verbose::Bool=false)

    # specify random state for reproducibility
    if !isnothing(state)
        Random.seed!(state)
    end

    if !(0.0 ≤ fraction_missing ≤ 1.0)
        throw(ArgumentError("fraction_missing must be between 0 and 1"))
    end

    X_corrupted, missing_idxs = _mcar_sample(X, fraction_missing, mechanism)
    if verbose
        actual_missing = percentage_missing_values(X_corrupted)
        println("Expected missing: $(100fraction_missing)%. Actual missing: $(round(actual_missing, digits=2))%")
    end
    
    return X_corrupted, missing_idxs

end

"""
```Julia
_mcar_sample(X::AbstractVector{Float64}, fraction_missing::Float64, ::BernoulliMCAR) -> Tuple{Vector{Float64}, Vector{Int64}}
```
Determine missing value indices by sampling from a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) with probability of success
``p`` determined by the target percentage missing.
Adapated from [Twala](https://www.tandfonline.com/doi/full/10.1080/08839510902872223).
"""
function _mcar_sample(X::AbstractVector{Float64}, fraction_missing::Float64, ::BernoulliMCAR)
    n = length(X)
    bernoulli_dist = Bernoulli(fraction_missing)
    mask = rand(bernoulli_dist, n)
    missing_idxs = collect(1:n)[mask]
    X_corrupted = remove_values(X, missing_idxs)
    return X_corrupted, missing_idxs
end

"""
```Julia
_mcar_sample(X::AbstractVector, fraction_missing::Float64, ::ExponentialMCAR) -> Tuple{Vector{Float64}, Vector{Int64}}
```
Determine missing value indices by sampling from an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) with rate
``lambda`` determined by the target percentage missing.
Adapted from [Moritz et al.](https://arxiv.org/pdf/1510.03924).
"""
function _mcar_sample(X::AbstractVector, fraction_missing::Float64, ::ExponentialMCAR)
    a = 1 # initialise index
    n = length(X)
    expon = Exponential(1/fraction_missing)
    # pre-allocate space for missing_idxs
    missing_idxs = Int[]
    sizehint!(missing_idxs, round(Int, fraction_missing * n))

    while a ≤ n
        a = ceil(Int, a + rand(expon))
        if a ≤ n
            push!(missing_idxs, a)
        end
    end
    X_corrupted = remove_values(X, missing_idxs)
    return X_corrupted, missing_idxs
end

"""
Missing at random (MAR).
"""
function mar(X::AbstractVector, fraction_missing::Float64, mechanism::MARMechanism=BlockMissingMAR();
    state::Union{Int, Nothing}=nothing, verbose::Bool=false)
    # specify random state for reproducibility
    if !isnothing(state)
        Random.seed!(state)
    end

    if !(0.0 ≤ fraction_missing ≤ 1.0)
        throw(ArgumentError("fraction_missing must be between 0 and 1"))
    end

    X_corrupted, missing_idxs = _mar_sample(X, fraction_missing, mechanism)
    if verbose
        actual_missing = percentage_missing_values(X_corrupted)
        println("Expected missing: $(100fraction_missing)%. Actual missing: $(round(actual_missing, digits=2))%")
    end
    
    return X_corrupted, missing_idxs
end

"""
    _mar_sample(X::AbstractVector, fraction_missing::Float64, ::BlockMissingMAR)

Remove a consecutive "block" of observations with size specified by the fraction missing. 
The block location starting point is chosen randomly from a list of valid starting points
(given the block size) and subsequent elements are removed.

The chosen missing block depends solely on the time index (i.e., the starting point is 
selected uniformly from valid indices) and *not* on the underlying data values. 
Thus, the probability of being missing is independent of unobserved values, relying only on an 
observed variable (time). 
This makes the missing mechanism "Missing at Random."
"""
function _mar_sample(X::AbstractVector, fraction_missing::Float64, ::BlockMissingMAR)
    n = length(X)
    npts_miss = round(Int, n * fraction_missing)
    start_idx = rand(1:(n-npts_miss+1)) # choose random starting location from valid locations
    missing_idxs = collect(start_idx:(start_idx+npts_miss - 1))
    X_corrupted = remove_values(X, missing_idxs)
    return X_corrupted, missing_idxs
end

