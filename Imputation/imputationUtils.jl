import Base.*
contractTensor = ITensors._contract
*(t1::Tensor, t2::Tensor) = contractTensor(t1, t2)

include("./samplingUtils.jl");


function condition_until_next!(
        i::Int,
        it::ITensor,
        x_samps::AbstractVector{Float64},
        known_sites::AbstractVector{Int},
        class_mps::MPS,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS
    )

    while i in known_sites
        # save the x we know
        known_x = timeseries[i]
        x_samps[i] = known_x

        # make projective measurement by contracting with the site
        it *= (class_mps[i] * dag(timeseries_enc[i]))
        i += 1
    end

    return i, it
end



function allocate_mem(dtype, j, sites, links, num_imputation_sites)
    if j == 1 
        inds = (sites[j], links[j])
    elseif j == num_imputation_sites
        inds = (sites[j], links[j-1])
    else
        inds = (sites[j], links[j-1], links[j])
    end

    dims = ITensors.dim.(inds)
    return ITensor(dtype, zeros(dtype, dims), inds)
    
end



function precondition(
        class_mps::MPS,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS,
        imputation_sites::AbstractVector{Int}
    )
    s = siteinds(class_mps)
    known_sites = setdiff(collect(1:length(class_mps)), imputation_sites)
    total_known_sites = length(class_mps)
    total_impute_sites = length(imputation_sites)

    x_samps = Vector{Float64}(undef, total_known_sites) # store imputed samples
    original_mps_length = length(class_mps)

    mps_conditioned = Vector{ITensor}(undef, total_impute_sites)

    mps_cond_idx = 1
    # condition the mps on the known values
    i = 1
    while i <= original_mps_length
        if mps_cond_idx == total_impute_sites
            # at the last site, condition all remaining
            it = ITensor(1)
            if i in known_sites
                # condition all the known sites up until the last imputed site 
                i, it = condition_until_next!(i, it, x_samps, known_sites, class_mps, timeseries, timeseries_enc)
            end
            last_site = class_mps[i] # last imputed sites
            it2 = ITensor(1)
            i += 1
            # condition all the remaining sites in the mps (ok if there aren't any)
            i, it2 = condition_until_next!(i, it2, x_samps, known_sites, class_mps, timeseries, timeseries_enc)

            mps_conditioned[mps_cond_idx] = it * last_site * it2# normalize!(it * last_site * it2)

        elseif i in known_sites
            it = ITensor(1)
            i, it = condition_until_next!(i, it, x_samps, known_sites, class_mps, timeseries, timeseries_enc)
            mps_conditioned[mps_cond_idx] = it * class_mps[i] # normalize!(it * class_mps[i])
        else
            mps_conditioned[mps_cond_idx] = deepcopy(class_mps[i])
        end
        mps_cond_idx += 1
        i += 1
    end
    return x_samps, MPS(mps_conditioned)

end


function impute_at!(
        mps::MPS,
        x_samps::AbstractVector{Float64},
        imputation_method::Function,
        opts::Options,
        enc_args::AbstractVector,
        x_guess_range::EncodedDataRange,
        imputation_sites::Vector{Int},
        args...;
        norm:Bool=true,
        kwargs...
    )
    mps_inds = 1:length(mps)
    s = siteinds(mps)
    errs = zeros(Float64, length(x_samps))#Vector{Float64}(undef, total_known_sites)
    total_impute_sites = length(imputation_sites)


    orthogonalize!(mps, first(mps_inds)) #TODO: this line is what breaks imputations of non sequential sites, fix
    A = mps[first(mps_inds)]

    imp_idx = imputation_sites[1]
    if isassigned(x_samps, imp_idx - 1) # isassigned can handle out of bounds indices
        x_prev = x_samps[imp_idx - 1]

    elseif isassigned(x_samps, imp_idx + 1)
        x_prev = x_samps[imp_idx + 1]

    else
        x_prev = nothing
    end

    for (ii,i) in enumerate(mps_inds)
        imp_idx = imputation_sites[i]
        site_ind = s[i]
        xvals = x_guess_range.xvals
        xvals_enc = x_guess_range.xvals_enc[imp_idx]

        rdm = prime(A, site_ind) * dag(A)

        mx, ms, err = imputation_method(rdm, xvals, xvals_enc, site_ind, opts, imp_idx, enc_args, x_prev, args...; kwargs...)
        x_samps[imp_idx] = mx
        x_prev = mx
        errs[imp_idx] = err
       
        # recondition the MPS based on the prediction
        if ii != total_impute_sites
            ms = itensor(ms, site_ind)
            Am = A * dag(ms)
            # A = normalize!(mps[mps_inds[ii+1]] * Am)
            A = mps[mps_inds[ii+1]] * Am
            if norm
                # necessary mathematically, but most imputation methods are normalisation agnostic. norm=false where possible saves a decent amount of time and memory
                normalize!(A)

                # the amount to normalize by can be calculated theoretically via a numerical integral, but thats slow
                # proba_state = get_conditional_probability(ms, rdm)
                # A ./= sqrt(proba_state)
            end
        end
    end 
    return (x_samps, errs)
end

"""
impute missing data points using the median of the conditional distribution (single site rdm ρ).

# Arguments
- `class_mps::MPS`: 
- `opts::Options`: MPS parameters.
- `enc_args::AbstractVector`
- `x_guess_range::EncodedDataRange`
- `timeseries::AbstractVector{<:Number}`: The input time series data that will be imputed.
- `timeseries_enc::MPS`: The encoded version of the time series represented as a product state. 
- `imputation_sites::Vector{Int}`: Indices in the time series where imputation is to be performed.
- `get_wmad::Bool`: Whether to compute the weighted median absolute deviation (WMAD) during imputation (default is `false`).

# Returns
A tuple containing:
- `median_values::Vector{Float64}`: The imputed median values at the specified imputation sites.
- `wmad_value::Union{Nothing, Float64}`: The weighted median absolute deviation if `get_wmad` is true; otherwise, `nothing`.

"""
function impute_median(
        class_mps::MPS,
        opts::Options,
        enc_args::AbstractVector,
        x_guess_range::EncodedDataRange,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS,
        imputation_sites::Vector{Int};
        get_wmad::Bool=true
    )
    
    x_samps, mps_conditioned = precondition(class_mps, timeseries, timeseries_enc, imputation_sites )

    x_samps, x_wmads = impute_at!(
        mps_conditioned,
        x_samps,
        get_median_from_rdm,
        opts,
        enc_args,
        x_guess_range,
        imputation_sites;
        norm=false,
        get_wmad=get_wmad
    )

    return (x_samps, x_wmads)
end


function impute_mean(
        class_mps::MPS, 
        opts::Options, 
        x_guess_range::EncodedDataRange,
        enc_args::AbstractVector,
        timeseries::Vector{Float64},
        imputation_sites::Vector{Int};
        get_std::Bool=true
    )
    """impute mps sites without respecting time ordering, i.e., 
    condition on all known values first, then impute remaining sites one-by-one.
    
    Use direct mean/variance"""
    # condition the mps on the known values
    x_samps, mps_conditioned = precondition(class_mps, timeseries, timeseries_enc, imputation_sites )

    x_samps, x_std = impute_at!(
        mps_conditioned,
        x_samps,
        get_mean_from_rdm,
        opts,
        enc_args,
        x_guess_range,
        imputation_sites;
        norm=false,
        get_std=get_std
    )

    return (x_samps, x_std)
end

function impute_mode(
        class_mps::MPS, 
        opts::Options, 
        x_guess_range::EncodedDataRange,
        enc_args::AbstractVector,
        timeseries::AbstractVector{<:Number}, 
        timeseries_enc::MPS,
        imputation_sites::Vector{Int}; 
        max_jump::Union{Number,Nothing}=nothing
    )

    """impute mps sites without respecting time ordering, i.e., 
    condition on all known values first, then impute remaining sites one-by-one.
    Use direct mode."""
    x_samps, mps_conditioned = precondition(class_mps, timeseries, timeseries_enc, imputation_sites )

    x_samps, errs = impute_at!(
        mps_conditioned,
        x_samps,
        get_mean_from_rdm,
        opts,
        enc_args,
        x_guess_range,
        imputation_sites;
        norm=false,
        get_std=get_std
    )
    return x_samps
end

"""
Impute a SINGLE trajectory using inverse transform sampling (ITS).\n
"""
function impute_ITS_single(
    class_mps::MPS, 
    opts::Options, 
    x_guess_range::EncodedDataRange,
    enc_args::AbstractVector,
    timeseries::AbstractVector{<:Number},
    timeseries_enc::MPS, 
    imputation_sites::Vector{Int};
    rejection_threshold::Union{Float64, Symbol}=1.0,
    max_trials::Int=10
    )


    x_samps, mps_conditioned = precondition(class_mps, timeseries, timeseries_enc, imputation_sites )

    x_samps, errs = impute_at!(
        mps_conditioned,
        x_samps,
        get_mean_from_rdm,
        opts,
        enc_args,
        x_guess_range,
        imputation_sites;
        norm=true,
        threshold=rejection_threshold,
        max_trials=max_trials
    )
    return x_samps
end
