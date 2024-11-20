using ITensors
using Random
using QuadGK
using NumericalIntegration
using Roots
using Plots, StatsPlots
using StatsBase
using Base.Threads
using KernelDensity, Distributions
using LegendrePolynomials
import NumericalIntegration
include("../LogLoss/structs/structs.jl")
include("../LogLoss/encodings/encodings.jl")



function get_state(
        x::Float64, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector
    )
    """Get the state for a time dependent encoding at site j"""
    if opts.encoding.istimedependent
        # enc_args_concrete = convert(Vector{Vector{Vector{Int}}}, enc_args) # https://i.imgur.com/cmFIJmS.png
        state = opts.encoding.encode(x, opts.d, j, enc_args...)
    else
        state = opts.encoding.encode(x, opts.d, enc_args...)
    end

    return state
    
end


function get_conditional_probability(s::Index, state::AbstractVector{<:Number}, rdm::ITensor)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x)."""
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    return state' * Matrix(rdm, [s', s]) * state |> abs

end

function get_conditional_probability(state::ITensor, rdm::ITensor)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x)."""
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    return abs(getindex(dag(state)' * rdm * state, 1))

end



function get_conditional_probability(
        x::Float64, 
        rdm::Matrix, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector
    )

    state = get_state(x, opts, j, enc_args)

    return abs(state' * rdm * state) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end

function get_normalisation_constant(s::Index, rdm::ITensor, args...)
    """Compute the normalisation constant, Z_k, such that 
    the conditional distribution integrates to one.
    """
    return get_normalisation_constant(Matrix(rdm, [s', s]), args...) #TODO Why is it done this way again?

end


function get_normalisation_constant(rdm::Matrix, opts::Options, j::Integer, enc_args::AbstractVector)
    prob_density_wrapper(x) = get_conditional_probability(x, rdm, opts, j, enc_args)
    lower, upper = opts.encoding.range
    Z, _ = quadgk(prob_density_wrapper, lower, upper)

    return Z
end

function get_normalisation_constant(xs::AbstractVector{Float64}, pdf::AbstractVector{Float64})
    return NumericalIntegration.integrate(xs, pdf, NumericalIntegration.TrapezoidalEvenFast())

end





function get_mean_from_rdm(
        rdm::ITensor, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Float64,
        dx::Float64;
        get_std::Bool=true
    )

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        prob = get_conditional_probability(itensor(state, s), rdm) 
        probs[index] = prob
    end

    Z = get_normalisation_constant(samp_xs, probs)

    # expectation
    expect_x = mapreduce(*,+,samp_xs, probs) * dx / Z
    expect_state = itensor(get_state(expect_x, opts, j, enc_args),s) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

    std_val = 0.
    if get_std
        # variance
        squared_diffs = (samp_xs .- expect_x).^2
        var = mapreduce(*,+,squared_diffs, probs) * dx / Z

        std_val = sqrt(var)

    end

    return expect_x, expect_state, std_val

end


function get_mode_from_rdm(
        rdm::ITensor, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        x_prev::Float64, 
        max_jump::Union{Number, Nothing}=nothing
    )
    """Much simpler approach to get the mode of the conditional 
    pdf (cpdf) for a given rdm. Simply evaluate P(x) over the x range,
    with interval dx, and take the argmax."""
    # don't even need to normalise since we just want the peak
    # Z = get_normalisation_constant(s, rdm, opts, j, enc_args)

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        prob = get_conditional_probability(itensor(state, s), rdm)
        probs[index] = prob
    end
    
    # get the mode of the pdf
    if isnothing(x_prev) || isnothing(max_jump)
        mode_idx = argmax(probs)
    else
        perm = sortperm(probs;rev=true)
        mode_idx = 0
        for i in perm
            if abs(samp_xs[i] - x_prev) <= max_jump
                mode_idx = i
                break
            end
        end
        if mode_idx == 0
            @warn("No valid guess withing max_jump of the previous imputation point. Increase max_jump")
            mode_idx = perm[1]
        end
    end
    mode_x = samp_xs[mode_idx]
    mode_state = itensor(samp_states[mode_idx], s)

    return mode_x, mode_state, 0. # the mode has no way to compute the error, hence always returning 0

end

get_mode_from_rdm(
    rdm::ITensor, 
    samp_xs::AbstractVector{Float64}, 
    samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
    s::Index, 
    opts::Options,
    j::Integer,
    enc_args::AbstractVector,
    x_prev::Union{Number, Nothing}=nothing, 
    max_jump::Union{Number, Nothing}=nothing
) = get_mode_from_rdm(rdm, samp_xs, samp_states, s, x_prev, max_jump)



function get_median_from_rdm(
        rdm::ITensor, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Float64;
        get_wmad::Bool=true
    )
    # return the median and the weighted median absolute deviation as a measure of uncertainty 

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        prob = get_conditional_probability(itensor(state, s), rdm)
        probs[index] = prob
    end

    cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
    Z = cdf[end]
    cdf /= Z
    probs /= Z

    median_arg = argmin(@. abs(cdf - 0.5))

    median_x = samp_xs[median_arg]
    median_s = itensor(get_state(median_x, opts, j, enc_args), s) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

    wmad_x = 0.
    if get_wmad
        # get the weighted median abs deviation
        wmad_x = median(abs.(samp_xs .- median_x), pweights(probs))
    end
    return (median_x, median_s, wmad_x)

end





function get_median_and_cdf(
    rdm::ITensor, 
    samp_xs::AbstractVector{Float64}, 
    samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
    s::Index, 
    opts::Options, 
    j::Integer,
    enc_args::AbstractVector,
    x_prev::Float64;
    get_wmad::Bool=true
)
# return the median and the weighted median absolute deviation as a measure of uncertainty 

probs = Vector{Float64}(undef, length(samp_states))
for (index, state) in enumerate(samp_states)
    prob = get_conditional_probability(itensor(state, s), rdm)
    probs[index] = prob
end

cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
Z = cdf[end]
cdf /= Z
probs /= Z

median_arg = argmin(@. abs(cdf - 0.5))

median_x = samp_xs[median_arg]
median_s = get_state(median_x, opts, j, enc_args) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

wmad_x = 0.
if get_wmad
    # get the weighted median abs deviation
    wmad_x = median(abs.(samp_xs .- median_x), pweights(probs))
end
return (median_x, median_s, wmad_x, cdf)

end


function get_cdf(
        rdm::ITensor, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}},
        s::Index
    )

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        prob = get_conditional_probability(itensor(state, s), rdm)
        probs[index] = prob
    end

    cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
    Z = cdf[end]
    cdf /= Z
    return cdf
end

function get_sample_from_rdm(
        rdm::ITensor, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Float64;
        rng::AbstractRNG,
        rejection_threshold::Union{Float64, Symbol}, 
        max_trials::Integer,
    )
    """Sample from the conditional distribution defined by the rdm, but 
    reject samples if they exceed a predetermined threshold which is set
    by the weighted median absolute deviation (WMAD).
    - threshold is the multiplier for WMAD as threshold i.e., threshold*WMAD 
    - atol is abs tolerance for the root finder
    - max trials is the maximum number of rejections
    """
    # sample without rejection if threshold is none
    sampled_x = 0.
    if rejection_threshold == :none
        u = rand(rng)
        #cdf_wrapper(x) = get_cdf(x, rdm, Z, opts, enc_args) - u
        cdf = get_cdf(rdm, samp_xs, samp_states, s)
        Z = cdf[end]
        x_ind = argmin(@. abs(cdf/Z - u ) )
        sampled_x = samp_xs[x_ind]
        wmad = 0.
        # sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
    else
        # now determine the median and wmad - don't need high precision here, just a general ballpark
        median, _, wmad, cdf = get_median_and_cdf(rdm, samp_xs, samp_states, s, opts, j, enc_args, x_prev; get_wmad=true)
        Z = cdf[end]
        rejections = 0 # rejected :(
        for i in 1:max_trials
            u = rand(rng) # sample a random value from ~ U(0, 1)
            # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
            x_ind = argmin(@. abs(cdf/Z - u ) )
            sampled_x = samp_xs[x_ind]
            # sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
            # choose whether to accept or reject
            if abs(sampled_x - median) < rejection_threshold*wmad
                break 
            end
            rejections += 1
        end
        #@show rejections
    end
    # map sampled x_k back to a state
    sampled_state = itensor(samp_states[x_ind], s) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 
    return sampled_x, sampled_state, wmad 
end



##### Maybe used for something?
# function check_inverse_sampling(
#     rdm::Matrix, 
#     opts::Options,
#     j::Integer,
#     enc_args::AbstractVector; 
#     dx::Float64=0.01)
#     """Check the inverse sampling approach to ensure 
#     that samples represent the numerical conditional
#     probability distribution."""
#     Z = get_normalisation_constant(rdm, opts, enc_args)
#     lower, upper = opts.encoding.range
#     xvals = collect(lower:dx:upper)
#     probs = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         prob = (1/Z) * get_conditional_probability(xval, rdm, opts, j, enc_args)
#         probs[index] = prob
#     end

#     return xvals, probs

# end


# function plot_samples_from_rdm(
#         rdm::Matrix, 
#         opts::Options, 
#         n_samples::Integer,
#         show_plot::Bool,
#         j::Integer,
#         enc_args::AbstractVector
#     )
#     """Plot a histogram of the samples drawn 
#     from the conditional distribution specified
#     by the conditional density matrix ρ_k."""
#     samples = Vector{Float64}(undef, n_samples)
#     bins = sqrt(n_samples)
#     @threads for i in eachindex(samples)
#         samples[i], _ = get_sample_from_rdm(rdm, opts, j, enc_args)
#     end
#     population_mean = mean(samples)
#     h = StatsPlots.histogram(samples, num_bins=bins, normalize=true, 
#         label="Inverse Transform Samples", 
#         xlabel="x",
#         ylabel="Density", 
#         title="Conditional Density Matrix, $n_samples samples")
#     h = vline!([population_mean], lw=3, label="Population Mean, μ = $(round(population_mean, digits=4))", c=:red)
#     xvals, numerical_probs = check_inverse_sampling(rdm, opts, j, enc_args)
#     h = plot!(xvals, numerical_probs, label="Numerical Solution", lw=3, ls=:dot, c=:black)
#     if show_plot
#         display(h)
#     end

#     return h, samples

# end


# function inspect_known_state_pdf(
#         x::Float64, 
#         opts::Options, 
#         enc_args::AbstractVector, 
#         j::Integer,
#         n_samples::Integer; 
#         show_plot=true
#     )
#     """ Inspect the distribution corresponding to 
#     a conditional density matrix, given a
#     known state ϕ(x_k). For an in ideal encoding with minimal uncertainty, 
#     the mean of the distribution should align closely with the known value."""
#     state = get_state(x, opts, j, enc_args)
#     # reduced density matrix is given by |x⟩⟨x|
#     rdm = state * state'
#     h, samples = plot_samples_from_rdm(rdm, opts, j, enc_args, n_samples)
#     if show_plot
#         title!("$(opts.encoding.name) encoding, d=$(opts.d), \n aux basis dim = $(opts.aux_basis_dim), site $j")
#         vline!([x], label="Known value: $x", lw=3, c=:green)
#         display(h)
#     end

#     return samples

# end


# function get_encoding_uncertainty(
#         opts::Options, 
#         j::Integer,
#         enc_args::AbstractVector, 
#         xvals::Vector
#     )

#     """Computes the error as the abs. diff between
#     a known x value (or equivalently, known state) and the
#     expectation obtained by sampling from the rdm defined by the
#     encoding"""
#     expects = Vector{Float64}(undef, length(xvals))
#     stds = Vector{Float64}(undef, length(xvals))
#     @threads for i in eachindex(xvals)
#         xval = xvals[i]
#         # make the rdm
#         state = get_state(xval, opts, j, enc_args)
#         rdm = state * state'
#         expect_x, std_val, _ = get_mean_std_from_rdm(rdm, opts, j, enc_args)
#         expects[i] = expect_x
#         stds[i] = std_val
#     end
#     # compute the abs. diffs
#     abs_diffs = abs.(expects - xvals)
#     return xvals, abs_diffs, stds
# end

# function get_dist_mean_difference(eval_intervals::Integer, opts::Options, n_samples::Integer, j::Integer, enc_args::AbstractVector)
#     """Get the difference between the known value
#     and distribution mean for the given encoding 
#     over the interval x_k ∈ [lower, upper]."""
#     lower, upper = opts.encoding.range
#     xvals = LinRange((lower+1E-8), (upper-1E-8), eval_intervals)
#     deltas = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         # get the state
#         println("Computing x = $xval")
#         state = get_state(xval, opts, j, enc_args)
#         # make the rdm 
#         rdm = state * state'
#         # get the
#         samples = Vector{Float64}(undef, n_samples)
#         @threads for i in eachindex(samples)
#             samples[i], _ = get_sample_from_rdm(rdm, opts, j, enc_args)
#         end
#         mean_val = mean(samples)
#         delta = abs((xval - mean_val))
#         deltas[index] = delta
#     end 

#     return collect(xvals), deltas

# end



########################################### Redundant ################################################
# function get_sample_from_rdm(
#         rdm::Matrix, 
#         opts::Options, 
#         j::Integer,
#         enc_args::AbstractVector;
#         threshold::Union{Float64, Symbol}=2.5, 
#         max_trials::Integer=10, atol=1e-5
#     )
#     """Sample from the conditional distribution defined by the rdm, but 
#     reject samples if they exceed a predetermined threshold which is set
#     by the weighted median absolute deviation (WMAD).
#     - threshold is the multiplier for WMAD as threshold i.e., threshold*WMAD 
#     - atol is abs tolerance for the root finder
#     - max trials is the maximum number of rejections
#     """
#     Z = get_normalisation_constant(rdm, opts, enc_args)
#     # sample without rejection if threshold is none
#     sampled_x = 0
#     if threshold == :none
#         u = rand()
#         #cdf_wrapper(x) = get_cdf(x, rdm, Z, opts, enc_args) - u
#         sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
#         #sampled_x = find_zero(cdf_wrapper, opts.encoding.range; atol=atol)
#     else
#         # now determine the median and wmad - don't need high precision here, just a general ballpark
#         median_x, _, wmad_x = get_median_from_rdm(rdm, opts, j, enc_args; binary_thresh=1e-2, dx=0.01)
#         sampled_x = 0
#         rejections = 0 # rejected :(
#         for i in 1:max_trials
#             u = rand() # sample a random value from ~ U(0, 1)
#             # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
#             # cdf_wrapper(x) = get_cdf(x, rdm, Z, opts, enc_args) - u
#             # sampled_x = find_zero(cdf_wrapper, opts.encoding.range; atol=atol)
#             sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
#             # choose whether to accept or reject
#             if abs(sampled_x - median_x) < threshold*wmad_x
#                 break 
#             end
#             rejections += 1
#         end
#         #@show rejections
#     end
#     # map sampled x_k back to a state
#     sampled_state = get_state(sampled_x, opts, j, enc_args)
#     return sampled_x, sampled_state    
# end

# function get_cpdf_mode(
#         rdm::Matrix, 
#         opts::Options,
#         enc_args::AbstractVector;
#         dx = 1E-4
#     )
#     """Much simpler approach to get the mode of the conditional 
#     pdf (cpdf) for a given rdm. Simply evaluate P(x) over the x range,
#     with interval dx, and take the argmax."""
#     # don't even need to normalise since we just want the peak
#     Z = get_normalisation_constant(rdm, opts, enc_args)
#     lower, upper = opts.encoding.range
#     xvals = collect(lower:dx:upper)
#     probs = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args)
#         probs[index] = prob
#     end
    
#     # get the mode of the pdf
#     mode_idx = argmax(probs)
#     mode_x = xvals[mode_idx]

#     # convert xval back to state
#     mode_state = get_state(mode_x, opts, enc_args)

#     return mode_x, mode_state

# end

# function get_cpdf_mode(
#         rdm::Matrix, 
#         opts::Options, 
#         enc_args::AbstractVector,
#         j::Integer; 
#         dx = 1E-4
#     )
#     Z = get_normalisation_constant(rdm, opts, enc_args, j)
#     lower, upper = opts.encoding.range
#     xvals = collect(lower:dx:upper)
#     probs = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args, j)
#         probs[index] = prob
#     end
    
#     # get the mode of the pdf
#     mode_idx = argmax(probs)
#     mode_x = xvals[mode_idx]

#     # convert xval back to state
#     mode_state = get_state(mode_x, opts, enc_args, j)

#     return mode_x, mode_state

# end

# function get_mean_from_rdm(
#         rdm::Matrix, 
#         opts::Options,
#         j::Integer,
#         enc_args::AbstractVector;
#         dx = 1E-4,
#         get_std::Bool=true
#     )

#     Z = get_normalisation_constant(rdm, opts, j, enc_args)
#     lower, upper = opts.encoding.range
#     xvals = collect(lower:dx:upper)
   
#     probs = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         prob = (1/Z) * get_conditional_probability(xval, rdm, opts, j, enc_args)
#         probs[index] = prob
#     end

#     # expectation
#     expect_x = mapreduce(*,+,samp_xs, probs) * dx
#     expect_state = get_state(expect_x, opts, j, enc_args)

#     std_val = 0
#     if get_std
#         # variance
#         squared_diffs = (xvals .- expect_x).^2
#         var = sum(squared_diffs .* probs) * dx

#         std_val = sqrt(var)

#     end

#     return expect_x, expect_state, std_val

# end

# function get_mean_std_from_rdm(
#         rdm::Matrix, 
#         opts::Options, 
#         enc_args::AbstractVector, 
#         j::Integer; 
#         dx = 1E-4
#     )

#     Z = get_normalisation_constant(rdm, opts, enc_args, j)
#     lower, upper = opts.encoding.range
#     xvals = collect(lower:dx:upper)
    
#     probs = Vector{Float64}(undef, length(xvals))
#     for (index, xval) in enumerate(xvals)
#         prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args, j)
#         probs[index] = prob
#     end

#     # expectation
#     expect_x = sum(xvals .* probs) * dx
    
#     # variance
#     squared_diffs = (xvals .- expect_x).^2
#     var = sum(squared_diffs .* probs) * dx

#     std_val = sqrt(var)
#     expect_state = get_state(expect_x, opts, enc_args, j)

#     return expect_x, std_val, expect_state

# end

function compute_entanglement_entropy_profile(class_mps::MPS)
    """Compute the entanglement entropy profile (page curve)
    for an un-labelled (class) mps.
        """
    mps = deepcopy(class_mps)
    @assert isapprox(norm(mps), 1.0; atol=1e-3) "MPS is not normalised!"
    mps_length = length(mps)
    entropy_vals = Vector{Float64}(undef, mps_length)
    # for each bi-partition coordinate, compute the entanglement entropy
    for oc in eachindex(mps)
        orthogonalize!(mps, oc) # shift orthogonality center to bipartition coordinate
        if oc == 1 || oc == mps_length
            _, S, _ = svd(mps[oc], (siteind(mps, oc)))
        else
            _, S, _ = svd(mps[oc], (linkind(mps, oc-1), siteind(mps, oc)))
        end
        SvN = 0.0
        # loop over the diagonal of the singular value matrix and extract the values
        for n = 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            if (p > 1E-12) # to avoid log(0)
                SvN += -p * log(p)
            end
        end
        entropy_vals[oc] = SvN
    end

    return entropy_vals

end


