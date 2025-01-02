
function _generate_params(param, default_range)
    if isnothing(param)
        rand(Uniform(default_range...))
    elseif isa(param, Tuple)
        rand(Uniform(param...))
    elseif isa(param, Vector)
        rand(param)
    else
        param
    end
end

"""
```Julia
trendy_sine(T::Int, n::Int; period=nothing, slope=nothing, phase=nothing, sigma=0.0, 
    state=nothing, return_metadata=true) -> Tuple{Matrix{Float64}, Dict{Symbol, Any}}
```
Generate `n` time series of length `T`, each composed of a sine wave with an optional linear trend and Gaussian noise defined by the
equation:
```math
x_t = \\sin\\left(\\frac{2\\pi}{\\tau}t + \\psi\\right) + \\frac{mt}{T} + \\sigma n_t
```
with period ``\\tau``, time point ``\\t``, linear trend slope ``\\m``, phase offset ``\\psi``, noise scale ``\\sigma`` and ``\\n_t \\sim \\N(0,1)``
# Arguments
- `T::Int`: Length of each time series
- `n::Int`: Number of time series instances to generate

# Keyword Arguments
- `period`: Period of the sinusoid, τ
    * `nothing`: Random values between 1.0 and 50.0 (default)
    * `Float64`: Fixed period for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (1.0, 20.0) → τ ~ U(1.0, 20.0)
    * `Vector`: Sample from discrete uniform distribution, e.g., τ ∈ 10, 20, 30
- `slope`: Linear trend gradient, m
    * `nothing`: Random values bewteen -5.0 and 5.0 (default)
    * `Float64`: Fixed slope for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (-3.0, 3.0) → m ~ U(-3.0, 3.0)
    * `Vector`: Sample from discrete uniform distribution, e.g., m ∈ -3.0, 0.0, 3.0
- `phase`: Phase offset, ψ
    * `nothing`: Random values between 0 and 2π (default)
    * `Float64`: Fixed phase for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (0.0, π) → ψ ~ U(0.0, π)
    * `Vector`: Sample from discrete uniform distribution
- `sigma::Real`: Standard deviation of Gaussian noise, σ (default: 0.0)
- `state::Union{Nothing, Int}`: Random seed for reproducibility (default: nothing)
- `return_metadata::Bool`: Return generation parameters (default: true)

# Returns
- Matrix{Float64} of shape (n, T)
- Dictionary of generation parameters (:period, :slope, :phase, :sigma, :T, :n)
"""
function trendy_sine(T::Int, n::Int; period::Union{Nothing, Real, Tuple, Vector}=nothing, slope::Union{Nothing, Real, Tuple, Vector}=nothing, 
    phase::Union{Nothing, Real, Tuple, Vector}=nothing, sigma::Real=0.0, state::Union{Nothing, Int}=nothing, return_metadata::Bool=true)

    !isnothing(state) && Random.seed!(state)
    # set default ranges for random
    DEFAULT_RANGES = (
        pe = (1.0, 50.0),
        sl = (-5.0, 5.0),
        ph = (0.0, 2π)
    )
    # Generate parameter vectors
    period_vals = [_generate_params(period, DEFAULT_RANGES.pe) for _ in 1:n]
    slope_vals = [_generate_params(slope, DEFAULT_RANGES.sl) for _ in 1:n]
    phase_vals = [_generate_params(phase, DEFAULT_RANGES.ph) for _ in 1:n]
    
    X = Matrix{Float64}(undef, n, T)
    ts = 1:T
    for (i, series) in enumerate(eachrow(X))
        @. series = sin(2pi/period_vals[i] * ts + phase_vals[i]) + (slope_vals[i] * ts) / T + sigma * randn()
    end

    info = nothing
    if return_metadata
        info = Dict(
            :period => period_vals,
            :slope => slope_vals,
            :phase => phase_vals,
            :sigma => sigma,
            :T => T,
            :n => n
        )
    end
    return X, info
end
