
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

function trendy_sine(T::Int, n::Int; period::Union{Nothing, Float64, Tuple, Vector}=nothing, slope::Union{Nothing, Float64, Tuple, Vector}=nothing, 
    phase::Union{Nothing, Float64, Tuple, Vector}=nothing, sigma::Float64=0.0, state::Union{Nothing, Int}=nothing)

    !isnothing(state) && Random.seed!(state)
    # set default ranges for random
    DEFAULT_RANGES = (
        pe = (1.0, 100.0),
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
    return X
end
