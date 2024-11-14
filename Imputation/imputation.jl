include("../LogLoss/RealRealHighDimension.jl");
include("./imputationMetrics.jl");
include("./samplingUtils.jl");
include("./imputationUtils.jl");

using JLD2
using StatsPlots, StatsBase, Plots.PlotMeasures

mutable struct EncodedDataRange
    dx::Float64
    guess_range::Tuple{R,R} where R <: Real
    xvals::Vector{Float64}
    site_index::Index
    xvals_enc::Vector{<:AbstractVector{<:Number}}
end 

mutable struct ImputationProblem
    mps::MPS
    class::Int
    X_train::Matrix{<:Real}
    X_test::Matrix{<:Real}
    opts::Options
    enc_args::Vector{Any}
    x_guess_range::EncodedDataRange
end

function find_label_index(mps::MPS; label_name::String="f(x)")
    """Find the label index on the mps. If the label does not exist,
    assume mps is already spliced i.e., corresponds to a single class."""
    l_mps = lastindex(ITensors.data(mps))
    posvec = [l_mps, 1:(l_mps-1)...]
    # loop through each site and check for label
    for pos in posvec
        label_idx = findindex(mps[pos], label_name)
        if !isnothing(label_idx)
            num_classes = ITensors.dim(label_idx)
            return label_idx, num_classes, pos
        end
    end

    @warn "Could not find label index on mps. Assuming single class mps."

    return nothing, 1, nothing 
end

# probably redundant if enc args are provided externally from training
function get_enc_args_from_opts(
        opts::Options, 
        X_train::Matrix, 
        y::Vector{Int}
    )
    """Rescale and then Re-encode the scaled training data using the time dependent
    encoding to get the encoding args."""

    X_train_scaled, norm = transform_train_data(X_train;opts=opts)


    if isnothing(opts.encoding.init)
        enc_args = []
    else
        println("Re-encoding the training data to get the encoding arguments...")
        enc_args = opts.encoding.init(X_train_scaled, y; opts=opts)
    end

    return enc_args

end

function slice_mps(label_mps::MPS, class_label::Int)
    """Slice an MPS along the specified class label index
    to return a single class state."""
    mps = deepcopy(label_mps)
    label_idx, num_classes, pos = find_label_index(mps);
    if !isnothing(label_idx)
        decision_state = onehot(label_idx => (class_label + 1));
        mps[pos] *= decision_state;
        normalize(mps);
    else
        @warn "MPS cannot be sliced, returning original MPS."
    end

    return mps

end


function init_imputation_problem(
        mps::MPS, 
        X_train::Matrix{R}, 
        y_train::Vector{Int}, 
        X_test::Matrix{R}, 
        y_test::Vector{Int},
        opts::AbstractMPSOptions; 
        verbosity::Integer=1,
        dx::Float64 = 1e-4,
        guess_range::Union{Nothing, Tuple{R,R}}=nothing
    ) where {R <: Real}
    """No saved JLD File, just pass in variables that would have been loaded 
    from the jld2 file. Need to pass in reconstructed opts struct until the 
    issue is resolved."""

    if opts isa MPSOptions
        _, _, opts = Options(opts)
    end

    if isnothing(guess_range)
        guess_range = opts.encoding.range
    end


    # extract info
    verbosity > 0 && println("+"^60 * "\n"* " "^25 * "Summary:\n")
    verbosity > 0 && println(" - Dataset has $(size(X_train, 1)) training samples and $(size(X_test, 1)) testing samples.")
    label_idx, num_classes, _ = find_label_index(mps)
    verbosity > 0 && println(" - $num_classes class(es) was detected. Slicing MPS into individual states...")
    fcastables = Vector{ImputationProblem}(undef, num_classes);
    if opts.encoding.istimedependent
        verbosity > 0 && println(" - Time dependent encoding - $(opts.encoding.name) - detected, obtaining encoding args...")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        verbosity > 0 && println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train)

    xvals=collect(range(guess_range...; step=dx))
    site_index=Index(opts.d)
    xvals_enc= [get_state(x, site_index, enc_args) for x in xvals]

    x_guess_range = EncodedDataRange(dx, guess_range, xvals, site_index, xvals_enc)

    for class in 0:(num_classes-1)
        class_mps = slice_mps(mps, class);
        train_samples = X_train[y_train .== class, :]
        test_samples = X_test[y_test .== class, :]
        fcast = ImputationProblem(class_mps, class, train_samples, test_samples, opts, enc_args, x_guess_range);
        fcastables[(class+1)] = fcast;
    end
    verbosity > 0 && println("\n Created $num_classes ImputationProblem struct(s) containing class-wise mps and test samples.")



    return fcastables

end


function NN_impute(fcastables::AbstractVector{ImputationProblem},
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::AbstractVector{<:Integer}; 
        X_train::AbstractMatrix{<:Real}, 
        y_train::AbstractVector{<:Integer}, 
        n_ts::Integer=1,
    )

    fcast = fcastables[(which_class+1)]
    mps = fcast.mps


    target_timeseries_full = fcast.test_samples[which_sample, :]


    known_sites = setdiff(collect(1:length(mps)), which_sites)
    target_series = target_timeseries_full[known_sites]

    c_inds = findall(y_train .== which_class)
    Xs_comparison = X_train[c_inds, known_sites]

    mses = Vector{Float64}(undef, length(c_inds))

    for (i, ts) in enumerate(eachrow(Xs_comparison))
        mses[i] = (ts .- target_series).^2 |> mean
    end
    
    min_inds = partialsortperm(mses, 1:n_ts)
    ts = Vector(undef, n_ts)

    for (i,min_ind) in enumerate(min_inds)
        ts_ind = c_inds[min_ind]
        ts[i] = X_train[ts_ind,:]
    end


    return ts


end


function get_predictions(
        fcastable::Vector{ImputationProblem},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}, 
        method::Symbol=:directMean;
        X_train::AbstractMatrix{<:Real}, 
        y_train::AbstractVector{<:Integer}=Int[], 
        invert_transform::Bool=true, # whether to undo the sigmoid transform/minmax normalisation, if this is false, timeseries that hve extrema larger than any training instance may give odd results
        dx::Float64 = 1E-4,
        mode_range=fcastable[1].opts.encoding.range,
        xvals::AbstractVector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(fcastable[1].opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, fcastable[1].opts, fcastable[1].enc_args) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        n_baselines::Integer=1,
        kwargs... # method specific keyword arguments
    )

    # setup imputation variables
    fcast = fcastable[(which_class+1)]
    X_test = vcat([fc.test_samples for fc in fcastable]...)

    mps = fcast.mps
    target_ts_raw = fcast.test_samples[which_sample, :]
    target_timeseries= deepcopy(target_ts_raw)

    # transform the data
    # perform the scaling

    X_train_scaled, norms = transform_train_data(X_train; opts=fcast.opts)
    target_timeseries_full, oob_rescales_full = transform_test_data(target_ts_raw, norms; opts=fcast.opts)

    target_timeseries[which_sites] .= mean(X_test[:]) # make it impossible for the unknown region to be used, even accidentally
    target_timeseries, oob_rescales = transform_test_data(target_timeseries, norms; opts=fcast.opts)

    pred_err = nothing
    if method == :directMean        
        if fcast.opts.encoding.istimedependent
            ts, pred_err = any_impute_directMean_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites, kwargs...)
        else
            ts, pred_err = any_impute_directMean(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites, kwargs...)
        end
    elseif method == :directMedian
        if fcast.opts.encoding.istimedependent
            error("Time dependent option not yet implemented!")
        else
            sites = siteinds(mps)

            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts, pred_err = any_impute_directMedian(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index, kwargs...)
        end

    elseif method == :directMedianOpt
        if fcast.opts.encoding.istimedependent
            error("Time dependent option not yet implemented!")
        else
            sites = siteinds(mps)

            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts, pred_err = any_impute_directMedianOpt(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index, kwargs...)
        end
    elseif method == :directMode
        if fcast.opts.encoding.istimedependent
            # xvals_enc = [get_state(x, opts) for x in x_vals]

            ts = any_impute_directMode_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites, kwargs...)
        else
            sites = siteinds(mps)
            
            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts = any_impute_directMode(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index, kwargs...)
        end
    elseif method == :MeanMode
        if fcast.opts.encoding.istimedependent
            # xvals_enc = [get_state(x, opts) for x in x_vals]
            error("Time dep not implemented for MeanMode")
        else
            sites = siteinds(mps)
            
            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts = any_impute_MeanMode(fcast.mps, fcast.opts, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index,  kwargs...)
        end

    elseif method == :ITS
        if fcast.opts.encoding.istimedependent
            error("Time dependent option not yet implemented!")
        else
            sites = siteinds(mps)
    
            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts = any_impute_ITS_single(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; kwargs...)
        end
    
    elseif method ==:nearestNeighbour
        ts = NN_impute(fcastable, which_class, which_sample, which_sites; X_train, y_train, n_ts=n_baselines) # Does not take kwargs!!



        if !invert_transform
            for i in eachindex(ts)
                ts[i], _ = transform_test_data(ts[i], norms; opts=fcast.opts)
            end
        end

    else
        error("Invalid method. Choose :directMean (Expect/Var), :directMode, :directMedian, :nearestNeighbour, :ITS, et. al")
    end


    if invert_transform && !(method == :nearestNeighbour)
        if !isnothing(pred_err )
            pred_err .+=  ts # remove the time-series, leaving the unscaled uncertainty

            ts = invert_test_transform(ts, oob_rescales, norms; opts=fcast.opts)
            pred_err = invert_test_transform(pred_err, oob_rescales, norms; opts=fcast.opts)

            pred_err .-=  ts # remove the time-series, leaving the unscaled uncertainty
        else
            ts = invert_test_transform(ts, oob_rescales, norms; opts=fcast.opts)

        end
        target = target_ts_raw

    else
        target = target_timeseries_full
    end

    return ts, pred_err, target
end




function MPS_impute(
        fcastable::Vector{ImputationProblem},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}, 
        method::Symbol=:directMean;
        NN_baseline::Bool=true, 
        get_metrics::Bool=true, # whether to compute goodness of fit metrics
        full_metrics::Bool=false, # whether to compute every metric or just MAE
        plot_fits=true,
        print_metric_table::Bool=false,
        kwargs... # passed on to the imputer that does the real work
    )

    fcast = fcastable[(which_class+1)]
    X_test = vcat([fc.test_samples for fc in fcastable]...)
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name

    ts, pred_err, target = get_predictions(fcastable, which_class, which_sample, which_sites, method; kwargs...)

    if plot_fits
        p1 = plot(ts, ribbon=pred_err, xlabel="time", ylabel="x", 
            label="MPS imputed", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm
        )

        p1 = plot!(target, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
        p1 = title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Imputation, 
            d = $d_mps, χ = $chi_mps, $enc_name encoding"
        )
        ps = [p1] # for type stability
    else
        ps = []
    end


    if get_metrics
        if full_metrics
            metrics = compute_all_forecast_metrics(ts[which_sites], target[which_sites], print_metric_table)
        else
            metrics = Dict(:MAE => mae(ts[which_sites], target[which_sites]))
        end
    else
        metrics = []
    end

    if NN_baseline
        mse_ts, _... = get_predictions(fcastable, which_class, which_sample, which_sites, :nearestNeighbour; kwargs...)

        if plot_fits 
            if length(ts) == 1
                p1 = plot!(mse_ts[1], label="Nearest Train Data", c=:red, lw=2, alpha=0.7, ls=:dot)
            else
                for (i,t) in enumerate(mse_ts)
                    p1 = plot!(t, label="Nearest Train Data $i", c=:red,lw=2, alpha=0.7, ls=:dot)
                end

            end
            ps = [p1] # for type stability
        end

        
        if get_metrics
            if full_metrics
                NN_metrics = compute_all_forecast_metrics(mse_ts[1][which_sites], target[which_sites], print_metric_table)
                for key in keys(NN_metrics)
                    metrics[Symbol("NN_" * string(key) )] = NN_metrics[key]
                end
            else
                metrics[:NN_MAE] = mae(mse_ts[1][which_sites], target[which_sites])
            end
        end
    end

    return ts, pred_err, metrics, ps
end