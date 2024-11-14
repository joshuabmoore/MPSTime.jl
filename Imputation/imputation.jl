include("../LogLoss/RealRealHighDimension.jl");
include("./imputationMetrics.jl");
include("./samplingUtils.jl");
include("./imputationUtils.jl");

using JLD2
using StatsPlots, StatsBase, Plots.PlotMeasures

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


function load_forecasting_info(
        data_loc::String;
         mps_id::String="mps",
        train_data_name::String="X_train", 
        test_data_name::String="X_test",
        opts_name::String="opts"
    )

    # yes, there are a lot of checks...
    f = jldopen(data_loc, "r")
    @assert length(f) >= 6 "Expected at least 6 data objects, only found $(length(f))."
    mps = read(f, "$mps_id")
    #@assert typeof(mps) == ITensors.MPS "Expected mps to be of type MPS."
    X_train = read(f, "$train_data_name");
    @assert typeof(X_train) == Matrix{Float64} "Expected training data to be a matrix."
    y_train = read(f, "y_train");
    @assert typeof(y_train) == Vector{Int64} "Expected training labels to be a vector."
    X_test = read(f, "$test_data_name");
    @assert typeof(X_test) == Matrix{Float64} "Expected testing data to be a matrix."
    y_test = read(f, "y_test");
    @assert typeof(y_test) == Vector{Int64} "Expected testing labels to be a vector."
    opts = read(f, "$opts_name");
    @assert typeof(opts) == Options "Expected opts to be of type Options"
    @assert size(X_train, 2) == size(X_test, 2) "Mismatch between training and testing data number of samples."
    # add checks for data range.

    close(f)

    # extract info
    println("+"^60 * "\n"* " "^25 * "Summary:\n")
    println(" - Dataset has $(size(X_train, 1)) training samples and $(size(X_test, 1)) testing samples.")
    label_idx, num_classes, _ = find_label_index(mps)
    println(" - $num_classes class(es) was detected. Slicing MPS into individual states...")
    fcastables = Vector{forecastable}(undef, num_classes);
    if opts.encoding.istimedependent
        println(" - Time dependent encoding - $(opts.encoding.name) - detected, obtaining encoding args...")
        println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train)
    for class in 0:(num_classes-1)
        class_mps = slice_mps(mps, class);
        idxs = findall(x -> x .== class, y_test);
        test_samples = X_test[idxs, :];
        fcast = forecastable(class_mps, class, test_samples, opts, enc_args);
        fcastables[(class+1)] = fcast;
    end
    println("\n Created $num_classes forecastable struct(s) containing class-wise mps and test samples.")

    return fcastables
    
end


function NN_impute(fcastables::AbstractVector{forecastable},
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


function predict(
        fcastable::Vector{forecastable},
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
        fcastable::Vector{forecastable},
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

    ts, pred_err, target = predict(fcastable, which_class, which_sample, which_sites, method; kwargs...)

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
        mse_ts, _... = predict(fcastable, which_class, which_sample, which_sites, :nearestNeighbour; kwargs...)

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