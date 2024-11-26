
mutable struct EncodedDataRange
    dx::Float64
    guess_range::Tuple{R,R} where R <: Real
    xvals::Vector{Float64}
    site_index::Index
    xvals_enc::Vector{<:AbstractVector{<:AbstractVector{<:Number}}} # https://i.imgur.com/cmFIJmS.png (I apologise)
end 

mutable struct ImputationProblem
    mpss::AbstractVector{<:MPS}
    X_train::Matrix{<:Real}
    y_train::Vector{<:Integer}
    X_test::Matrix{<:Real}
    y_test::Vector{<:Integer}
    opts::Options
    enc_args::Vector{Any}
    x_guess_range::EncodedDataRange
end

# probably redundant if enc args are provided externally from training
function get_enc_args_from_opts(
        opts::Options, 
        X_train::Matrix, 
        y::Vector{Int}
    )
    """Rescale and then Re-encode the scaled training data using the time dependent
    encoding to get the encoding args."""

    X_train_scaled, norm = transform_train_data(permutedims(X_train) ;opts=opts)

    
    if isnothing(opts.encoding.init)
        enc_args = []
    else
        println("Re-encoding the training data to get the encoding arguments...")
        order = sortperm(y)
        enc_args = opts.encoding.init(X_train_scaled[:,order], y[order]; opts=opts)
    end

    return enc_args

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
    verbosity > 0 && println("Slicing MPS into individual states...")
    mpss, label_idx = expand_label_index(mps)
    num_classes = length(mpss)
    verbosity > 0 && println(" - $num_classes class(es) were detected.")

    if opts.encoding.istimedependent
        verbosity > 0 && println(" - Time dependent encoding - $(opts.encoding.name) - detected")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        verbosity > 0 && println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train)

    xvals=collect(range(guess_range...; step=dx))
    site_index=Index(opts.d)
    if opts.encoding.istimedependent
        println("Pre-computing possible encoded values of x_t, this may take a while... ")
        # be careful with this variable, for d=20, length(mps)=100, this is nearly 1GB for a basis that returns complex floats
        xvals_enc = [[get_state(x, opts, j, enc_args) for x in xvals] for j in 1:length(mps)] # a proper nightmare of preallocation, but necessary
    else
        xvals_enc_single = [get_state(x, opts, 1, enc_args) for x in xvals]
        xvals_enc = [view(xvals_enc_single, :) for _ in 1:length(mps)]
    end

    x_guess_range = EncodedDataRange(dx, guess_range, xvals, site_index, xvals_enc)
    mpss, l_ind = expand_label_index(mps)

    imp_prob = ImputationProblem(mpss, X_train, y_train, X_test, y_test, opts, enc_args, x_guess_range);

    verbosity > 0 && println("\n Created $num_classes ImputationProblem struct(s) containing class-wise mps and test samples.")



    return imp_prob

end


function NN_impute(
        imp::ImputationProblem,
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::AbstractVector{<:Integer}; 
        n_ts::Integer=1,
    )

    mps = imp.mpss[which_class+1]
    X_train = imp.X_train
    y_train = imp.y_train

    cl_inds = (1:length(imp.y_test))[imp.y_test .== which_class] # For backwards compatibility reasons
    target_timeseries_full = imp.X_test[cl_inds[which_sample], :]

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
        imp::ImputationProblem,
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::Vector{Int}, 
        method::Symbol=:directMean;
        invert_transform::Bool=true, # whether to undo the sigmoid transform/minmax normalisation, if this is false, timeseries that hve extrema larger than any training instance may give odd results
        n_baselines::Integer=1,
        kwargs... # method specific keyword arguments
    )

    # setup imputation variables
    X_test = imp.X_test
    X_train = imp.X_train

    mps = imp.mpss[which_class + 1]
    cl_inds = (1:length(imp.y_test))[imp.y_test .== which_class] # For backwards compatibility reasons
    target_ts_raw = imp.X_test[cl_inds[which_sample], :]
    target_timeseries= deepcopy(target_ts_raw)

    # transform the data
    # perform the scaling

    X_train_scaled, norms = transform_train_data(X_train; opts=imp.opts)
    target_timeseries_full, oob_rescales_full = transform_test_data(target_ts_raw, norms; opts=imp.opts)

    target_timeseries[which_sites] .= mean(X_test[:]) # make it impossible for the unknown region to be used, even accidentally
    target_timeseries, oob_rescales = transform_test_data(target_timeseries, norms; opts=imp.opts)

    sites = siteinds(mps)
    target_enc = MPS([itensor(get_state(x, imp.opts, j, imp.enc_args), sites[j]) for (j,x) in enumerate(target_timeseries)])

    return [target_enc], [], target_timeseries
    pred_err = []
    if method == :directMean        
        ts, pred_err = impute_mean(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, which_sites, kwargs...)
        ts = [ts] # type stability
        pred_err = [pred_err]

    elseif method == :directMedian
        ts, pred_err = impute_median(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, which_sites; kwargs...)
        ts = [ts] # type stability
        pred_err = [pred_err]

    elseif method == :directMode
        ts = impute_mode(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, which_sites; kwargs...)
        ts = [ts] # type stability

    elseif method == :ITS
        ts = impute_ITS(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, which_sites; kwargs...)

    elseif method ==:nearestNeighbour
        ts = NN_impute(imp, which_class, which_sample, which_sites; n_ts=n_baselines) # Does not take kwargs!!

        if !invert_transform
            for i in eachindex(ts)
                ts[i], _ = transform_test_data(ts[i], norms; opts=imp.opts)
            end
        end

    else
        error("Invalid method. Choose :directMean (Expect/Var), :directMode, :directMedian, :nearestNeighbour, :ITS, et. al")
    end


    if invert_transform && !(method == :nearestNeighbour)
        if !isempty(pred_err)
            for i in eachindex(ts)
                pred_err[i] .+=  ts[i] # add the time-series, so nonlinear rescaling is reversed correctly

                ts[i] = invert_test_transform(ts[i], oob_rescales, norms; opts=imp.opts)
                pred_err[i] = invert_test_transform(pred_err[i], oob_rescales, norms; opts=imp.opts)

                pred_err[i] .-=  ts[i] # remove the time-series, leaving the unscaled uncertainty          
            end
            
        else
            for i in eachindex(ts)
                ts[i] = invert_test_transform(ts[i], oob_rescales, norms; opts=imp.opts)            
            end

        end

        target = target_ts_raw

    else
        target = target_timeseries_full
    end

    if isempty(pred_err)
        pred_err = [nothing for _ in eachindex(ts)] # helps the plotting functions not crash later
    end
    

    return ts, pred_err, target
end




function MPS_impute(
        imp::ImputationProblem,
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::Vector{Int}, 
        method::Symbol=:directMedian;
        NN_baseline::Bool=true, 
        get_metrics::Bool=true, # whether to compute goodness of fit metrics
        full_metrics::Bool=false, # whether to compute every metric or just MAE
        plot_fits=true,
        print_metric_table::Bool=false,
        kwargs... # passed on to the imputer that does the real work
    )


    mps = imp.mpss[which_class + 1]
    chi_mps = maxlinkdim(mps)
    d_mps = siteinds(mps)[1] |> ITensors.dim
    enc_name = imp.opts.encoding.name

    ts, pred_err, target = get_predictions(imp, which_class, which_sample, which_sites, method; kwargs...)

    return ts, pred_err, target
    if plot_fits
        p1 = plot(ts[1], ribbon=pred_err[1], xlabel="time", ylabel="x", 
            label="MPS imputed", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm
        )

        for i in eachindex(ts)[2:end]
            p1 = plot!(ts[i], ribbon=pred_err[i], label="MPS imputed $i", ls=:dot, lw=2, alpha=0.8)
        end

        p1 = plot!(target, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
        p1 = title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Imputation, 
            d = $d_mps, χ = $chi_mps, $enc_name encoding"
        )
        ps = [p1] # for type stability
    else
        ps = []
    end


    metrics = []
    if get_metrics
        for t in ts
            if full_metrics
                push!(metrics, compute_all_forecast_metrics(t[which_sites], target[which_sites], print_metric_table))
            else
                push!(metrics, Dict(:MAE => mae(t[which_sites], target[which_sites])))
            end
        end


    end

    if NN_baseline
        mse_ts, _... = get_predictions(imp, which_class, which_sample, which_sites, :nearestNeighbour; kwargs...)

        if plot_fits 
            for (i,t) in enumerate(mse_ts)
                p1 = plot!(t, label="Nearest Train Data $i", c=:red,lw=2, alpha=0.7, ls=:dot)
            end

            ps = [p1] # for type stability
        end

        
        if get_metrics
            if full_metrics # only compute the first NN_MAE
                NN_metrics = compute_all_forecast_metrics(mse_ts[1][which_sites], target[which_sites], print_metric_table)
                for key in keys(NN_metrics)
                    metrics[1][Symbol("NN_" * string(key) )] = NN_metrics[key]
                end
            else
                metrics[1][:NN_MAE] = mae(mse_ts[1][which_sites], target[which_sites])
            end
        end
    end

    return ts, pred_err, metrics, ps
end





function get_rdms(
        imp::ImputationProblem,
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::Vector{Int};
        kwargs... # method specific keyword arguments
    )

    # setup imputation variables
    X_test = imp.X_test
    X_train = imp.X_train

    mps = imp.mpss[which_class + 1]
    cl_inds = (1:length(imp.y_test))[imp.y_test .== which_class] # For backwards compatibility reasons
    target_ts_raw = imp.X_test[cl_inds[which_sample], :]
    target_timeseries = deepcopy(target_ts_raw)

    # transform the data
    # perform the scaling

    X_train_scaled, norms = transform_train_data(X_train; opts=imp.opts)
    target_timeseries_full, oob_rescales_full = transform_test_data(target_ts_raw, norms; opts=imp.opts)

    target_timeseries[which_sites] .= mean(X_test[:]) # make it impossible for the unknown region to be used, even accidentally
    target_timeseries, oob_rescales = transform_test_data(target_timeseries, norms; opts=imp.opts)

    sites = siteinds(mps)
    target_enc = MPS([itensor(get_state(x, imp.opts, j, imp.enc_args), sites[j]) for (j,x) in enumerate(target_timeseries)])


    ts, pred_err, cdfs = get_rdms_with_med(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, which_sites; kwargs...)
    ts = [ts] # type stability
    pred_err = [pred_err]
    

    return cdfs, ts, pred_err, target_timeseries_full
end
