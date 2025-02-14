function encode_TS(sample::AbstractVector, site_indices::AbstractVector{Index{Int64}}, encoding_args::AbstractVector; opts::Options=Options())
    """Function to convert a single normalised sample to a product state
    with local dimension as specified by the feature map."""

    n_sites = length(site_indices) # number of mps sites
    product_state = Matrix(undef, opts.d, n_sites)
    
    
    # check that the number of sites matches the length of the time series
    if n_sites !== length(sample)
        error("Number of MPS sites: $n_sites does not match the time series length: $(length(sample))")
    end

    for j=1:n_sites

        if opts.encoding.istimedependent
            states = opts.encoding.encode(sample[j], opts.d, j, encoding_args...)
        else
            states = opts.encoding.encode(sample[j], opts.d, encoding_args...)
        end

        product_state[:,j] = states
    end

    return product_state

end

function encode_dataset(ES::EncodeSeparate, X_orig::AbstractMatrix, X_norm::AbstractMatrix, y::AbstractVector, args...; kwargs...)
    """Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states"""
    # sort the arrays by class. This will provide a speedup if classes are trained/encoded separately
    # the loss grad function assumes the timeseries are sorted! Removing the sorting now breaks the algorithm
    if size(X_norm, 2) == 0
        encoding_args = []
        return EncodedTimeSeriesSet(eltype(y)), encoding_args
    end

    order = sortperm(y)

    return encode_safe_dataset(ES, X_orig[order, :], X_norm[:,order], y[order], args...; kwargs...)
end



function encode_safe_dataset(::EncodeSeparate{true}, X_orig::AbstractMatrix, X_norm::AbstractMatrix, y::AbstractVector, type::String, site_indices::AbstractVector{Index{Int64}}; kwargs...)
    # X_norm has dimension num_elements * numtimeseries

    classes = unique(y)
    states = Vector{PState}(undef, length(y))

    enc_args = []

    for c in classes
        cis = findall(y .== c)
        ets, enc_as = encode_safe_dataset(EncodeSeparate{false}(), X_orig[cis, :], X_norm[:, cis], y[cis], type * " Sep Class", site_indices; kwargs...)
        states[cis] .= ets.timeseries
        push!(enc_args, enc_as)
    end
    
    class_map = countmap(y)
    class_distribution = collect(values(class_map))[sortperm(collect(keys(class_map)))]  # return the number of occurances in each class sorted in order of class index
    return EncodedTimeSeriesSet(states, X_orig, class_distribution), enc_args
end


function encode_safe_dataset(
        ::EncodeSeparate{false}, 
        X_orig::AbstractMatrix, 
        X_norm::AbstractMatrix, 
        y::AbstractVector, 
        type::String, 
        site_indices::AbstractVector{Index{Int64}}; 
        opts::Options=Options(),
        rng=MersenneTwister(1234), 
        class_keys::Dict{T, I}, 
        training_encoding_args=nothing
    ) where {T, I<:Integer}

    # Convert an entire dataset of normalised time series to a corresponding dataset of product states, assumes that input dataset is sorted by class

    verbosity = opts.verbosity
    # pre-allocate
    spl = String.(split(type; limit=2))
    type = spl[1]

    num_ts = size(X_norm)[2] 

    types = ["train", "test", "valid"]
    if type in types
        if verbosity > 0
            if length(spl) > 1
                println("Initialising $type states for class $(y[1]).")
            else
                println("Initialising $type states.")
            end
        end
    else
        error("Invalid dataset type. Must be train, test, or valid.")
    end

    # check data is in the expected range first
    a,b = opts.encoding.range
    name = opts.encoding.name
    if all((a .<= X_norm) .& (X_norm .<= b)) == false
        error("Data must be rescaled between $a and $b before a $name encoding.")
    end

    # check class balance
    cm = countmap(y)
    balanced = all(i-> i == first(values(cm)), values(cm))
    if verbosity > 1 && !balanced
        println("Classes are not Balanced:")
        pretty_table(cm, header=["Class", "Count"])
    end

    # handle the encoding initialisation
    if type == "train"
        encoding_args = opts.encoding.init(X_norm, y; opts=opts)

    elseif !isnothing(training_encoding_args)

        encoding_args=training_encoding_args
    else
        throw(ArgumentError("Can't encode a test or val set without training encoding arguments!"))
    end

    dtype = opts.encoding.iscomplex ? opts.dtype : real(opts.dtype) # encoding can be real with a complex MPS if that's how you roll
    all_product_states = TimeSeriesIterable{dtype}(undef, opts.d, num_ts, size(X_norm,1))
    for i=1:num_ts
        sample_pstate = encode_TS(X_norm[:, i], site_indices, encoding_args; opts=opts)
        # sample_label = y[i]
        # label_idx = class_keys[sample_label]
        # product_state = PState(sample_pstate, sample_label, label_idx)
        all_product_states[:, i, :] = sample_pstate
    end
       
    

    class_map = countmap(y)
    class_distribution = collect(values(class_map))[sortperm(collect(keys(class_map)))] # return the number of occurances in each class sorted in order of class index
    
    return EncodedTimeSeriesSet(all_product_states, X_orig, class_distribution), encoding_args

end
