
function make_folds(X::AbstractMatrix, k::Int; rng::Union{Nothing, AbstractRNG}=nothing)
    if isnothing(rng)
        rng = Xoshiro()
    end
    ninstances = size(X, 1)
    X_idxs = randperm(rng, ninstances)
    # split into k folds
    fold_size = ceil(Int, ninstances/k)
    all_folds = [X_idxs[(i-1)*fold_size+1 : min(i*fold_size, ninstances)] for i in 1:k]
    # build pairs
    X_train_idxs = Vector{Vector{Int}}(undef, k)
    X_val_idxs = Vector{Vector{Int}}(undef, k)
    for i in 1:k
        X_val_idxs[i] = all_folds[i]
        X_train_idxs[i] = vcat(all_folds[1:i-1]..., all_folds[i+1:end]...)
    end
    return zip(X_train_idxs, X_val_idxs)
end

function make_stratified_cvfolds(X::AbstractMatrix, y::AbstractVector, nfolds::Integer; rng=Union{Integer, AbstractRNG}, shuffle::Bool=true)
    stratified_cv = MLJ.StratifiedCV(; nfolds=nfolds,shuffle=shuffle, rng=rng)

    return MLJBase.train_test_pairs(stratified_cv, 1:size(X,1), y)
end

function make_windows(windows::Union{Nothing, AbstractVector, Dict}, pms::Union{Nothing, AbstractVector}, X::AbstractMatrix)

    if ~isnothing(windows) 
        if ~isnothing(pms)
            @show pms
            @show windows
            throw(ArgumentError("Cannot specifiy both windows and pms!"))
        end

        if windows isa Dict
            return vcat([windows[key] for key in sort(collect(keys(windows)))]...)
        end
        return windows
    elseif ~isnothing(pms) 
        ts_length = size(X, 2)
        if eltype(pms) <: Integer
            pms ./ 100
        end
        return [mar(collect(1.:ts_length), pm)[2] for pm in pms]
    else
        throw(ArgumentError("Must specify either windows or pms!"))
        return []
    end
end


function rtime(tstart::Float64)

    return round(time() - tstart; digits=2)
end