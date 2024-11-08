using JLD2
using Plots 
using StatsBase
using Random

struct KeplerInstance
    ts::Vector{Float64}
    class::Int
end

function create_instances(dloc::String)
    # load in the data from dloc and create instances
    f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
    # merge data
    Xs, ys = vcat(X_train, X_test), vcat(y_train, y_test)
    # vector of KeplerInstances
    dset = [KeplerInstance(Xs[i, :], ys[i]) for i in 1:size(Xs, 1)]
    return dset
end

function detect_flat_regions(ts::Vector{Float64})
    flat_regions = []
    for i in 2:length(ts)
        if ts[i] == ts[i-1]
            push!(flat_regions, i)
        end
    end
    return flat_regions
end

function make_windows(ts::Vector{Float64}, window_size::Int, stride::Int)
    n = length(ts)
    windows = [ts[i:i+window_size-1] for i in 1:stride:n-window_size+1]
    return windows
end

function instance_to_dataset(ts::Vector, window_size::Int; stride::Int=window_size, 
    test_size::Int=1, keep_artefact::Bool=false)
    """Takes in time series instance, windows according to desired size, 
    then creates a train/test split.""" 
    # create windows
    windows = make_windows(ts, window_size, stride)
    # check for artefacts
    corr_win_idxs = findall(length.(detect_flat_regions.(windows)) .> 8)
    train_size = length(windows) - length(corr_win_idxs) - test_size
    train_window_idxs = setdiff(collect(1:length(windows)), corr_win_idxs)
    println(length(train_window_idxs))
    train_idxs = sample(train_window_idxs, train_size; replace=false)
    X_train = windows[train_idxs]
    y_train = zeros(Int64, length(X_train))
    test_idxs = setdiff(train_window_idxs, train_idxs)
    if keep_artefact
        test_idxs = vcat(test_idxs, corr_win_idxs)
    end
    X_test = windows[test_idxs]
    y_test = zeros(Int64, length(X_test))
    X_train_mat = vcat(X_train'...)
    X_test_mat = vcat(X_test'...)
    return X_train_mat, y_train, X_test_mat, y_test
end

function get_class_instances(dset::Vector{KeplerInstance}, class::Int)
    """Return a vector of instances pertaining to a single class"""
    class_labels_all = [i.class for i in dset]
    idxs = findall(x -> x .== class, class_labels_all)
    println("Found $(length(idxs)) instances in class $class.")
    class_instances = [dset[i] for i in idxs];
    return class_instances
end

function sample_instances(dset::Vector{KeplerInstance}, class::Int, num_instances::Int)
    """Randomly sample instances from a given class"""
    class_labels = [i.class for i in dset]
    idxs = findall(x -> x .== class, class_labels)
    if length(idxs) < num_instances
        error("Number of instances to be sampled is greater than number of instances in the class")
    end
    sampled_idxs = sample(idxs, num_instances; replace=false)
    return sampled_idxs
end

function sampled_instances_to_joint_dataset(dset::Vector{KeplerInstance}, class::Int, 
        num_instances::Int)
    """Place all of the sampled instances into a single dataset for unsupervised learning"""
    sampled_idxs = sample_instances(dset, class, num_instances)
    X_train = []
    y_train= []
    X_test = []
    y_test = []
    for i in sampled_idxs
        X_tr, y_tr, X_te, y_te = instance_to_dataset(dset[i].ts, 100; keep_artefact=false)
        push!(X_train, X_tr)
        push!(X_test, X_te)
        push!(y_train, y_tr)
        push!(y_test, y_te)
    end
    return vcat(X_train...), vcat(y_train...), vcat(X_test...), vcat(y_test...)
end

function instances_to_joint_dataset(dset::Vector{KeplerInstance})
    """Place all of the sampled instances into a single dataset for unsupervised learning"""
    X_train = []
    y_train= []
    X_test = []
    y_test = []
    for i in eachindex(dset)
        X_tr, y_tr, X_te, y_te = instance_to_dataset(dset[i].ts, 100; keep_artefact=false, test_size=1)
        push!(X_train, X_tr)
        push!(X_test, X_te)
        push!(y_train, y_tr)
        push!(y_test, y_te)
    end
    return vcat(X_train...), vcat(y_train...), vcat(X_test...), vcat(y_test...)
end

function find_impute_region(ts::Vector{Float64})
    """Finds the imputation sites for real-world artefacts"""
    flat_range = detect_flat_regions(ts)
    if length(flat_range) == 0
        @warn "No artefacts detected."
    else
        return vcat(flat_range[1]-1, flat_range) # append previous site
    end
    return nothing
end

c6_instances = get_class_instances(dset, 6)

X_train, y_train, X_test, y_test = instances_to_joint_dataset(c6_instances)


JLD2.@save "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/Data/NASA_KeplerV2/datasets/imputation/c6.jld2" X_train y_train X_test y_test

# dset = create_instances("Data/NASA_KeplerV2/datasets/KeplerLightCurveOrig.jld2")

# # Random.seed!(42)
# # c0_subset = sample_instances(dset, 0, 10)
# X_train, y_train, X_test, y_test = sampled_instances_to_joint_dataset(dset, 2, 10)
# JLD2.@save "c2_subset.jld2" X_train y_train X_test y_test