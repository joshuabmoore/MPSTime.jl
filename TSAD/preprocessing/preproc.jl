# tools for inpsecting and pre-processing UCR time series anomaly detection data 
using DelimitedFiles
using Plots
using ProgressMeter
using StatsBase
using PrettyTables
using Plots.PlotMeasures
using JLD2


function ucr_data_loader(fname::String)
    data_f = readdlm(fname)
    fname_split = split.(split(fname, "_"), ".")
    dname = fname_split[4][1]
    train_end = parse(Int, fname_split[5][1])
    anomaly_start = parse(Int, fname_split[6][1])
    anomaly_end = parse(Int, fname_split[7][1])
    data = size(data_f, 1) > size(data_f, 2) ? data_f[:, 1] : data_f[1, :];
    X_train = data[1:train_end] # inclusive of train end
    X_test = data[train_end+1:end];
    anom_range = (anomaly_start-length(X_train):anomaly_end-length(X_train))
    return X_train, X_test, anom_range
end