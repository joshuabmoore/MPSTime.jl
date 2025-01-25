# tools for inpsecting and pre-processing UCR time series anomaly detection data 
using DelimitedFiles
using Plots
using ProgressMeter
using StatsBase
using PrettyTables
using Plots.PlotMeasures

mutable struct dataset
    name::String
    X_train::Vector{Float64}
    X_test::Vector{Float64}
    X_anomaly::Vector{Float64}
end

function ucr_data_loader(fname::String)

end