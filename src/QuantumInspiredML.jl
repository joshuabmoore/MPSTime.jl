module QuantumInspiredML

# Import these two libraries First and in this order
using GenericLinearAlgebra
using MKL

using Strided # strided array support
using ITensors 
using NDTensors # Library that ITensors is built on, used for some hacks
import Base.* # Allows overloading of NDTensor * NDTensor in loss_functions

using Random
using StableRNGs # Used for some hyperparameter tuning/MLJ


# Allow Optimisation algorithms from external libraries
using Optim
using OptimKit
using Normalization # Standardised normalisation by Brendan :). Used to do the preprocessing / denormalising steps



using LegendrePolynomials # For legendre polynomial basis
using KernelDensity # Used for histogram-derived data-driven bases


#TODO remove some of these
using QuadGK # used to be used in imputation.jl, may be redundant
using NumericalIntegration # Used extensively in imputation.jl
using Integrals # Not sure if used
import SciMLBase # used in encodings for SL coeffs, (remove)

# allows converting BBOpt to string in structs.jl, and converting the abstract MPSOptions to a concrete Options type
import Base.convert

import MLBase # For some useful multivariate stats calculations
using StatsBase: countmap, sample
using Distributions # Used to compute stats in stats.jl and the imputation library


using JLD2 # for save/load
using StatsPlots, Plots, Plots.PlotMeasures # Plotting in imputation.jl
using PrettyTables # Nicely formatted training / imputation text output 
import ProgressMeter 

using Tables # Used for MLJ
using MLJ # Used for MLJ Integration
import MLJModelInterface # MLJ Integration
import MLJTuning # Custom imputation tuning algorithm
using MLJParticleSwarmOptimization # Used in hyperparameter tuning

# Custom Data Structures and types - include first
include("Structs/structs.jl")
include("Structs/options.jl")

# Functions and structs used to define basis functions / encodings
include("Encodings/encodings.jl")
include("Encodings/bases.jl")
include("Encodings/splitbases.jl")


include("summary.jl") # Some useful stats functions
include("utils.jl") # Some utils used by the entire library

include("Training/loss_functions.jl") # Where loss functions and the LossFunction type are defined
include("Training/RealRealHighDimension.jl"); # The training algorithm, fitMPS and co

# Imputation
include("Imputation/imputation.jl") # Some structs, and scaffolds for setting up and solving ImpuationProblems
include("Imputation/imputationMetrics.jl"); # Metrics used to measure imputation performance
include("Imputation/imputationUtils.jl"); # contains the functions to project states onto an MPS / get most likely states for a region
include("Imputation/samplingUtils.jl"); # contains functions to compute a rdm matrix from an MPS, and pretty much every way you can think of to sample from it



# MLJ
include("MLJIntegration/MLJ_integration.jl") # MLJ Integration
include("MLJIntegration/MLJUtils.jl")
include("MLJIntegration/imputation_hyperopt_hack.jl") # Hyperoptimising imputation using MAE. MLJ was not designed for this at all 


export 
    # Options struct
    MPSOptions,

    # functions to construct Bases
    stoudenmire,
    fourier,
    legendre,
    legendre_no_norm,
    sahand,
    uniform,
    function_basis,
    hist_split,
    uniform_split,

    # nicley formatted training summaries
    get_training_summary,
    get_predictions,
    plot_training_summary,

    # Training functions
    fitMPS, # gotta fit those MPSs somehow

    # Imputation
    init_imputation_problem, # generate and collect data necessary for imputation
    MPS_impute, # The main imputation function, all functionality can be accessed hyperparameter
    get_rdms, # compute the reduced density matrix (as a cdf) at every site. Useful for data vis/debugging weird imputations
    
    # MLJ 
    MPSClassifier
end