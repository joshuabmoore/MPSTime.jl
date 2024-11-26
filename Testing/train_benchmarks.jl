include("../LogLoss/RealRealHighDimension.jl")
include("../Imputation/imputation.jl");
using JLD2
using DelimitedFiles
using Plots


# GenericLinearAlgebra.LinearAlgebra.BLAS.set_num_threads(1)
verbosity = -10
test_run = false
track_cost = false
encoding = :legendre_no_norm
projected_basis = true
encode_classes_separately = false
train_classes_separately = false
eta = 0.5 # 0.1
nsweeps = 5 # 3

ipd_dloc = "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
ipd_resamp_folds_path = "FinalBenchmarks/ItalyPower/Julia/ipd_resample_folds_julia_idx.jld2"
ipd_windows_path = "FinalBenchmarks/ItalyPower/Julia/ipd_windows_julia_idx.jld2"

ecg_dloc = "Data/ecg200/datasets/ecg200.jld2"
ecg_resamp_folds_path = "FinalBenchmarks/ECG200/Julia/resample_folds_julia_idx.jld2"
ecg_windows_path = "FinalBenchmarks/ECG200/Julia/windows_julia_idx.jld2"

f = jldopen(ecg_dloc, "r")
    ecg_X_train = read(f, "X_train")
    ecg_y_train = read(f, "y_train")
    ecg_X_test = read(f, "X_test")
    ecg_y_test = read(f, "y_test")
close(f)

f = jldopen(ipd_dloc, "r")
    ipd_X_train = read(f, "X_train")
    ipd_y_train = read(f, "y_train")
    ipd_X_test = read(f, "X_test")
    ipd_y_test = read(f, "y_test")
close(f)


tstart=time()
# low d, chi
d = 3
chi_max = 15 # 

opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max, update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0, projected_basis=projected_basis
)

svpath_ecg = "Data/ecg200/mps_saves/PL_d$(d)_chi$(chi_max).jld2"
svpath_ipd = "Data/italypower/mps_saves/PL_d$(d)_chi$(chi_max).jld2"

println("t: $(round(time()-tstart; digits=3)) ECG, d: $(d) chi: $(chi_max)")
W, _... = fitMPS( ecg_X_train, ecg_y_train,ecg_X_test, ecg_y_test, opts; test_run=false)
f = jldopen(svpath_ecg, "w")
        write(f, "mps", W)
        write(f, "opts", opts)
close(f)

println("t: $(round(time()-tstart; digits=3)) IPD d: $(d) chi: $(chi_max)")

W, _... = fitMPS( ipd_X_train, ipd_y_train,ipd_X_test, ipd_y_test, opts; test_run=false)
f = jldopen(svpath_ipd, "w")
        write(f, "mps", W)
        write(f, "opts", opts)
close(f)




# # mid d, chi
# d = 10
# chi_max = 20 # 

# opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max, update_iters=1, verbosity=verbosity, loss_grad=:KLD,
#     bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
#     encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
#     exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0, projected_basis=projected_basis
# )
# svpath_ecg = "Data/ecg200/mps_saves/L_d$(d)_chi$(chi_max).jld2"
# svpath_ipd = "Data/italypower/mps_saves/L_d$(d)_chi$(chi_max).jld2"

# println("t: $(round(time()-tstart; digits=3)) ECG, d: $(d) chi: $(chi_max)")
# W, _... = fitMPS( ecg_X_train, ecg_y_train,ecg_X_test, ecg_y_test, opts; test_run=false)
# f = jldopen(svpath_ecg, "w")
#         write(f, "mps", W)
#         write(f, "opts", opts)
# close(f)

# println("t: $(round(time()-tstart; digits=3)) IPD d: $(d) chi: $(chi_max)")

# W, _... = fitMPS( ipd_X_train, ipd_y_train,ipd_X_test, ipd_y_test, opts; test_run=false)
# f = jldopen(svpath_ipd, "w")
#         write(f, "mps", W)
#         write(f, "opts", opts)
# close(f)

# # high d, chi
# d = 20
# chi_max = 40 # 

# opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max, update_iters=1, verbosity=verbosity, loss_grad=:KLD,
#     bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
#     encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
#     exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0, projected_basis=projected_basis
# )
# svpath_ecg = "Data/ecg200/mps_saves/L_d$(d)_chi$(chi_max).jld2"
# svpath_ipd = "Data/italypower/mps_saves/L_d$(d)_chi$(chi_max).jld2"

# println("t: $(round(time()-tstart; digits=3)) ECG, d: $(d) chi: $(chi_max)")
# W, _... = fitMPS( ecg_X_train, ecg_y_train,ecg_X_test, ecg_y_test, opts; test_run=false)
# f = jldopen(svpath_ecg, "w")
#         write(f, "mps", W)
#         write(f, "opts", opts)
# close(f)

# println("t: $(round(time()-tstart; digits=3)) IPD d: $(d) chi: $(chi_max)")

# W, _... = fitMPS( ipd_X_train, ipd_y_train,ipd_X_test, ipd_y_test, opts; test_run=false)
# f = jldopen(svpath_ipd, "w")
#         write(f, "mps", W)
#         write(f, "opts", opts)
# close(f)