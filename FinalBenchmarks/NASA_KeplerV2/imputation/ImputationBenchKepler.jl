using Pkg
Pkg.activate(".")
include("../../../LogLoss/RealRealHighDimension.jl")
include("../../../Interpolation/imputation.jl");
using JLD2
using DelimitedFiles
using Plots

# load the original split
dloc = "Data/NASA_KeplerV2/datasets/classification/KeplerBinaryOrigUnbal.jld2"

f = jldopen(dloc, "r")
    X_train = read(f, "X_train")[:, 1:100]
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")[:, 1:100]
    y_test = read(f, "y_test")
close(f)

c0_idxs_tr = findall(x -> x .== 0, y_train)
c0_idxs_te = findall(x -> x .== 0, y_test)
X_train_c0 = X_train[c0_idxs_tr, 1:100]
X_test_c0 = X_test[c0_idxs_te, 1:100]
y_train_c0 = y_train[c0_idxs_tr]
y_test_c0 = y_test[c0_idxs_te]

# recombine the original train/test split
Xs = vcat(X_train, X_test)
ys = vcat(y_train, y_test)

# define structs for the results
struct WindowScores
    mps_scores::Vector{Float64}
    nn_scores::Vector{Float64}
end

struct InstanceScores
    pm_scores::Vector{WindowScores}
end

struct FoldResults 
    fold_scores::Vector{InstanceScores}
end

Rdtype = Float64

# training related stuff
verbosity = 0
test_run = false
track_cost = false
encoding = :legendre_no_norm
encode_classes_separately = false
train_classes_separately = false

d = 8 #10
chi_max=50 #20
nsweeps = 3 #5
eta = 1.0

opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
            bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
            encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
            exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
opts_safe, _... = safe_options(opts, nothing, nothing)

W, info, train_states, test_states = fitMPS(X_train_c0, y_train_c0, X_test_c0, y_test_c0; chi_init=4, opts=opts, test_run=false)
fstyle=font("sans-serif", 23)
fc = load_forecasting_info_variables(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)
dx = 1e-4
mode_range=(-1,1)
xvals=collect(range(mode_range...; step=dx))
mode_index=Index(opts_safe.d)
xvals_enc= [get_state(x, opts_safe) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];
interp_sites = collect(11:42) # 11 - 42
stats, p2_ns = any_impute_median(fc, 0, 1, interp_sites; 
    NN_baseline=true, X_train=X_train,
    y_train=y_train, 
    n_baselines=1, plot_fits=true)
p = plot(p2_ns, xtickfont=fstyle,
    ytickfont=fstyle,
    guidefont=fstyle,
    titlefont=fstyle,
    bottom_margin=10mm, 
    left_margin=10mm,
    xlabel="t")