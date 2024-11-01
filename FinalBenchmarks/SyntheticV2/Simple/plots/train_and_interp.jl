include("../../../../LogLoss/RealRealHighDimension.jl")
include("../../../../Interpolation/imputation.jl");
using JLD2
dloc =  "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/Data/syntheticV2/complex/datasets/eta_0.1_m_disc_range_3_phi_cont_range_0_6.283185307179586_tau_disc_range_3_LARGE.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
encoding = :legendre_no_norm
encode_classes_separately = false
train_classes_separately = false

d = 12
chi_max = 40

opts=MPSOptions(; nsweeps=10, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.05, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
opts_safe, _... = safe_options(opts, nothing, nothing)

W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; chi_init=4, opts=opts, test_run=false)
fstyle=font("sans-serif", 23)
fc = load_forecasting_info_variables(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)
dx = 1e-4
mode_range=(-1,1)
xvals=collect(range(mode_range...; step=dx))
mode_index=Index(opts_safe.d)
xvals_enc= [get_state(x, opts_safe) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];
interp_sites = collect(15:85)
# stats, p1_ns = any_impute_single_timeseries(fc, 0, 1, interp_sites, :directMedian; 
#         invert_transform=true, 
#            NN_baseline=true, X_train=X_train, y_train=y_train, wmad=true, 
#             n_baselines=1, plot_fits=true, dx=dx, mode_range=mode_range, xvals=xvals, 
#             mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
stats, p1_ns = any_impute_median(fc, 0, 12, interp_sites; invert_transform=true, 
           NN_baseline=false, X_train=X_train, y_train=y_train,n_baselines=1, plot_fits=true)
plot(p1_ns...,
    xtickfont=fstyle, ytickfont=fstyle, titlefont=fstyle,
    guidefont=fstyle, title="$(floor(length(interp_sites)/96 * 100))%",
    bottom_margin=10mm, left_margin=10mm)