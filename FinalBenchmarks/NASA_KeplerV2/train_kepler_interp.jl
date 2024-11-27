include("../../LogLoss/RealRealHighDimension.jl")
include("../../Imputation/imputation.jl");
using JLD2
dloc =  "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/Data/NASA_KeplerV2/datasets/imputation/kepler_c0c6split.jld2"
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
encoding = :Legendre_No_Norm 
encode_classes_separately = false
train_classes_separately = false

d = 18
chi_max = 80

opts=MPSOptions(; nsweeps=1, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=1.0, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
opts_safe = safe_options(opts)

W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test, opts; chi_init=4,  test_run=false)
fstyle=font("sans-serif", 23)
fc = init_imputation_problem(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)
dx = 1e-4
mode_range=(-1,1)
xvals=collect(range(mode_range...; step=dx))
mode_index=Index(opts_safe.d)
xvals_enc= [get_state(x, opts_safe, fc[1].enc_args) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];
impute_sites = collect(25:56)
# ts, pred_err, stats, p1_ns = MPS_impute(fc, 0, 45, impute_sites, :directMedian; 
#         invert_transform=true, 
#            NN_baseline=true, X_train=X_train, y_train=y_train, wmad=true, 
#             n_baselines=1, plot_fits=true, dx=dx, mode_range=mode_range, xvals=xvals, 
#             mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
# plot(p1_ns...)
# traj = any_impute_ITS(fc, 0, 2, impute_sites; X_train=X_train)
# plot(traj)

stats, p2_ns = any_impute_median(fc, 0, 2, impute_sites; 
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
name = "c0_i1_2"
savefig("/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/FinalBenchmarks/NASA_KeplerV2/imputation/DemoPlots/NoGT/$name.svg")
# range = model_encoding(opts.encoding).range

# X_train_scaled = transform_data(X_train; range=range, minmax_output=opts.minmax)
# X_test_scaled = transform_data(X_test; range=range, minmax_output=opts.minmax)
# svpath = "/Users/joshua/Desktop/QuantumInspiredML/Data/NASA_KeplerV2/mps_saves/eb_legendre_ns_d$(d)_chi$(chi_max).jld2"
# f = jldopen(svpath, "w")
#     write(f, "X_train_scaled", X_train_scaled)
#     write(f, "y_train", y_train)
#     write(f, "X_test_scaled", X_test_scaled)
#     write(f, "y_test", y_test);
#     write(f, "mps", W)
#     write(f, "opts", opts)
# close(f)
