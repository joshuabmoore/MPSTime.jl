include("../../../LogLoss/RealRealHighDimension.jl");
include("../../../Imputation/imputation.jl");
using JLD2

dloc =  "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
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

d = 20
chi_max=50

opts=MPSOptions(; nsweeps=3, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.05, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
opts_safe, _... = safe_options(opts, nothing, nothing)
if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    #summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    #sweep_summary(info)
end
fstyle = font("sans-serif", 23)
fc = load_forecasting_info_variables(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)
dx=1E-4
mode_range=(-1,1)
xvals=collect(range(mode_range...; step=dx))
mode_index=Index(opts_safe.d)
xvals_enc= [get_state(x, opts_safe, fc[1].enc_args) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];
samps_per_class = [size(f.test_samples, 1) for f in fc]
impute_sites = collect(9:21)
# ts, pred_err, stats, p1_ns = MPS_impute(fc, 0, 2, impute_sites, :directMedian; invert_transform=true, 
#                 NN_baseline=true, X_train=X_train, y_train=y_train, 
#                 n_baselines=1, plot_fits=true, dx=dx, mode_range=mode_range, xvals=xvals, 
#                mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
# plot(p1_ns...)
stats, p2_ns = any_impute_median(fc, 0, 2, impute_sites; NN_baseline=true, X_train=X_train, y_train=y_train, 
    n_baselines=1, plot_fits=true,);
p = plot(p2_ns, xtickfont=fstyle,
    ytickfont=fstyle,
    guidefont=fstyle,
    titlefont=fstyle,
    bottom_margin=10mm, 
    left_margin=10mm,
    xlabel="t");
csdi_test = [-0.9097205 , -1.2692149 , -1.5328441 , -1.6287092 , -1.6287092 ,
    -1.4369789 , -1.2692149 , -0.3345295 ,  0.5482441 ,  1.5392717 ,
    1.275716  ,  1.4230101 ,  0.8625613 ,  0.83981323,  0.7238322 ,
    0.8664967 ,  0.73932505,  0.22383207,  0.2923627 ,  0.42905995,
    0.63270175,  0.6960211 ,  0.12083006, -0.35849577]
brits_test = [-0.9097205 , -1.2692149 , -1.5328441 , -1.6287092 , -1.6287092 ,
    -1.4369789 , -1.2692149 , -0.3345295 ,  0.48031974,  0.6740845 ,
    0.6213454 ,  0.54483604,  0.4386366 ,  0.35488838,  0.33323413,
    0.34027874,  0.33814144,  0.33191878,  0.34368128,  0.3672295 ,
    0.3594555 ,  0.6960211 ,  0.12083006, -0.35849577]
cdrec_test = [-0.90972049, -1.2692149 , -1.5328441 , -1.6287092 , -1.6287092 ,
    -1.4369789 , -1.2692149 , -0.33452949, -0.00660355,  0.86036017,
    0.98848474,  0.99096361,  0.88990993,  0.54205756,  0.22770504,
    0.22570072,  0.23317682,  0.12092547,  0.08198852,  0.39039126,
    0.49292114,  0.69602106,  0.12083006, -0.35849578]
p = plot!(csdi_test, label="CSDI", lw=2);
p = plot!(brits_test, label="BRITS", lw=2);
p = plot!(cdrec_test, label="CDRec", lw=2);
display(p)
savefig("./ipd_all_imputers_sample2_class0.svg")
#range = model_encoding(opts.encoding).range

# X_train_scaled = transform_data(X_train; range=range, minmax_output=opts.minmax)
# X_test_scaled = transform_data(X_test; range=range, minmax_output=opts.minmax)
# svpath = "/Users/joshua/Desktop/QuantumInspiredML/FinalBenchmarks/ItalyPower/demo_imputation_plots/legendre_ns_d$(d)_chi$(chi_max).jld2"
# f = jldopen(svpath, "w")
#     write(f, "X_train_scaled", X_train_scaled)
#     write(f, "y_train", y_train)
#     write(f, "X_test_scaled", X_test_scaled)
#     write(f, "y_test", y_test);
#     write(f, "mps", W)
#     write(f, "opts", opts)
# close(f)
