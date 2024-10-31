include("../../LogLoss/RealRealHighDimension.jl")
include("../../Interpolation/imputation.jl")
using JLD2

dloc =  "Data/epilepsy/datasets/Epilepsy2.jld2"
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

d = 10
chi_max=20
nsweeps=5

opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.5, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4, log_level=0)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    #sweep_summary(info)
end
opts_safe, _... = safe_options(opts, nothing, nothing)
fstyle=font("sans-serif", 23)
fc = load_forecasting_info_variables(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)
dx = 5e-4
mode_range=(-1,1)
xvals=collect(range(mode_range...; step=dx))
mode_index=Index(opts_safe.d)
xvals_enc= [get_state(x, opts_safe) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];
interp_sites = collect(50:70)
stats, p1_ns = any_impute_single_timeseries(fc, 0, 1, interp_sites, :directMedian; 
        invert_transform=true, 
           NN_baseline=true, X_train=X_train, y_train=y_train, 
            n_baselines=1, plot_fits=true, dx=dx, mode_range=mode_range, xvals=xvals, 
            mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it)
plot(p1_ns...)
# stats, p1_ns = any_impute_median(fc, 1, 900, interp_sites; X_train=X_train, y_train=y_train, 
#     NN_baseline=true,
#     plot_fits=true,
#     get_metrics=true, # whether to compute goodness of fit metrics
#     full_metrics=false, # whether to compute every metric or just MAE
#     print_metric_table=false,
#     wmad=true)
# plot(p1_ns)

range = model_encoding(opts.encoding).range

X_train_scaled = transform_data(X_train; range=range, minmax_output=opts.minmax)
X_test_scaled = transform_data(X_test; range=range, minmax_output=opts.minmax)
svpath = "Data/epilepsy/mps_saves/$(nsweeps)_sw_legendre_ns_d$(d)_chi$(chi_max).jld2"
f = jldopen(svpath, "w")
    write(f, "X_train_scaled", X_train_scaled)
    write(f, "y_train", y_train)
    write(f, "X_test_scaled", X_test_scaled)
    write(f, "y_test", y_test);
    write(f, "mps", W)
    write(f, "opts", opts)
close(f)
