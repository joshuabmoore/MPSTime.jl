include("../../../../LogLoss/RealRealHighDimension.jl")
include("../../../../Imputation/imputation.jl");
using JLD2
using ProgressMeter
dloc =  "/Users/joshua/Desktop/QuantumInspiredMLFinal/QuantumInspiredML/Data/syntheticV2/simple/datasets/eta_01_m_3_tau_20.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = zeros(Int64, size(X_train, 1))
    X_test = read(f, "X_test")
    y_test = zeros(Int64, size(X_test, 1))
close(f)

verbosity = 0
test_run = false
track_cost = false
encoding = :legendre_no_norm
encode_classes_separately = false
train_classes_separately = false

d = 12
chi_max = 80

opts=MPSOptions(; nsweeps=3, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=1.0, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)
opts_safe, _... = safe_options(opts, nothing, nothing)

W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test; chi_init=4, opts=opts, test_run=false)
fstyle=font("sans-serif", 23)
fc = init_imputation_problem(W, X_train, y_train, X_test, y_test, opts_safe; verbosity=0)

n_traj = 20
trajectories = zeros(Float64, n_traj, size(X_train,2))
@showprogress for t in 1:n_traj
    trajectories[t, :] = any_impute_ITS(fc, 0, 55, collect(20:80); X_train=X_train, y_train=y_train, rejection_threshold=2.0)
end

function filter_trajectories_by_probas(trajectories::AbstractMatrix{<:Real}, fc::Vector{forecastable}, 
    X_train::AbstractMatrix{<:Real})
    _, norms = transform_train_data(X_train; opts=fc[1].opts)
    # rescale the trajectories so we can compute their probas.
    X_traj_scaled, _ = transform_test_data(trajectories, norms; opts=fc[1].opts)
    sites = siteinds(fc[1].mps)
    num_traj = size(X_traj_scaled, 1)
    probas = zeros(Float64, num_traj)
    @threads for traj in 1:num_traj
        traj_pstate = MPS([itensor(fc[1].opts.encoding.encode(t, fc[1].opts.d, fc[1].enc_args...), sites[i]) for (i,t) in enumerate(X_traj_scaled[traj, :])])
        accum = ITensor(1)
        for i in 1:size(X_traj_scaled, 2)
            accum *= traj_pstate[i] * fc[1].mps[i]
        end
        log_proba = log.(abs2.(accum))
        probas[traj] = log_proba[1]
    end
    # sort trajectories by probabilities
    sorted_probas_idxs = reverse(sortperm(probas))
    trajectories_sorted = trajectories[sorted_probas_idxs, :]

    return trajectories_sorted, probas[sorted_probas_idxs]
end

_, probas_train = filter_trajectories_by_probas(X_train, fc, X_train)
_, probas_test = filter_trajectories_by_probas(X_test, fc, X_train)
sorted_trajectories, probas = filter_trajectories_by_probas(trajectories, fc, X_train)
max_diffs = maximum(diff(sorted_trajectories, dims=2), dims=2)[:]
clean_idxs = findall(x -> x .< 1.5, max_diffs)
unclean_idxs = findall(x -> x .> 1.5, max_diffs)
histogram(probas[clean_idxs], label="Non-Spiky", alpaha=0.3);
histogram!(probas[unclean_idxs], label="Spiky", alpha=0.3)

# create animation
pal = palette(:tab10);
p = plot(xlabel="t", ylabel="x", xlims=(0, 96), grid=:none);
anim = @animate for i in 1:100
    plot(sorted_trajectories[i, :], label="", 
        c=pal[1], title="2.0*WMAD, Trajectory: $i", ylims=(-1, 4.5), dpi = 150, xlabel="t", ylabel="x", grid=:none, lw=2, alpha=1.0)
    #p = vline!([49], lw=3, ls=:dot, c=pal[2], label="");
    # p = vline!([36], lw=3, ls=:dot, c=pal[2], label="");
    # p = vline!([74], lw=3, ls=:dot, c=pal[2], label="");
    # p = vline!([100], lw=3, ls=:dot, c=pal[2], label="");
end;
# gif(anim, "sinusoid_2wmad.gif", fps = 10)

# traj = trajectories[2, :];
# anim = @animate for i in collect(50:100)
#     plot()
#     plot(traj[1:i], xlims=(0, 100), ylims=(-1, 4.5), dpi=150, xlabel="t", label="", c=pal[1])
# end
# gif(anim, "test.gif", fps = 50)
function make_trajecotry_animation(trajectories::AbstractMatrix{<:Real}, n_trajectories::Int)
    n_frames_per_traj = length(50:100)  # Number of frames per trajectory
    anim = @animate for frame_num in 1:(n_trajectories * n_frames_per_traj)
        # calculate the current trajectory index and frame within that trajectory
        traj_idx = div(frame_num - 1, n_frames_per_traj) + 1
        i = 50 + mod(frame_num - 1, n_frames_per_traj)
    
        # Start a new plot with consistent axes and labels
        plt = plot(xlims=(0, 100), ylims=(-1, 4.5),
                   dpi=150, xlabel="t", ylabel="Value", label="")
    
        # Plot previous trajectories with low transparency
        for prev_traj_idx in 1:(traj_idx - 1)
            prev_traj = trajectories[prev_traj_idx, :]
            plot!(plt, prev_traj, c=pal[prev_traj_idx], alpha=0.2, label="", dpi=150)
        end
    
        vline!([49], lw=2, ls=:dot, c=:black, label="")
    
        # Plot the current trajectory up to point i
        traj = trajectories[traj_idx, :]
        plot!(plt, traj[1:i], c=pal[traj_idx], alpha=1.0, label="",
              title="Trajectory $(traj_idx)", dpi=150)
    end
    #gif(anim, "noisy_sine_evolving_trajectories.gif", fps = 50)    
    return anim 
end