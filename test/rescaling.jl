using Statistics
dloc =  "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
    X_train_ECG = read(f, "X_train")
    y_train_ECG = read(f, "y_train")
    X_test_ECG = read(f, "X_test")
    y_test_ECG = read(f, "y_test")
close(f)

dloc =  "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
f = jldopen(dloc, "r")
    X_train_IPD = read(f, "X_train")
    y_train_IPD = read(f, "y_train")
    X_test_IPD = read(f, "X_test")
    y_test_IPD = read(f, "y_test")
close(f)

opts = MPSOptions(; d=2, minmax=true, sigmoid_transform=false, verbosity=0)
opts_sig = MPSOptions(; d=2, minmax=true, sigmoid_transform=true, verbosity=0)

threshold=0.1

# 5 rescalesno ub, lb=-0.017352639931286586
X_train_scaled_ECG_sig, X_test_scaled_ECG_sig, norms, oob_rescales_ECG_sig = MPSTime.transform_data(permutedims(X_train_ECG), permutedims(X_test_ECG); opts=opts_sig, test_print_threshold=threshold);
shifts_ECG_sig = hcat(oob_rescales_ECG_sig...);
# X_train_scaled_ECG, X_test_scaled_ECG, norms, oob_rescales = MPSTime.transform_data(permutedims(X_train_ECG), permutedims(X_test_ECG); opts=opts, test_print_threshold=threshold)
lbs_ECG = abs.(shifts_ECG_sig[2,shifts_ECG_sig[2,:] .< 0.0]);
ubs_ECG = shifts_ECG_sig[3,shifts_ECG_sig[3,:] .> 1.0];
print("n: ", size(shifts_ECG_sig, 2), "\nmean lb: ", mean(lbs_ECG;), ", max lb: ",maximum(lbs_ECG))

X_train_scaled_IPD_sig, X_test_scaled_IPD_sig, norms, oob_rescales_IPD_sig = MPSTime.transform_data(permutedims(X_train_IPD), permutedims(X_test_IPD); opts=opts_sig, test_print_threshold=threshold);
shifts_IPD_sig = hcat(oob_rescales_IPD_sig...);
lbs_IPD = abs.(shifts_IPD_sig[2,shifts_IPD_sig[2,:] .< 0.0]);
ubs_IPD = shifts_IPD_sig[3,shifts_IPD_sig[3,:] .> 1.0];
print("n: ", size(shifts_IPD_sig, 2), "\nmean lb: ", mean(lbs_IPD;), ", max lb: ",maximum(lbs_IPD;), "\nmean ub: ", mean(ubs_IPD;), ", max ub: ", maximum(ubs_IPD;))
# X_train_scaled_IPD, X_test_scaled_IPD, norms, oob_rescales = MPSTime.transform_data(permutedims(X_train_IPD), permutedims(X_test_IPD); opts=opts, test_print_threshold=threshold)