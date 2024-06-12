using Random
using Plots
using Statistics

include("lahmc.jl")

global dU_count = 0

function U_rough_well(X, scale1, scale2)
    cosX = cos.(X * 2 * pi / scale2)
    E = sum((X.^2) / (2 * scale1^2) .+ cosX)
    return E
end

function dU_rough_well(X, scale1, scale2)
    sinX = sin.(X * 2 * pi / scale2)
    dEdX = X ./ scale1^2 .- sinX * 2 * pi ./ scale2
    return dEdX
end

function U(X)
    return U_rough_well(X, theta[1], theta[2])
end

function dU(X)
    global dU_count += 1
    return dU_rough_well(X, theta[1], theta[2])
end

# function calc_corr(samples)
# 	"""
# 	Calculate autocorrelation given history.  Assumes 0 mean.
# 	"""

# 	T = len(samples)
# 	N = samples[1]['X'].shape[0]
# 	# nbatch = samples[0]['X'].shape[1]

# 	X = zeros((N,nbatch,T))
# 	for tt in 1:T
# 		X[:,:,tt] = samples[tt]['X']
#     end
# 	c = zeros((T-1,))
# 	c[1] = mean(X.^2)
# 	for t_gap in 1:T-1
# 		c[t_gap] = mean(X[:,:,:-t_gap]*X[:,:,t_gap:])
#     end
# 	return c/c[0]
# end

function calculate_autocorrelation(samples)
    n_samples = size(samples, 2)
    mean_samples = mean(samples, dims=2)
    samples_centered = samples .- mean_samples
    acf = zeros(Float64, size(samples, 1), n_samples - 1)
    
    for i in 1:size(samples, 1)
        for lag in 1:n_samples-1
            acf[i, lag] = mean(samples_centered[i, 1:end-lag] .* samples_centered[i, lag+1:end])
        end
        acf[i, :] /= var(samples[i, :])
    end
    
    return acf
end

theta = [100, 4]
epsilon = 1
L = 10
K = 4
beta = 1
n_samples = 500
 
DataSize = 2
# batch_size = 3
X_init = randn(DataSize)*theta[1]

samples, acceptRate = lahmc(U, dU, X_init, epsilon, L, K, beta, n_samples)

plt_samples = plot(transpose(samples), xlabel="Samples", ylabel="Value")

display(plt_samples)
print("Acceptance Rate: ", acceptRate)

# Calculate autocorrelation
autocorrelation = calculate_autocorrelation(samples)
avg_autocorrelation = mean(autocorrelation, dims=1)

# Calculate gradient evaluations
gradient_evaluations = dU_count

# Plot
plt_ac = plot(1:n_samples-1, avg_autocorrelation', title="Gradient Evaluations vs Autocorrelation", xlabel="Sample", ylabel="Autocorrelation")

display(plt_ac)