using Random
using Plots
using Statistics

include("lahmc.jl")

function U_rough_well(X, scale1, scale2)
    cosX = cos.(X * 2 * pi / scale2)
    E = sum((X.^2) ./ (2 * scale1^2) .+ cosX, dims=1)
    return E
end

function dU_rough_well(X, scale1, scale2)
    sinX = sin.(X * 2 * pi / scale2)
    dEdX = X ./ scale1^2 .- sinX * 2 * pi ./ scale2
    return dEdX
end

function calculate_autocorrelation(samples)
    n_samples = size(samples, 3)
    acf = zeros(n_samples)
    acf[1] = mean(samples.^2)
    
    for lag in 2:(n_samples-1)
        acf[lag] = mean(vcat(samples[:, :, 1:end-lag] .* samples[:, :, lag+1:end]...))
    end
    
    return acf / acf[1]
end

# Example usage
Random.seed!(1234)
theta = [100, 4]
epsilon = 1
L = 10
K = 4
beta = 1
n_samples = 200

DataSize = 2
n_batch = 100
X_init = randn(DataSize, n_batch) * theta[1]

U(X) = U_rough_well(X, theta[1], theta[2])
dU(X) = dU_rough_well(X, theta[1], theta[2])

lahmc = LAHMC(U, dU, X_init, epsilon, L, K, beta, n_samples)
sample!(lahmc)

plt_trace = plot(transpose(lahmc.samples[:,1,:]), xlabel="Sample", ylabel="Chain value", title="Trace plot")

display(plt_trace)
print("Acceptance Rate: ", lahmc.accept_count/(lahmc.n_samples*lahmc.n_batch))

autocorrelation = calculate_autocorrelation(lahmc.samples)

gradient_evaluations = collect(0:(lahmc.dU_count/(length(autocorrelation)-1)):lahmc.dU_count)

plt_ac = plot(gradient_evaluations, autocorrelation, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation")

display(plt_ac)