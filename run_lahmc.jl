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

function calculate_autocorrelation(samples)
    n_samples = length(samples[1,:])
    acf = zeros(n_samples)
    acf[1] = mean(samples.^2)

    for lag in 2:(n_samples-1)
        acf[lag] = mean(vcat(samples[:, 1:end-lag] .* samples[:, lag+1:end]...))
    end
    
    return acf/acf[1]
end

theta = [100, 4]
epsilon = 1
L = 10
K = 4
beta = 1
n_samples = 5000
 
DataSize = 2
X_init = randn(DataSize)*theta[1]

samples, acceptRate = lahmc(U, dU, X_init, epsilon, L, K, beta, n_samples)

plt_trace = plot(transpose(samples), xlabel="Sample", ylabel="Chain value", title="Trace plot")

display(plt_trace)
print("Acceptance Rate: ", acceptRate)

autocorrelation = calculate_autocorrelation(samples)

gradient_evaluations = collect(0:(dU_count/(length(autocorrelation)-1)):dU_count)

plt_ac = plot(gradient_evaluations, autocorrelation, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation")

display(plt_ac)