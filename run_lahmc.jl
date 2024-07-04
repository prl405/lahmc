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

function calculate_looped_autocorrelation(samples)
    n_samples = size(samples, 3)
    acf = zeros(n_samples)
    acf[1] = mean(samples.^2)
    
    for lag in 2:(n_samples-1)
        acf[lag] = mean(vcat(samples[:, :, 1:end-lag] .* samples[:, :, lag+1:end]...))
    end
    
    return acf / acf[1]
end

function sample_loop(n_chains, U, dU, epsilon, L, K, beta, n_param, n_samples)
    theta = [100, 4]
    samples = fill(NaN, n_param, n_chains, n_samples)
    avg_accptRate = fill(NaN, n_chains)

    for i in 1:n_chains
        q_init = randn(n_param)*theta[1]
        lahmc = LAHMC(U, dU, q_init, epsilon, L, K, beta, n_samples)
        chain, acceptRate = sample!(lahmc)
        samples[:,i,:] = chain
        avg_accptRate[i] = acceptRate
    end
    print("Average Acceptance Rate: ", mean(avg_accptRate))   
    return samples
end

theta = [100, 4]
epsilon = 1
L = 10
beta = 1
n_samples = 200
n_param = 2

autocorrelation = calculate_looped_autocorrelation(sample_loop(100, U, dU, epsilon, L, 4, beta, n_param, n_samples))

gradient_evaluations = collect(0:(dU_count/(length(autocorrelation)-1)):dU_count)

plt_ac = plot(gradient_evaluations, autocorrelation, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation")

dU_count = 0

autocorrelation = calculate_looped_autocorrelation(sample_loop(100, U, dU, epsilon, L, 1, beta, n_param, n_samples))

gradient_evaluations = collect(0:(dU_count/(length(autocorrelation)-1)):dU_count)

plt_hmc_ac = plot!(gradient_evaluations, autocorrelation)

display(plt_ac)