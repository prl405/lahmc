using Random
using Plots
using Statistics

include("lahmc.jl")
include("distributions.jl")

function U(X)
    return U_rough_well(X, theta[1], theta[2])
end

function dU(X)
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
    acf = zeros(n_samples-1)
    acf[1] = mean(samples.^2)
    
    for lag in 2:(n_samples-2)
        acf[lag] = mean(samples[:, :, 1:end-lag] .* samples[:, :, lag+1:end])
    end
    
    return acf / acf[1]
end

function sample_loop(n_chains, U, dU, epsilon, L, K, beta, n_param, n_samples)
    theta = [100, 4]
    samples = fill(NaN, n_param, n_chains, n_samples)
    avg_accptRate = fill(NaN, n_chains)
    grad_evals = 0

    for i in 1:n_chains
        q_init = randn(n_param)*theta[1]
        lahmc = LAHMC(U, dU, q_init, epsilon, L, K, beta, n_samples)
        chain, acceptRate = sample!(lahmc)
        samples[:,i,:] = chain
        avg_accptRate[i] = acceptRate
        grad_evals += lahmc.dU_count
    end
    print("Average Acceptance Rate: ", mean(avg_accptRate))   
    return samples, grad_evals
end

# Input Parameters
theta = [100, 4]
epsilon = 1
L = 10
beta = 0.1
n_samples = 200
n_param = 2
n_chains = 100

# LAHMC

lahmc_samples, grad_count = sample_loop(n_chains, U, dU, epsilon, L, 4, beta, n_param, n_samples)
autocorrelation_lahmc = calculate_looped_autocorrelation(lahmc_samples)

gradient_evaluations_lahmc = LinRange(0, grad_count/n_chains, length(autocorrelation_lahmc))

plt_lahmc_ac = plot(gradient_evaluations_lahmc, autocorrelation_lahmc, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation", label="LAHMC")


# HMC

hmc_samples, grad_count = sample_loop(n_chains, U, dU, epsilon, L, 1, beta, n_param, n_samples)
autocorrelation_hmc = calculate_looped_autocorrelation(hmc_samples)

gradient_evaluations_hmc = LinRange(0, grad_count/n_chains, length(autocorrelation_hmc))

plt_hmc_ac = plot!(gradient_evaluations_hmc, autocorrelation_hmc, label="HMC")

display(plt_lahmc_ac)

post = fill(NaN, n_samples)
for k in 1:n_samples 
    post[k] = U_rough_well(rand(2)*100) 
end

post_lahmc = fill(NaN, n_samples)
for k in 1:n_samples 
    post_lahmc[k] = U_rough_well(lahmc_samples[:, 1, k]) 
end

sc_post = histogram(post, bins=30, alpha=0.5, title="Samples on posterior", xlabel="Sample", ylabel="Value", label="True Posterior")
sc_lahmc_post = histogram!(post_lahmc, bins=30, alpha=0.5, label="Sampled Posterior")
display(sc_post)