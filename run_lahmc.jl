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

function plot_histograms(U::Function, lahmc_samples, n_samples::Int, title::String; bin_width=0.2, alpha=0.5, hmc_samples=nothing)
    post_true = fill(NaN, n_samples)
    post_lahmc = fill(NaN, n_samples)
    post_hmc = fill(NaN, n_samples)

    for k in 1:n_samples 
        post_true[k] = U(rand(2)*100) # Only works for Rough Well
        post_lahmc[k] = U(lahmc_samples[:, 1, k])
        if !isnothing(hmc_samples)
            post_hmc[k] = U(hmc_samples[:, 1, k])
        end 
    end

    hist_post = histogram(post_true, bin_width=bin_width, alpha=alpha, title=title, xlabel="Sample", ylabel="Frequency", label="True Posterior")
    histogram!(post_lahmc, bin_witdh=bin_width, alpha=alpha, label="LAHMC Sampled Posterior")
    if !isnothing(hmc_samples)
        histogram!(post_hmc, bin_witdh=bin_width, alpha=alpha, label="HMC Sampled Posterior")
    end
    display(hist_post)
end

function sample_loop(n_chains, U, dU, epsilon, L, K, beta, n_param, n_samples)
    theta = [100, 4]
    samples = fill(NaN, n_param, n_chains, n_samples)
    avg_transitions = fill(0, K+1, n_chains)
    grad_evals = 0

    for i in 1:n_chains
        q_init = randn(n_param)*theta[1]
        lahmc = LAHMC(U, dU, q_init, epsilon, L, K, beta, n_samples)
        result = sample!(lahmc)
        samples[:,i,:] = result.samples
        avg_transitions[:, i] = result.transitions
        grad_evals += result.dU_count
    end
    print("Average Acceptance Rate: ", mean(sum(avg_transitions[1:K,:]./n_samples, dims=1), dims=2)[1,1]) 
    return samples, grad_evals
end

# Input Parameters
theta = [100, 4]
epsilon = 1
L = 10
beta = 0.1
n_samples = 10000
n_param = 2
n_chains = 10

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

plot_histograms(U_rough_well, lahmc_samples, n_samples, "2D Rough Well Histogram"; hmc_samples=hmc_samples)