using Random
using Plots
using Statistics

include("lahmc.jl")
include("distributions.jl")

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

function plot_histograms(U::Function, init_q::Function, lahmc_samples, n_samples::Int, title::String; bin_width=0.2, alpha=0.5, hmc_samples=nothing)
    post_true = fill(NaN, n_samples)
    post_lahmc = fill(NaN, n_samples)
    post_hmc = fill(NaN, n_samples)

    for k in 1:n_samples 
        post_true[k] = U(init_q())
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

function print_transitions(transitions, K, n_samples, n_chains)
    avg_transitions = (sum(transitions, dims=2))./((n_samples-1)*n_chains) # n_samples-1 as first sample is generated
    print("Average transition rates: ")
    for i in 1:K+1
        if i < K + 1
            print("L$(i): $(avg_transitions[i]) ")
        else
            print("F: $(avg_transitions[i])\n")
        end
    end
end

function sample_loop(n_chains, U::Function, dU::Function, init_q::Function, epsilon, L, K, beta, n_param, n_samples)
    samples = fill(NaN, n_param, n_chains, n_samples)
    transitions = fill(0, K+1, n_chains)
    grad_evals = 0

    for i in 1:n_chains
        init_sample = init_q()
        lahmc = LAHMC(U, dU, init_sample, epsilon, L, K, beta, n_samples)
        result = sample!(lahmc)
        samples[:,i,:] = result.samples
        transitions[:, i] = result.transitions
        grad_evals += result.dU_count
    end
    
    print("Average acceptance rate: ", mean(sum(transitions[1:K,:]./(n_samples-1), dims=1), dims=2)[1,1], "\n")
    print_transitions(transitions, K, n_samples, n_chains)
    return samples, grad_evals
end

n_samples = 10000
n_chains = 10
epsilon = 1
L = 10
beta = 0.1

######################## Rough Well #########################

# rw = Rough_Well(2, 100, 4)
# function U(X)
#     return U_rough_well(X, rw)
# end

# function dU(X)
#     return dU_rough_well(X, rw)
# end
# function init_q()
#     return init_rough_well(rw)
# end
# n_param = 2


# lahmc_samples, grad_count = sample_loop(n_chains, U, dU, init_q, epsilon, L, 4, beta, n_param, n_samples)
# autocorrelation_lahmc = calculate_looped_autocorrelation(lahmc_samples)

# gradient_evaluations_lahmc = LinRange(0, grad_count/n_chains, length(autocorrelation_lahmc))

# plt_lahmc_ac = plot(gradient_evaluations_lahmc, autocorrelation_lahmc, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation", label="LAHMC")

# hmc_samples, grad_count = sample_loop(n_chains, U, dU, init_q, epsilon, L, 1, beta, n_param, n_samples)
# autocorrelation_hmc = calculate_looped_autocorrelation(hmc_samples)

# gradient_evaluations_hmc = LinRange(0, grad_count/n_chains, length(autocorrelation_hmc))

# plt_hmc_ac = plot!(gradient_evaluations_hmc, autocorrelation_hmc, label="HMC")

# display(plt_lahmc_ac)

# plot_histograms(U, init_q, lahmc_samples, n_samples, "2D Rough Well Histogram"; hmc_samples=hmc_samples)

######################## Gaussian ##############################

gauss_2d = Gaussian(2, 0.6)

function U(X)
    return U_gaussian(X, gauss_2d)
end

function dU(X)
    return dU_gaussian(X, gauss_2d)
end
function init_q()
    return init_gaussian(gauss_2d)
end
n_param = 2

lahmc_samples, grad_count = sample_loop(n_chains, U, dU, init_q, epsilon, L, 4, beta, n_param, n_samples)
autocorrelation_lahmc = calculate_looped_autocorrelation(lahmc_samples)

gradient_evaluations_lahmc = LinRange(0, grad_count/n_chains, length(autocorrelation_lahmc))

plt_lahmc_ac = plot(gradient_evaluations_lahmc, autocorrelation_lahmc, title="Gradient Evaluations vs Autocorrelation", xlabel="Gradient Evaluations", ylabel="Autocorrelation", label="LAHMC")

hmc_samples, grad_count = sample_loop(n_chains, U, dU, init_q, epsilon, L, 1, beta, n_param, n_samples)
autocorrelation_hmc = calculate_looped_autocorrelation(hmc_samples)

gradient_evaluations_hmc = LinRange(0, grad_count/n_chains, length(autocorrelation_hmc))

plt_hmc_ac = plot!(gradient_evaluations_hmc, autocorrelation_hmc, label="HMC")

display(plt_lahmc_ac)

plot_histograms(U, init_q, lahmc_samples, n_samples, "2D Gaussian Histogram")