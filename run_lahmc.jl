using Random
using Plots
using Statistics
using DataFrames

include("lahmc.jl")
include("distributions.jl")


function autocorrelation(samples)
    n_samples = size(samples, 3)
    acf = zeros(n_samples-1)
    acf[1] = mean(samples.^2)
    
    for lag in 2:(n_samples-1)
        acf[lag] = mean(samples[:, :, 1:end-lag] .* samples[:, :, lag+1:end])
    end
    
    return acf / acf[1]
end

function plot_histograms(U::Function, init_q::Function, lahmc_samples, n_samples::Int, title::String; bin_width=0.2, alpha=0.5, hmc_samples=nothing, data=nothing)
    post_true = isnothing(data) ? fill(NaN, n_samples) : data
    post_lahmc = fill(NaN, n_samples)
    post_hmc = fill(NaN, n_samples)

    for k in 1:n_samples 
        if isnothing(data)
            post_true[k] = U(init_q())
        end
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

function calculate_transitions(transitions, K, n_samples, n_chains)
    avg_transitions = (sum(transitions, dims=2))./((n_samples-1)*n_chains) # n_samples-1 as first sample is generated
    
    track_rates = Dict{String, Float64}()
    track_rates["avg_accept"] = mean(sum(transitions[1:K,:]./(n_samples-1), dims=1), dims=2)[1,1]
    for i in 1:K+1
        if i < K + 1
            track_rates["L$(i)"] = avg_transitions[i]
        else
            track_rates["F"] = avg_transitions[i]
        end
    end
    return track_rates
end

function sample_loop(n_chains, U::Function, dU::Function, init_q::Function, epsilon, L, K, beta, n_param, n_samples)
    samples = fill(NaN, n_param, n_chains, n_samples)
    trans = fill(0, K+1, n_chains)
    grad_evals = 0

    for i in 1:n_chains
        init_sample = init_q()
        lahmc = LAHMC(U, dU, init_sample, epsilon, L, K, beta, n_samples)
        result = sample!(lahmc)
        samples[:,i,:] = result.samples
        trans[:, i] = result.transitions
        grad_evals += result.dU_count
    end
    
    output = Dict{Any, Any}()
    output[:samples] = samples
    output[:grad_evals] = grad_evals
    output[:trans] = calculate_transitions(trans, K, n_samples, n_chains)
    return output
end
