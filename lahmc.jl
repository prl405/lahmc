mutable struct LAHMC
    U::Function
    dU::Function
    epsilon::Float64
    L::Int
    K::Int
    beta::Float64
    n_param::Int
    n_samples::Int
    samples::Array{Float64}
    accept_count::Int
    dU_count::Int
end

function LAHMC(U, dU, init_q, epsilon, L, K, beta, n_samples)
	n_param = length(init_q)
    samples = fill(NaN, n_param, n_samples)
	samples[:, 1] = init_q
    LAHMC(U, dU, epsilon, L, K, beta, n_param, n_samples, samples, 0, 0)
end

function sample!(lahmc::LAHMC)
	p = randn(lahmc.n_param)

    for i in 1:lahmc.n_samples-1
		q_chain = [lahmc.samples[:, i]]
		p_chain = [p]
		# use the same random number for comparison for the entire chain
		rand_comparison = rand(1)
		# the current cumulative probability of acceptance
		p_cum = 0
		# the cumulative probability matrix, so we only need to visit each leaf once when recursing
		C = ones((lahmc.K+1, lahmc.K+1))*NaN
        
        for j in 1:lahmc.K
			proposed_q, proposed_p = leapfrog(q_chain[end], p_chain[end], lahmc.epsilon, lahmc.L, lahmc.dU)
			push!(q_chain, proposed_q)
			push!(p_chain, proposed_p)

			# recursively calculate the cumulative probability of doing this many leaps
			p_cum, Cl = leap_prob_recurse(q_chain, p_chain, C[1:j+1, 1:j+1], lahmc.U)
			C[1:j+1, 1:j+1] = Cl
			
			accept = p_cum >= rand_comparison[1]
			if accept
				lahmc.accept_count += 1
				lahmc.samples[:, i+1] = q_chain[end]
				p = p_chain[end]
				break
			end

			# flip the momenutm
			if (j == lahmc.K) & (!accept)
				p = -p
				lahmc.samples[:, i+1] = q_chain[1]
			end
        end
		
		# corrupt the momentum
		p = p*sqrt(1-lahmc.beta) + randn(lahmc.n_param)*sqrt(lahmc.beta)  
    end

    acceptRate = lahmc.accept_count / lahmc.n_samples
    return lahmc.samples, acceptRate
end

function leap_prob_recurse(q_chain, p_chain, C, U)
	"""
	Recursively compute to cumulative probability of transitioning from
	the beginning of the chain q_chain to the end of the chain q_chain.
	"""
	if !isnan(C[1,end])
		# we've already visited this leaf
		cumu = C[1,end]
		return cumu, C
    end
	if length(q_chain) == 2
		# the two states are one apart
		H0 = U(q_chain[1]) + 0.5 * sum(p_chain[1].^2)
    	H1 = U(q_chain[2]) + 0.5 * sum(p_chain[2].^2)
		p_acc = exp(H0 - H1)
		C[1,end] = p_acc
		return p_acc, C
    end

	cum_forward, Cl = leap_prob_recurse(q_chain[1:end-1], p_chain[1:end-1], C[1:end-1, 1:end-1], U)
	C[1:end-1,1:end-1] = Cl
	cum_reverse, Cl = leap_prob_recurse(q_chain[end:-1:2], p_chain[end:-1:2], C[end:-1:2, end:-1:2], U)
	C[end:-1:2, end:-1:2] = Cl

	H0 = U(q_chain[1]) + 0.5 * sum(p_chain[1].^2)
    H1 = U(q_chain[end]) + 0.5 * sum(p_chain[end].^2)
	start_state_ratio = exp(H0 - H1)

	prob = min(1 .- cum_forward, start_state_ratio*(1 .- cum_reverse))
	cumu = cum_forward + prob
	C[1,end] = cumu

	return cumu, C
end

function leapfrog(current_q, current_p, epsilon, L, dU)
	proposed_p = current_p
    proposed_q = current_q

    for i in 1:L
        proposed_p = proposed_p - 0.5 * epsilon * dU(proposed_q)
        proposed_q = proposed_q + epsilon .* proposed_p
        proposed_p = proposed_p - 0.5 * epsilon * dU(proposed_q)
    end
    return proposed_q, proposed_p
end

# function leapfrog(current_q, current_p, epsilon, L, dU)
# 	proposed_p = current_p
#     proposed_q = current_q

#     # Initial half-step for momentum
#     proposed_p = proposed_p .- 0.5 * epsilon * dU(proposed_q)

#     # Full steps for position and momentum    
#     for i in 1:L
#         proposed_q = proposed_q .+ epsilon .* proposed_p
#         if i < L
#             proposed_p = proposed_p .- epsilon .* dU(proposed_q)
#         end
#     end

#     # Final half-step for momentum
#     proposed_p = proposed_p .- 0.5 * epsilon .* dU(proposed_q)
	
#     return proposed_q, proposed_p
# end