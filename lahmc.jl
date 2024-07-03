

function lahmc(U, dU, init_q, epsilon, L, K, beta, n_samples)
    samples = zeros(length(init_q), n_samples)
	samples[:, 1] = init_q
	N = length(init_q)
    # current_q = init_q
	p = randn(N)
    accept_count = 0

    for i in 1:n_samples
		q_chain = i==1 ? [init_q] : [samples[:, i-1]]
		p_chain = [p]
		# use the same random number for comparison for the entire chain
		rand_comparison = rand(1)
		# the current cumulative probability of acceptance
		p_cum = 0
		# the cumulative probability matrix, so we only need to visit each leaf once when recursing
		C = ones((K+1, K+1))*NaN
		# the current set of indices for samples that have not yet been accepted for a transition
		# active_idx = collect(range(nbatch))
        
        for j in 1:K
			proposed_q, proposed_p = leapfrog(q_chain[end], p_chain[end], epsilon, L, dU)
			push!(q_chain, proposed_q)
			push!(p_chain, proposed_p)

			# recursively calculate the cumulative probability of doing this many leaps
			p_cum, Cl = leap_prob_recurse(q_chain, p_chain, C[1:j+1, 1:j+1], U)
			C[1:j+1, 1:j+1] = Cl
			# find all the samples that did this number of leaps, and update self.state with them
			accept = p_cum >= rand_comparison[1]
			if accept
				accept_count += 1
				samples[:, i] = q_chain[end]
				p = p_chain[end]
				break
			end

			# flip the momenutm for any samples that were unable to place elsewhere
			if (j == K) & (!accept)
				p = -p
				samples[:, i] = q_chain[1]
			end
        end
		
		# corrupt the momentum
		p = p*sqrt(1-beta) + randn(N)*sqrt(beta)  
    end

    acceptRate = accept_count / n_samples
    return samples, acceptRate
end

function leap_prob_recurse(q_chain, p_chain, C, U)
	"""
	Recursively compute to cumulative probability of transitioning from
	the beginning of the chain Z_chain to the end of the chain Z_chain.
	"""
	if !isnan(C[1,end])
		# we've already visited this leaf
		cumu = C[1,end]
		return cumu, C
    end
	if length(q_chain) == 2
		# the two states are one apart
		H0 = U(q_chain[1]) - 0.5 * sum(p_chain[1].^2)
    	H1 = U(q_chain[2]) - 0.5 * sum(p_chain[2].^2)
		p_acc = exp(H0 - H1)
		# p_acc = p_acc[:,active_idx]
		#print C.shape, C[0,-1,:].shape, p_acc.shape, p_acc.ravel().shape
		C[1,end] = p_acc
		return p_acc, C
    end

	cum_forward, Cl = leap_prob_recurse(q_chain[1:end-1], p_chain[1:end-1], C[1:end-1, 1:end-1], U)
	C[1:end-1,1:end-1] = Cl
	cum_reverse, Cl = leap_prob_recurse(q_chain[end:-1:2], p_chain[end:-1:2], C[end:-1:2, end:-1:2], U)
	C[end:-1:2, end:-1:2] = Cl

	H0 = U(q_chain[1]) - 0.5 * sum(p_chain[1].^2)
    H1 = U(q_chain[end]) - 0.5 * sum(p_chain[end].^2)
	start_state_ratio = exp(H0 - H1)

	prob = min(1 .- cum_forward, start_state_ratio*(1 .- cum_reverse))
	cumu = cum_forward + prob
	C[1,end] = cumu

	return cumu, C
end

function leapfrog(current_q, current_p, epsilon, L, dU)

	p = current_p

    # Initial half-step for momentum
    p = p .- 0.5 * epsilon * dU(current_q)

    # Full steps for position and momentum
    proposed_q = copy(current_q)
    for i in 1:L
        proposed_q = proposed_q .+ epsilon .* p
        if i < L
            p = p .- epsilon .* dU(proposed_q)
        end
    end

    # Final half-step for momentum
    proposed_p = p .- 0.5 * epsilon .* dU(proposed_q)
	
    return proposed_q, proposed_p
end