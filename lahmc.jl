

function lahmc(U, dU, init_q, epsilon, L, K, beta, n_samples)
    samples = zeros(length(init_q), n_samples)
	Z_chain = [init_q]
	N = length(init_q)
    # current_q = init_q
	p = randn(N)
    accept_count = 0

    for i in 1:n_samples
        proposed_p = NaN
		# use the same random number for comparison for the entire chain
		rand_comparison = rand(1)
		# the current cumulative probability of acceptance
		p_cum = 0
		# the cumulative probability matrix, so we only need to visit each leaf once when recursing
		C = ones((K+1, K+1))*NaN
		# the current set of indices for samples that have not yet been accepted for a transition
		# active_idx = collect(range(nbatch))
        
        for j in 1:K
			proposed_q, proposed_p = leapfrog(Z_chain[end], p, epsilon, L, U, dU)
			push!(Z_chain, proposed_q)

			# recursively calculate the cumulative probability of doing this many leaps
			p_cum, Cl = leap_prob_recurse(Z_chain, C[1:j, 1:j], U)
			C[1:j, 1:j] = Cl
			# find all the samples that did this number of leaps, and update self.state with them
			accept = p_cum >= rand_comparison[1]
			if accept
				accept_count += 1
				# current_q = Z_chain[end]
				break
			end

			# flip the momenutm for any samples that were unable to place elsewhere
			if j == K
				p = -proposed_p
			end

			# counter['L%d'%(j+1)] += len(accepted_idx)
			# state.update(accepted_idx, Z_chain[-1])
			# update the set of active indices, so we don't do simulate trajectories for samples that are already assigned to a state
			# active_idx = active_idx[p_cum.vec() < rand_comparison[active_idx]]
			# if length(active_idx) == 0
			# 	break
            # end
			# Z_chain[-1].active_idx = active_idx
        end
		
		# counter['F'] += len(active_idx)
		# if self.display > 1:
		# 	print("Transition counts "),
		# 	for k in sorted(self.counter.keys()):
		# 		print ("%s:%d"%(k, self.counter[k])),
		# corrupt the momentum
		p = p*sqrt(1-beta) + randn(N)*sqrt(beta)
		# state.update_EV()
        
        samples[:, i] = Z_chain[end]
    end

    acceptRate = accept_count / n_samples
    return samples, acceptRate
end

function leap_prob_recurse(Z_chain, C, U)
	"""
	Recursively compute to cumulative probability of transitioning from
	the beginning of the chain Z_chain to the end of the chain Z_chain.
	"""
	if isnan(C[1,end])
		# we've already visited this leaf
		cumu = C[1,end]
		return cumu, C
    end
	if length(Z_chain) == 2
		# the two states are one apart
		H0 = U(Z_chain[1]) - 0.5 * sum(Z_chain[1].^2)
    	H1 = U(Z_chain[2]) - 0.5 * sum(Z_chain[2].^2)
		p_acc = exp(H0 - H1)
		# p_acc = p_acc[:,active_idx]
		#print C.shape, C[0,-1,:].shape, p_acc.shape, p_acc.ravel().shape
		C[1,end] = p_acc
		return p_acc, C
    end

	cum_forward, Cl = leap_prob_recurse(Z_chain[1:end-1], C[1:end-1, 1:end-1], U)
	C[1:end-1,1:end-1] = Cl
	cum_reverse, Cl = leap_prob_recurse(Z_chain[end:-1:2], C[end:-1:2, end:-1:2], U)
	C[end:-1:2, end:-1:2] = Cl

	H0 = U(Z_chain[1]) - 0.5 * sum(Z_chain[1].^2)
    H1 = U(Z_chain[end]) - 0.5 * sum(Z_chain[end].^2)
	start_state_ratio = exp(H0 - H1)

	prob = min(vcat((1 .- cum_forward, start_state_ratio*(1 .- cum_reverse)))).reshape((1,-1))
	cumu = cum_forward + prob
	C[1,end] = cumu.vec()

	return cumu, C
end

function leapfrog(current_q, current_p, epsilon, L, U, dU)

	p = current_p

    # Initial half-step for momentum
    p = p - 0.5 * epsilon * dU(current_q)

    # Full steps for position and momentum
    proposed_q = copy(current_q)
    for i in 1:L
        proposed_q .= proposed_q .+ epsilon .* p
        if i < L
            p .= p .- epsilon .* dU(proposed_q)
        end
    end

    # Final half-step for momentum
    proposed_p = p .- 0.5 * epsilon .* dU(proposed_q)
	
    return proposed_q, proposed_p
end