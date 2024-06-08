

function lahmc(U, dU, init_q, epsilon, L, K, beta, n_samples)
    samples = zeros(length(init_q), n_samples)
    current_q = init_q
    accept_count = 0
	p = randn(length(current_q))
    # nbatch = shape(init_q)

    for i in 1:n_samples
        proposed_q, accept = leapfrog(current_q, epsilon, L, U, dU)
		

        if accept
			current_q = proposed_q
			accept_count += 1
		else
            # first do the HMC part of the step
		    Z_chain = [proposed_q.copy(),]
		    # use the same random number for comparison for the entire chain
		    rand_comparison = rand(1)
		    # the current cumulative probability of acceptance
		    p_cum = 0
		    # the cumulative probability matrix, so we only need to visit each leaf once when recursing
		    C = ones((K+1, K+1))*NaN
		    # the current set of indices for samples that have not yet been accepted for a transition
		    # active_idx = collect(range(nbatch))
            
            for kk in 1:K
		    	Z_chain.append(leapfrog(Z_chain[-1].copy(), epsilon, L, U, dU))

		    	# recursively calculate the cumulative probability of doing this many leaps
		    	p_cum, Cl = leap_prob_recurse(Z_chain, C[:kk+2, :kk+2])
		    	C[:kk+2, :kk+2] = Cl

		    	# find all the samples that did this number of leaps, and update self.state with them
		    	accept = p_cum.vec() >= rand_comparison

				if accept
					accept_count += 1
					current_q = Z_chain[end]
					break
				end
		    	# counter['L%d'%(kk+1)] += len(accepted_idx)
		    	# state.update(accepted_idx, Z_chain[-1])

		    	# update the set of active indices, so we don't do simulate trajectories for samples that are already assigned to a state
		    	# active_idx = active_idx[p_cum.vec() < rand_comparison[active_idx]]

		    	# if length(active_idx) == 0
		    	# 	break
                # end
		    	# Z_chain[-1].active_idx = active_idx
            end
		    # flip the momenutm for any samples that were unable to place elsewhere
		    state.V[:,active_idx] = -state.V[:,active_idx]
		    # counter['F'] += len(active_idx)

		    # if self.display > 1:
		    # 	print("Transition counts "),
		    # 	for k in sorted(self.counter.keys()):
		    # 		print ("%s:%d"%(k, self.counter[k])),

		    # corrupt the momentum
		    p = p*sqrt(1-beta) + randn(N)*sqrt(beta)
		    state.update_EV()
        end

        samples[:, i] = current_q
    end

    acceptRate = accept_count / n_samples
    return samples, acceptRate
end

function leap_prob_recurse(Z_chain, C)
	"""
	Recursively compute to cumulative probability of transitioning from
	the beginning of the chain Z_chain to the end of the chain Z_chain.
	"""
	if isfinite(C[1,end])
		# we've already visited this leaf
		cumu = C[1,end].reshape((1,-1))
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

	cum_forward, Cl = leap_prob_recurse(Z_chain[:end], C[:end,:end])
	C[:end,:end] = Cl
	cum_reverse, Cl = leap_prob_recurse(Z_chain[:1:end], C[:1:end,:1:end])
	C[:1:end,:1:end] = Cl

	# H0 = self.H(Z_chain[1])
	# H1 = self.H(Z_chain[end])

	# Ediff = H0 - H1
	# Ediff = Ediff[:,active_idx]
	# start_state_ratio = exp(Ediff)

	H0 = U(Z_chain[1]) - 0.5 * sum(Z_chain[1].^2)
    H1 = U(Z_chain[end]) - 0.5 * sum(Z_chain[end].^2)
	start_state_ratio = exp(H0 - H1)

	prob = min(vcat((1. - cum_forward, start_state_ratio*(1. - cum_reverse)))).reshape((1,-1))
	cumu = cum_forward + prob
	C[1,end] = cumu.vec()

	return cumu, C
end

function leapfrog(current_q, epsilon, L, U, dU)
    p = randn(length(current_q))
    current_p = p

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

    p = -p

    # Accept or reject the new state
    current_H = U(current_q) - 0.5 * sum(current_p.^2)
    proposed_H = U(proposed_q) - 0.5 * sum(proposed_p.^2)
    accept = rand() < exp(current_H - proposed_H)
    return proposed_q, accept
end