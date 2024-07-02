mutable struct LAHMC
    U::Function
    dU::Function
    epsilon::Float64
    L::Int
    K::Int
    beta::Float64
    n_param::Int
    n_samples::Int
	n_batch::Int
    samples::Array{Float64}
    accept_count::Int
    dU_count::Int
end

function LAHMC(U, dU, init_q, epsilon, L, K, beta, n_samples)
	n_param = size(init_q, 1)
    n_batch = size(init_q, 2)
    samples = fill(NaN, n_param, n_batch, n_samples)
    samples[:, :, 1] = init_q
    LAHMC(U, dU, epsilon, L, K, beta, n_param, n_samples, n_batch, samples, 0, 0)
end

function call_dU!(lahmc::LAHMC)
    lahmc.dU_count += 1
    return lahmc.dU
end

function sample!(lahmc::LAHMC)
    p = randn(lahmc.n_param, lahmc.n_batch)

    for i in 1:(lahmc.n_samples-1)
        q_chain = [lahmc.samples[:, :, i]]
        p_chain = [p]
        rand_comparison = rand(1, lahmc.n_batch)
        p_cum = zeros(1, n_batch)
        C = fill(NaN, lahmc.K+1, lahmc.K+1, lahmc.n_batch)
        active_idx = collect(1:lahmc.n_batch)
        
        for j in 1:lahmc.K
            proposed_q, proposed_p = leapfrog(q_chain[end], p_chain[end], lahmc.epsilon, lahmc.L, call_dU!(lahmc))
            push!(q_chain, proposed_q)
            push!(p_chain, proposed_p)
            
            p_cum, Cl = leap_prob_recurse(q_chain, C[1:j+1, 1:j+1, active_idx], active_idx, lahmc.U)
            C[1:j+1, 1:j+1, active_idx] = Cl
            
            accepted_idx = active_idx[vcat(p_cum...) .>= rand_comparison[active_idx]]
            lahmc.accept_count += length(accepted_idx)

            if length(accepted_idx) != 0
                lahmc.samples[:, accepted_idx, i+1] = q_chain[end][:, accepted_idx]
                p[:, accepted_idx] = p_chain[end][:, accepted_idx]
            end

            active_idx = active_idx[vcat(p_cum...) .< rand_comparison[active_idx]]

			if length(active_idx) == 0
				break
            end
        end

        if length(active_idx) != 0
            p[:, active_idx] = -p[:, active_idx]
            lahmc.samples[:, active_idx, i+1] = lahmc.samples[:, active_idx, i]
        end

        p = p * sqrt(1 - lahmc.beta) + randn(lahmc.n_param, lahmc.n_batch) * sqrt(lahmc.beta)
        
    end
end

function leap_prob_recurse(q_chain, C, active_idx, U)
	"""
	Recursively compute to cumulative probability of transitioning from
	the beginning of the chain q_chain to the end of the chain q_chain.
	"""
	if !isnan(C[1,end, 1])
		# we've already visited this leaf
		cumu = C[1,end, :]
		return cumu', C
    end
    
	if length(q_chain) == 2
		# the two states are one apart
		H0 = U(q_chain[1]) .- 0.5 * sum(q_chain[1].^2, dims=1)
    	H1 = U(q_chain[2]) .- 0.5 * sum(q_chain[2].^2, dims=1)
        diff = H0 .- H1
        p_acc = ones((1, size(diff, 2)))
		p_acc[diff.<0] = exp.(diff[diff.<0]) 
		p_acc = p_acc[:, active_idx]
		#print C.shape, C[0,-1,:].shape, p_acc.shape, p_acc.ravel().shape
		C[1,end, :] = vcat(p_acc...)
		return p_acc, C
    end

	cum_forward, Cl = leap_prob_recurse(q_chain[1:end-1], C[1:end-1, 1:end-1, :], active_idx, U)
	C[1:end-1,1:end-1, :] = Cl
	cum_reverse, Cl = leap_prob_recurse(q_chain[end:-1:2], C[end:-1:2, end:-1:2, :], active_idx, U)
	C[end:-1:2, end:-1:2, :] = Cl

	H0 = U(q_chain[1]) .- 0.5 * sum(q_chain[1].^2, dims=1)
    H1 = U(q_chain[end]) .- 0.5 * sum(q_chain[end].^2, dims=1)
    diff = H0 .- H1
    diff = diff[:, active_idx]
	start_state_ratio = exp.(diff) # This sometimes may be Inf. Combine with cum_reverse = 1
    # then line 107 tries to calculate 0*Inf = NaN causing downstream problems

    # if (x in start_state_ratio, y in cum_reverse -> (x == Inf) & (y == 1.0))
    #    throw(error("Multiplying 0 by Inf error."))
    # end

    for x in eachindex(start_state_ratio)
        if (start_state_ratio[x] == Inf) & (cum_reverse[x] == 1.0)
            start_state_ratio[x] = 0
        end
    end


	prob = minimum(vcat(1 .- cum_forward, start_state_ratio.*(1 .- cum_reverse)), dims=1)
	cumu = cum_forward + prob
	C[1,end,: ] = vcat(cumu...)

	return cumu, C
end

function leapfrog(current_q, current_p, epsilon, L, dU)
    p = current_p .- 0.5 * epsilon .* dU(current_q)
    proposed_q = current_q
    
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