function hmc(U, dU, init_q, epsilon, L, n_samples)
    samples = zeros(length(init_q), n_samples)
    current_q = init_q
    accept_count = 0

    for i in 1:n_samples
        proposed_q, accept = leapfrog(current_q, epsilon, L, U, dU)

        if accept
            current_q = proposed_q
            accept_count += 1
        end
        samples[:, i] = current_q
    end

    acceptRate = accept_count / n_samples
    return samples, acceptRate
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