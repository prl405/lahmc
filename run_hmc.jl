using Random
using Plots

include("lahmc.jl")

function U_rough_well(X, scale1, scale2)
    cosX = cos.(X * 2 * pi / scale2)
    E = sum((X.^2) / (2 * scale1^2) .+ cosX)
    return E
end

function dU_rough_well(X, scale1, scale2)
    sinX = sin.(X * 2 * pi / scale2)
    dEdX = X ./ scale1^2 .- sinX * 2 * pi ./ scale2
    return dEdX
end

function U(X)
    return U_rough_well(X, theta[1], theta[2])
end

function dU(X)
    return dU_rough_well(X, theta[1], theta[2])
end

theta = [100, 4]
epsilon = 1
L = 10
n_samples = 500
 
DataSize = 2
batch_size = 3
X_init = randn(DataSize)*theta[1]

samples, acceptRate = hmc(U, dU, X_init, epsilon, L, n_samples)

plot(transpose(samples), xlabel="Samples", ylabel="Value")

display(plt_samples)
print("Acceptance Rate: ", acceptRate)
