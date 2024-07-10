using LinearAlgebra

mutable struct Rough_Well
    scale1::Number
    scale2::Number
end

function U_rough_well(X, scale1=100, scale2=4)
    cosX = cos.(X * 2 * pi / scale2)
    E = sum((X.^2) / (2 * scale1^2) + cosX)
    return E
end

function dU_rough_well(X, scale1=100, scale2=4)
    sinX = sin.(X * 2 * pi / scale2)
    dEdX = (X ./ scale1^2) .- (sinX * 2 * pi ./ scale2)
    return dEdX
end

mutable struct Gaussian
    dims::Int
    J::Matrix
end

function Gaussian(dims=2, log_conditioning=0.6)
    conditioning = 10*LinRange(-log_conditioning, 0, dim)
    J = Diagonal(conditioning)
    Gaussian(dims, J)
end

function U_gaussian(X, gaussian::Gaussian)
    J = gaussian.J
    return sum(X*(J.*X), dims=1)/2
end

function dU_gaussian(X, gaussian::Gaussian)
    J = gaussian.J
    return (J.*X/2) .+ (J'.*X/2)
end

# conditioning = 10**np.linspace(-log_conditioning, 0, ndims)
# 		self.J = np.diag(conditioning)
# 		self.Xinit = (1./np.sqrt(conditioning).reshape((-1,1))) * np.random.randn(ndims,nbatch)
# 		self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)
# 	def E(self, X):
# 		return np.sum(X*np.dot(self.J,X), axis=0).reshape((1,-1))/2.
# 	def dEdX(self, X):
# 		return np.dot(self.J,X)/2. + np.dot(self.J.T,X)/2.