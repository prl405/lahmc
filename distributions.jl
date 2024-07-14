using LinearAlgebra

mutable struct Rough_Well
    dims::Int
    scale1::Number
    scale2::Number
end

function init_rough_well(rw::Rough_Well)
    return randn(rw.dims)*rw.scale1
end

function U_rough_well(X, rw::Rough_Well)
    cosX = cos.(X * 2 * pi / rw.scale2)
    E = sum((X.^2) / (2 * rw.scale1^2) + cosX)
    return E
end

function dU_rough_well(X, rw::Rough_Well)
    sinX = sin.(X * 2 * pi / rw.scale2)
    dEdX = (X ./ rw.scale1^2) .- (sinX * 2 * pi ./ rw.scale2)
    return dEdX
end

mutable struct Gaussian
    dims::Int
    J::Matrix
    conditioning::Array
end

function Gaussian(dims=2, log_conditioning=0.6)
    conditioning = 10 .^ LinRange(-log_conditioning, 0, dims)
    J = Diagonal(conditioning)
    Gaussian(dims, J, conditioning)
end

function init_gaussian(gaussian::Gaussian)
    # return (inv(sqrt.(gaussian.conditioning)))*randn(gaussian.dims)
    return (1.0 ./ sqrt.(gaussian.conditioning)) .* randn(gaussian.dims)
end

function U_gaussian(X, gaussian::Gaussian)
    J = gaussian.J
    return sum(X.*(J*X))/2
end

function dU_gaussian(X, gaussian::Gaussian)
    J = gaussian.J
    return (J*(X/2)) + (transpose(J)*(X/2))
end

# conditioning = 10**np.linspace(-log_conditioning, 0, ndims)
# 		self.J = np.diag(conditioning)
# 		self.Xinit = (1./np.sqrt(conditioning).reshape((-1,1))) * np.random.randn(ndims,nbatch)
# 		self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)
# 	def E(self, X):
# 		return np.sum(X*np.dot(self.J,X), axis=0).reshape((1,-1))/2.
# 	def dEdX(self, X):
# 		return np.dot(self.J,X)/2. + np.dot(self.J.T,X)/2.