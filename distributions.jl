
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