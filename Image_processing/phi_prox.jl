using LinearAlgebra
using ProximalOperators
using Wavelets

export phi, Phi

function soft_threshold(x::Vector{Float64}, gamma::Float64)
    return sign.(x) .* max.(abs.(x) .- gamma, 0)
end

struct Phi{R}
    mu_array::Vector{R}
end

function phi(f::Phi, x)
    y = Wavelets.dwt(x, wavelet(WT.sym4))
    for i in 1:length(y)
        y[i] = abs(y[i])
    end
    linear_op = Linear(f.mu_array)
    result = linear_op(y)
    return result
end

function gradient!(y, f::Phi, x)
    # Not defined for the Phi function, since it's not needed for prox calculation
    error("Gradient calculation is not defined for Phi")
end

function prox!(y, f::Phi, x, gamma)
    dwt = Wavelets.dwt(x, wavelet(WT.sym4))
    st = soft_threshold(dwt, gamma * f.mu_array[1])
    idwt = Wavelets.idwt(st, wavelet(WT.sym4))
    y .= idwt
    return phi(f, idwt)
end

function prox(f::Phi, x, gamma)
    y = similar(x)
    prox!(y, f, x, gamma)
    return y, phi(f, y)
end
