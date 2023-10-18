using ProximalOperators

# Define the custom HingeDot function type
struct HingeDot
    beta::Vector{Float64}
    mu::Vector{Vector{Float64}}
    k::Int
    
    function HingeDot(beta::Vector{Float64}, mu::Vector{Vector{Float64}}, k::Int)
        if k < 1 || k > length(beta) || k > length(mu)
            error("Invalid index k")
        else
            new(beta, mu, k)
        end
    end
end

is_convex(f::Type{<:HingeDot}) = true
is_smooth(f::Type{<:HingeDot}) = true

# Implement the call method for the HingeDot function type
function (f::HingeDot)(x)
    linear_result = Linear(f.mu[f.k])(x)
    hinge_loss_result = HingeLoss([f.beta[f.k]], 10)([linear_result])
    return hinge_loss_result
end

# Example usage
beta = [1.5, 2.0, 1.8]
mu = [[0.8, 1.2, 1.5] ,[0.8, 1.2, 1.5]] #d dimensioned p vectors
k = 2

hinge_dot_function = HingeDot(beta, mu, k)

# Call the custom HingeDot function on a vector x
x = [0.07, .12, .034]
result = hinge_dot_function(x)

println("Result of HingeDot function call on x: ", result)

function ProximalOperators.prox!(y, f::HingeDot, x, gamma)
    mu = SqrNormL2(2)(f.mu[f.k])
    Lx = Linear(f.mu[f.k])(x)
    p,v = prox(HingeLoss([f.beta[f.k]], 10), [Lx], gamma)
    p = p[1]
    p = p - Lx
    p = p*f.mu[f.k] #L*(v \in R^1) = v mu_k
    y .= (x + (1/mu)*p)
    return f(y)
end

#prox(f, x, gamma)
println("Prox of a pre-exisitng fn = ", prox(CubeNormL2(1), x, 1.0))
println("Prox of hinge_dot_function = ", prox(hinge_dot_function, x, 1.0))
#We want to be able to call both preexisitng functions in ProximalOperators.jl and ones that we define ourselves.