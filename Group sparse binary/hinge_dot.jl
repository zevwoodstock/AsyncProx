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


# println("Result of HingeDot function call on x: ", result)

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
