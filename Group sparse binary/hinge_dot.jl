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
    datacenter_index = mapping_ping[f.k]
    sleep_time = ping_array[datacenter_index]
    n_error = randn()*0.005 #implies std_dev of normal error is 0.005 
    sleep(abs(n_error) + sleep_time)
    mu2 = SqrNormL2(2)(f.mu[f.k])
    Lx = Linear(f.mu[f.k])(x)
    p,v = prox(HingeLoss([f.beta[f.k]], 10), [Lx], gamma)
    p = p[1]
    p = p - Lx
    p = p*f.mu[f.k] #L*(v \in R^1) = v mu_k
    y .= (x + (1/mu2)*p)
    return f(y)
end

#prox(f, x, gamma)
# println("Prox of a pre-exisitng fn = ", prox(CubeNormL2(1), x, 1.0))
# println("Prox of hinge_dot_function = ", prox(hinge_dot_function, x, 1.0))
#We want to be able to call both preexisitng functions in ProximalOperators.jl and ones that we define ourselves.
