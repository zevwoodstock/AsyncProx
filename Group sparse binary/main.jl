using LinearAlgebra
using ProximalOperators
using Random
# include("masking.jl")
include("problem.jl")

#intialising our problem
global D = 40
global iters = 10
global epsilon = 0.5

global constant_g = []   # this is being defined if for generate_gamma the strategy being taken is generate_gamma_constant
global constant_m = []   # this is being defined if for generate_mu the strategy being taken is generate_mu_constant
for i in 1:functions_I
    constant1 = epsilon + ((1/epsilon - epsilon) * rand())
    # constant1 = 1
    append!(constant_g, constant1)
end
for i in 1:functions_K
    # constant2 = epsilon + ((1/epsilon - epsilon) * rand())
    constant2 = 1
    append!(constant_m, constant2)
end
include("problem.jl")
include("variables.jl")
include("loop.jl")
println()
print("Final ans: ")
# println("reached here")
# x1 = res.x[iters][1]
# x2 = res.x[iters][2]
x_res = []
for i in 1:functions_I
    push!(x_res,res.x[iters][i])
end
println(size(x_res))
global y_pred::Vector{Float64} = fill(0.0, d)

for j in 1:length(x_res)
    # println(x_res[j])
    global y_pred += x_res[j]
end

global beta_res = Float64[] #The predicted beta (classifications)
global corr_pred::Float64 = 0 #the correct predictions count
for i in 1:p
    push!(beta_res, sign(dot(mu_k[i], y_pred)))
    # println(beta_k[i], " ", sign(dot(mu_k[i], y_pred)))
    if beta_res[i] == beta_k[i]
        global corr_pred+=1
    end
end
println("corr_pred = ", corr_pred, "\n Accuracy = ", (corr_pred / p))
println(check_feasibility())
println()
