using LinearAlgebra
using ProximalOperators
using Random
include("optimisation_problem2.jl")

#intialising our problem
global D = 40
global iters = 5000
# global iters = 10000
global epsilon = 0.5

global constant_g = []   # this is being defined if for generate_gamma the strategy being taken is generate_gamma_constant
global constant_m = []   # this is being defined if for generate_mu the strategy being taken is generate_mu_constant

for i in 1:functions_I
    constant1 = epsilon + ((1/epsilon) * rand())
    # constant1 = 2
    append!(constant_g, constant1)
end
for i in 1:functions_K
    constant2 = epsilon + ((1/epsilon) * rand())
    # constant2 = 2
    append!(constant_m, constant2)
end

include("optimisation_problem2.jl")
include("variables.jl")
include("loop.jl")

println()
print("Final ans: ")
println(res.x[iters])

println(check_feasibility())
println()
#things left to do 
#plot
