using LinearAlgebra
using ProximalOperators
using Random

#intialising our problem
global functions_I = 2
global functions_K = 4
global dims = 2
global D = 40
global iters = 5000
global epsilon = 0.5 
global block_function = get_block_cyclic #user can choose which block selection strategy they want here
#replace In with block_function(n, m, M) ; default m = 20, M = 5

include("optimisation_problem.jl")
include("variables.jl")
include("loop.jl")


println()
print("Final ans: ")
println(res.x[iters])

println(check_feasibility())
println()
#things left to do 
#plot