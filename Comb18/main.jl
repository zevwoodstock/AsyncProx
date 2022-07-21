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