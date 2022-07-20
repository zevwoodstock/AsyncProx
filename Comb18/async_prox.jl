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

print("Final ans: ")
println(res.x[iters])

#things left to do 
#plot
#feasibilty, func vals