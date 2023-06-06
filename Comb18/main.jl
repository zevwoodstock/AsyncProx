using LinearAlgebra
using ProximalOperators
using Random

#intialising our problem
global functions_I = 2    # m -  size of set I 
global functions_K = 4    # p -  size of set K
# global dims = 2
# global dims_I = [dims1,dims2,...,dimsI]
# global dims_K = [dims1,dims2,...,dimsK]
global dims_I = [2,3]
global dims_K = [1,2,3,2]
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
