using LinearAlgebra
using ProximalOperators
using Random
println()
println("-------------------------------------------------------------------")
println("execution started")

include("variables.jl")
include("functions.jl")
include("init.jl")
include("parameters.jl")
include("hyperparameters.jl")
include("precomputations.jl")

include("loop.jl")

record()
get_accuracy()
println(check_feasibility())
print_params()
println()
