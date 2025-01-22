
@everywhere using LinearAlgebra
@everywhere using ProximalOperators
@everywhere using Random
using TickTock
println()
println("-------------------------------------------------------------------")
println("execution started")

@everywhere include("variables.jl")
@everywhere include("functions.jl")
@everywhere include("init.jl")
@everywhere include("parameters.jl")
@everywhere include("hyperparameters.jl")
@everywhere include("precomputations.jl")

tick()
include("loop.jl")
tock()

record()
get_accuracy()
println(check_feasibility())
print_params()
println()
