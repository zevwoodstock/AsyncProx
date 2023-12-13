using LinearAlgebra
using ProximalOperators
using Random

include("parameters.jl")
include("variables.jl")
include("loop.jl")

get_accuracy()
println(check_feasibility())
println()
