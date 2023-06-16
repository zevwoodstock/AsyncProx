using LinearAlgebra
using ProximalOperators
using Random
include("functions.jl")

global functions_I = 1
global functions_K = 3

global dims_I = [3]
global dims_K = [3,3,3]
global block_function = get_block_cyclic
global generate_gamma = generate_gamma_seq
global generate_mu = generate_mu_seq

global functions = []

I_3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
Null_2 = zeros(2, 2)
Null_3 = zeros(3, 3)

centers = [[0,1], [0,1], [1,1], [1,1]]

global L = [[I_3], [I_3], [I_3]]

append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 3,3), 1, -[1,1,1])]) #sphere for L3

append!(functions, [Linear([0,1,1])])
append!(functions, [Linear([1,1,0])])
append!(functions, [Linear([1,0,1])])
