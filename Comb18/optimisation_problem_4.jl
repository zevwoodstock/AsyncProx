# This is the optimisation problem designed which has no soultion for 2 dimensional function
# The function comprises of indicator circles and spheres
# For the 2 dimensional problem, the intersection of the domains is an empty set as there is one circle centred at (1,0) and another at (3,1) each of radii 1
# As a result of empty domain, there exists no solution for the problem of 2 dimensional functions 

# Final ans: [[1.1073049800296961, 1.0000056169877425], [1.1565494592838546, 0.9930389064428765], [0.7327387580875756, 0.465477516175151, 0.19821627426272653]]
# false

# On running the below problem, we get a "false" indicating that the minimum solution obtained doesn't exist in the domain given to us and hence is not the right solution

using LinearAlgebra
using ProximalOperators
using Random
include("functions.jl")

global functions_I = 3
global functions_K = 5

global dims_I = [2,2,3]
global dims_K = [2,2,2,2,3]
global block_function = get_block_cyclic
global generate_gamma = generate_gamma_constant
global generate_mu = generate_mu_constant

global functions = []

I_3 = Matrix(I, 3, 3)
I_2 = Matrix(I, 2, 2)
Null_2 = zeros(2, 2)
Null_3 = zeros(3, 3)

centers = [[0,1], [0,1], [3,1], [3,1]]

global L = [[I_2 , Null_2, zeros(2,3)], [Null_2,I_2,zeros(2,3)], [I_2, Null_2, zeros(2,3)], [ Null_2 ,I_2, zeros(2,3)], [zeros(3,2), zeros(3,2), I_3]]

for i in 1:functions_I-1
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i])]) #circle 1 for x1 x2
end 

append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 3,3), 1, -[1,1,1])]) #sphere for L3

for i in 1:functions_K - 3
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i+2])]) #circle 2 for x1 x2
end 

append!(functions, [Linear([1,0])])
append!(functions, [Linear([0,1])])
append!(functions, [Linear([1,2,3])])
