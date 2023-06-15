using LinearAlgebra
using ProximalOperators
using Random
include("functions.jl")

global functions_I = 3
global functions_K = 5

global dims_I = [2,2, 3]
global dims_K = [2,2,2,2,3]
global block_function = get_block_cyclic

global functions = []

I_3 = Matrix(I, 3, 3)
I_2 = Matrix(I, 2, 2)
Null_2 = zeros(2, 2)
Null_3 = zeros(3, 3)

centers = [[0,1], [0,1], [1,1], [1,1]]

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

#solution should be x1 = [0, 1x] , x2 =[0.5, 0.134] and x3 = [0.73, 0.47, 0.2]
#modified
