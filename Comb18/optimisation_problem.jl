
global functions = []

centers = [[0,1], [0,1], [1,1], [1,1]]

identity = [1 0;0 1]
null_mat = [0 0;0 0]
global L_matrix = [identity null_mat; null_mat identity; identity null_mat; null_mat identity]

identity_function(x) = x
zero_function(x) = zero(x)
minus_identity_function(x) = -x

#have to define tranposes of matrix and function


global L = [[1,0], [0,1], [1,0], [0,1]]

global L_function = [  [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]], 
                [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]]
                ]

for i in 1:functions_I+functions_K-2
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, dims,dims), 1, -centers[i])])
end

append!(functions, [Linear([1,0])])
append!(functions, [Linear([0,1])])
#append!(functions, [Linear([0,1])])
#append!(functions, [IndBallL2(0.001)])