using LinearAlgebra
using ProximalOperators
using Random


global functions_I = 2
global functions_K = 5

global dims_I = [2,2]
global dims_K = [2,2,2,2,2]

global functions = []

centers = [[0,1], [0,1], [1,1], [1,1]]

identity = [1 0;0 1]
null_mat = [0 0;0 0]
global L_matrix = [identity null_mat; null_mat identity; identity null_mat; null_mat identity]

identity_function(x) = x
zero_function(x) = zero(x)
minus_identity_function(x) = -x

function Ind_D(x1, x2)
    if x1 == x2
        return 0
    else
        return Inf
    end
end
#have to define tranposes of matrix and function


global L = [[1,0], [0,1], [1,0], [0,1], [1,-1]]

global L_function = [  [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]], 
                [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]]
                ]

for i in 1:functions_I+functions_K-3
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i])])
end 



append!(functions, [Linear([1,0])])
append!(functions, [Linear([0,1])])
append!(functions, [IndBallL2(0.000001)])
#append!(functions, [Linear([0,1])])
#append!(functions, [IndBallL2(0.001)])

##definition of Precompose
# Return the function
#     \[g(x) = f(Lx + b)\]
#     where $f$ is a convex function and $L$ is a linear mapping: this must satisfy $LL^* = μI$ for $μ > 0$. Furthermore, either $f$ is separable or parameter μ is a scalar, for the prox of $g$ to be computable.
#     Parameter L defines $L$ through the mul! method. Therefore L can be an AbstractMatrix for example, but not necessarily.
