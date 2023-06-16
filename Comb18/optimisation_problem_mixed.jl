using LinearAlgebra
using ProximalOperators
using Random

global functions_I = 3
global functions_K = 5

global dims_I = [2,2,3]
global dims_K = [2,2,2,2,3]
global block_function = get_block_cyclic        #To be set by user
global generate_gamma = generate_gamma_constant      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

global functions = []

I_3 = Matrix(I, 3, 3)
I_2 = Matrix(I, 2, 2)
Null_2 = zeros(2, 2)
Null_3 = zeros(3, 3)
centers = [[0,1], [0,1], [1,1], [1,1]]

identity_function(x) = x
null_2_function(x) = vec([0, 0])
null_3_function(x) = vec([0,0,0])

global L = [[I_2 , Null_2, zeros(2,3)], [Null_2,I_2,zeros(2,3)], [I_2, Null_2, zeros(2,3)], [ Null_2 ,I_2, zeros(2,3)], [zeros(3,2), zeros(3,2), I_3]]

global L_function_bool = false #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:

global L_function = [[identity_function, null_2_function, null_2_function], 
                        [null_2_function, identity_function, null_2_function],
                        [identity_function, null_2_function, null_2_function],
                        [null_2_function, identity_function, null_2_function],
                        [null_3_function, null_3_function, identity_function]]

global L_star_function = [[identity_function, null_2_function, null_3_function], 
                            [null_2_function, identity_function, null_3_function],
                            [identity_function, null_2_function, null_3_function],
                            [null_2_function, identity_function, null_3_function],
                            [null_2_function, null_2_function, identity_function]]

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
#modified to include L input as a function
