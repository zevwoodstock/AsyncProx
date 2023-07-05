using LinearAlgebra
using ProximalOperators
using Random


global functions_I = 3
global functions_K = 5

global dims_I = [2,2,3]
global dims_K = [2,2,2,2,3]

function get_block_cyclic(n::Int64, m::Int64 = 20, M::Int64 = 5) 
    block_size = div(m, M)
    start = (((n%M) - 1) * block_size) % m + 1
    fin = start + block_size - 1

    if n % M == 0
        start = ((M - 1)* block_size)%m + 1
        fin = max(fin, m)
    end

    arr = Int64[]
    for i in start:fin
        push!(arr, i)
        # println(arr)
    end
    return arr
end

function generate_gamma_constant(i,j)
    return constant_g[i]
end

function generate_mu_constant(k,j)
    return constant_m[k]
end

global block_function = get_block_cyclic        #To be set by user
global generate_gamma = generate_gamma_constant      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

#record_residual = 1 for storing ||x_{n+1} - x_n||^2
global record_residual = true
global record_func = false
global record_dist = true

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

global final_ans = [[0.0, 1.0], [0.5, 0.134], [0.73, 0.47, 0.2]]
global f_ans = [0, 0, 0]
global g_ans = [0,0, 0, 0.134, 2.27]

#solution should be x1 = [0, 1x] , x2 =[0.5, 0.134] and x3 = [0.73, 0.47, 0.2]
#modified
