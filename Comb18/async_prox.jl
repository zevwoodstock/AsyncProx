using LinearAlgebra
using ProximalOperators
using Random

include("functions.jl")
include("loop.jl")

#inputs
#intialising our problem
global I = 2
global K = 4
global dims = 2
global D = 40
global iterations = 5000
global gamma_history = []
global mu_history = []
centers = [[0,1], [0,1], [1,1], [1,1]]

identity = [1 0;0 1]
null_mat = [0 0;0 0]
L_matrix = [identity null_mat; null_mat identity; identity null_mat; null_mat identity]

identity_function(x) = x
zero_function(x) = zero(x)
minus_identity_function(x) = -x

#have to define tranposes of matrix and function
L = [[1,0], [0,1], [1,0], [0,1]]

L_function = [  [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]], 
                [[identity_function, identity_function], [zero_function, zero_function]], 
                [[zero_function, zero_function], [identity_function, identity_function]]
                ]
#make a dictionary of the functions, and then find the tranposes using the dictionary

functions = []
for i in 1:I+K-2
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, dims,dims), 1, -centers[i])])
end

append!(functions, [Linear([-1,0])])
append!(functions, [Linear([0,1])])
#append!(functions, [Linear([0,1])])
#append!(functions, [IndBallL2(0.001)])

zeros_I = []
zeros_K = []
global sum_vector_i = []
global sum_vector_k = []
for i in 1:I
    append!(zeros_I, [zeros(dims)])
    append!(sum_vector_i, [0.0])
end
for i in 1:K
    append!(zeros_K, [zeros(dims)])
    append!(sum_vector_k, [0.0])
end

#initialing an array to store the values of our main variables at every iteration
x_history = [zeros_I]
v_history = [zeros_K]

#Variables for asynchronous implementation
epsilon = 0.5
global minibatches = [[0,0], [0,0]]

struct variables                  #these are not really hyperparameters, but it makes the code look more organised~
    a::Vector{Vector{Float64}}
    a_star::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
    b_star::Vector{Vector{Float64}}
    t::Vector{Vector{Float64}}
    t_star::Vector{Vector{Float64}}
    l::Vector{Vector{Float64}}
    l_star::Vector{Vector{Float64}}
    sum_i::Vector{Float64}
    sum_k::Vector{Float64}
    birthdates::Vector{Vector{Int32}}
    tasks_num::Vector{Int32}
    task_number::Vector{Vector{Int32}}
    running_tasks::Vector{Vector{Any}}
    dims:: Int32
    D:: Int32
    epsilon:: Float64
    I:: Int32
    K:: Int32
    iters:: Int32
end

#gamma_history[j] and mu_history[j] are positive sequences in [epsilon, 1/epsilon] where epislon is (0,1)

#initialising all the global variables
vars = variables(zeros_I, zeros_I, zeros_K, zeros_K,
                        zeros_K, zeros_I, zeros_K, zeros_I,zeros(I), zeros(K), [[],[]],[0,0],[[],[]],
                        [[],[]], dims, D, epsilon, I, K, iterations)

global x_history, v_history = execute(vars, gamma_history, mu_history, L, v_history, x_history,functions)

print("Final ans: ")
println(x_history[vars.iters])

#things left to do 
#plot
#feasibilty, func vals