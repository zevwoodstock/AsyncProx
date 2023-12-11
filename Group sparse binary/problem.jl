using LinearAlgebra
using ProximalOperators
using Random
include("hinge_dot.jl")
global L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:
include("functions.jl")

# m = 1429 # The number of intervals (number of Gi)
m = 100
# d = 10000
d = 1000
# p = 100
p = 300
global q_datacenters = 100
global ping_array = generate_random_array(q_datacenters, 0.05)
global mapping_ping = rand(1:100, p)
global mu_k::Vector{Vector{Float64}} = []
global beta_k = Float64[]

# user has to input the values of m and p
global functions_I = m
global functions_K = p
global randomize_initial = false                      # this bool must be set to true if you want to randomize the intiial vector
global initialize_with_zi = false                      # this bool must be set to true if you want to initialize the initial vector with the defected images itself
#record_residual = 1 for storing ||x_{n+1} - x_n||^2
global compute_epoch_bool = false #Nec
global record_residual = false
global record_func = false
global record_dist = false
# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number, need to mark compute_epoch_bool = true as well then
# "2" is used to plot against the number of prox calls
# "3" is used to plot against the wall clock time
global record_method = "0"      
# global final_ans = [imageToVector(image_to_vector(path_1)), imageToVector(image_to_vector(path_2))]                                                 # to be declared for finding the various statistic values
# function constant_vector(N, c)
#     return fill(c, N)
# end
# global mu_array = constant_vector(N,0.1)

global L = fill(fill(1.0,m),p)
global dims_I = fill(d,m)
global dims_K = fill(d,p)
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_seq          #To be set by user
global generate_mu = generate_mu_constant           #To be set by user
calculate_beta()

global functions = []
for i in 1:functions_I
    append!(functions,[SqrNormL2(2)])
end
for k in 1:functions_K
    append!(functions,[HingeDot(beta_k, mu_k, k)])
end
