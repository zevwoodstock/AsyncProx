using LinearAlgebra
using ProximalOperators
using Random

include("hinge_dot.jl")

# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 

global L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions.

# m = 1429 # The number of intervals (number of Gi)
global m = 100
# d = 10000
global d = 1000
# p = 100
global p = 300
global q_datacenters = 100

include("functions.jl")

global block_function = get_block_cyclic
global generate_gamma = generate_gamma_seq
global generate_mu = generate_mu_constant

global randomize_initial = false                    # this bool must be set to true if you want to randomize the intiial vector
global initialize_with_zi = false                   # this bool must be set to true if you want to initialize the initial vector with the defected images itself
global compute_epoch_bool = false                   # Necessary if record method is against epochs
global record_residual = false                      # record_residual = 1 for storing ||x_{n+1} - x_n||^2
global record_func = false
global record_dist = false
# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number, need to mark compute_epoch_bool = true as well then
# "2" is used to plot against the number of prox calls
# "3" is used to plot against the wall clock time
global record_method = "0"    

# -- X -- X -- X -- X -- X -- X -- X -- X -- Code Starts here -- X -- X -- X -- X -- X -- X -- X -- X 

global mu_k::Vector{Vector{Float64}} = []
global beta_k = Float64[] 
calculate_mu_beta()

global functions = []
for i in 1:functions_I
    append!(functions,[SqrNormL2(2)])
end
for k in 1:functions_K
    append!(functions,[HingeDot(beta_k, mu_k, k)])
end
