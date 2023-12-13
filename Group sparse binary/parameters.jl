using LinearAlgebra
using ProximalOperators
using Random

include("hinge_dot.jl")

# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 

global D = 40
global iters = 10

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

global generate_gamma = generate_gamma_nonlinear_decrease
global generate_mu = generate_mu_linear_decrease

# For generate_gamma_constant
global constant_gamma = 1

# For generate_mu_constant
global constant_mu = 0.35

# For generate_gamma and mu random
global epsilon = 0.01

# For generate_gamma_linear_decrease
global gamma_start = []
global gamma_end = []
global gamma_step = []
for i in 1:functions_I
    append!(gamma_start, 10)
    append!(gamma_end, 0.5)
    append!(gamma_step, 1)
end

# For generate_mu_linear_decrease

# start - step*j
global mu_start = []
global mu_end = []
global mu_step = []
for k in 1:functions_K
    append!(mu_start, 10)
    append!(mu_end, 0.25)
    append!(mu_step, 1)
end

# For generate_gamma_nonlinear_decrease

# a + b/j
global gamma_a = []
global gamma_b = []
for i in 1:functions_I
    append!(gamma_a, 0.5)
    append!(gamma_b, 5)
end

# For generate_mu_nonlinear_decrease
global mu_a = []
global mu_b = []
for i in 1:functions_K
    append!(mu_a, 0)
    append!(mu_b, 5)
end


global randomize_initial = false                    # this bool must be set to true if you want to randomize the initial vector
global initialize_with_zi = false                   # this bool must be set to true if you want to initialize the initial vector with the defected images

global compute_epoch_bool = false                   # Necessary if record method is "1" - epoch numbers

global record_residual = false                      # record_residual = 1 for storing ||x_{n+1} - x_n||^2
global record_func = true                           # record_func = 1 for storing f(x_n)
global record_dist = false                          # record_dist = 1 for storing ||x_* - x_n||^2

# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number
# "2" is used to plot against the number of prox calls
# "3" is used to plot against the wall clock time
global record_method = "0"  

global alpha = 0.5
global beta = 0.5

# -- X -- X -- X -- X -- X -- X -- X -- X -- Some Precomputations -- X -- X -- X -- X -- X -- X -- X -- X 

if record_method == "1"
    compute_epoch_bool = true
end

global constant_g = []   # this is defined if for generate_gamma the strategy taken is generate_gamma_constant
global constant_m = []   # this is defined if for generate_mu the strategy taken is generate_mu_constant
global mu_k::Vector{Vector{Float64}} = []
global beta_k = Float64[] 
calculate_mu_beta()

global functions = []
for i in 1:functions_I
    append!(functions,[SqrNormL2(2)])
    append!(constant_g, constant_gamma)
end
for k in 1:functions_K
    append!(functions,[HingeDot(beta_k, mu_k, k)])
    append!(constant_m, constant_mu)
end