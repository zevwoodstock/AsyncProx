using LinearAlgebra
using ProximalOperators
using Random
using Wavelets
include("hinge_dot.jl")
global L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:

m = 3
p = 2
# functions_I = m , functions_K = p
# user has to input the values of m and p
global functions_I = m
global functions_K = p


global beta = fill(1.0,p)


global sigma = 0.01
global sigma_1 = 0.001
global sigma_2 = 0.001
global theta_main = 0.1
global randomize_initial = false                      # this bool must be set to true if you want to randomize the intiial vector
global initialize_with_zi = false                      # this bool must be set to true if you want to initialize the initial vector with the defected images itself
#record_residual = 1 for storing ||x_{n+1} - x_n||^2
global compute_epoch_bool = false
global record_residual = false
global record_func = false
global record_dist = false

# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number
# "2" is used to plot against the number of prox calla
# "3" is used to plot against the wall clock time
global record_method = "0"      
# global final_ans = [imageToVector(image_to_vector(path_1)), imageToVector(image_to_vector(path_2))]                                                 # to be declared for finding the various statistic values

# function constant_vector(N, c)
#     return fill(c, N)
# end
# global mu_array = constant_vector(N,0.1)

function phi(x)
    y = Wavelets.dwt(x, wavelet(WT.sym4))
    n = length(y)
    for i in 1:n
        y[i] = abs(y[i])
    end
    linear_op = Linear(mu_array)
    result = linear_op(y)
    return result
end

function generate_random_vector(N, sigma)
    rng = MersenneTwister(1234)  # Set a random number generator seed for reproducibility
    mu = 0.0  # Mean of the normal distribution (default: 0.0)
    sigma_squared = sigma^2  # Variance of the normal distribution
    
    random_vector = sqrt(sigma_squared) * randn(rng, N) .+ mu
    return random_vector
end

global norm_function = SqrNormL2(1)

identity_function(x)::Function = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end

# global L_function =  [[masking_l, null_func], [null_func, masking_r], [identity_function, define_D]]
global L_function = [[identity,identity,identity],[identity,identity,identity]]
# global L_function::Vector{Vector{Function}} = fill(fill(identity,m),p)
# global L_function = [[identity , null_func], [null_func, identity], [identity, define_D]]        #for no masking

# global L_star_function = [[masking_l, null_func], [null_func, masking_r], [identity, define_D_star]]
# global L_star_function::Vector{Vector{Function}} = fill(fill(identity,m),p)
global L_star_function = [[identity,identity,identity],[identity,identity,identity]]

global L = fill(fill(1.0,m),p)
global L_star_function = fill(fill(1.0,m),p)
global functions = []

global d = 10
d_one = fill(1.0, d)
global mu = fill(d_one,p)

include("functions.jl")

global dims_I = fill(d,m)
global dims_K = fill(d,p)
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_seq      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

for i in 1:functions_I
    append!(functions,[SqrNormL2(2)])
end

for i in 1:functions_K
    append!(functions,[HingeDot(beta, mu, k)])
end
