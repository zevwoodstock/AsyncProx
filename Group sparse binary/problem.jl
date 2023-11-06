using LinearAlgebra
using ProximalOperators
using Random
# using Wavelets
include("hinge_dot.jl")
global L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:

function generate_G_x(m, d)
    g = [Int[] for _ in 1:m]
    x = [Int[0 for _ in 1:d] for _ in 1:m]
    start = 1
    for i in 1:m
        if i == 1
            start = 1
        else
            start += 7
        end
        
        temp = Int[]
        for j in 1:10
            push!(temp, start + j - 1)
            x[i][start + j - 1] = 1
        end
        g[i]= temp
    end
    return g, x
end

m = 7  # The number of intervals (number of Gi)
d = 52
p = 300
G, original_x = generate_G_x(m, d)
original_y = sum(original_x, dims=1)[1]
d_one = fill(1.0, d)
global mu1::Vector{Vector{Float64}} = []
for _ in 1:p
    random_vector = randn(Float64, d)
    rnorm = NormL2(1)(random_vector)
    random_vector = random_vector/rnorm
    # println(random_vector)
    push!(mu1, random_vector)
end

w_temp = []
for i in 1:p
    if i%4==0
        push!(w_temp, -1)
    else
        push!(w_temp, 1)
    end
end
w = shuffle(w_temp)


# functions_I = m , functions_K = p
# user has to input the values of m and p
global functions_I = m
global functions_K = p


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

global L = fill(fill(1.0,m),p)

global functions = []

global beta_k = Float64[]
for i in 1:p
    push!(beta_k, w[i]*sign(dot(mu1[i], original_y)))
end

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
    append!(functions,[HingeDot(beta_k, mu1, k)])
end
