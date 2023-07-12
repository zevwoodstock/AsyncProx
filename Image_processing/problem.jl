using LinearAlgebra
using ProximalOperators
using Random
using Images
using ImageView
using Wavelets

global L_function_bool = true  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:

include("masking.jl") 

function imageToVector(image::Array{T, 2}) where {T}
    rows, columns = size(image)
    vector = vec(image)
    return vector
end

function vectorToImage(rows::Int64, columns::Int64, vector::Vector{T}) where {T}
    image = reshape(vector, rows, columns)
    return image
end

# global path_1 = "/Users/kashishgoel/Downloads/imagel.jpg"
global path_1 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/image_left.jpeg"
global path_2 = "/Users/kashishgoel/Downloads/imager.jpg" 

img_arr_1 = imageToVector(image_to_vector(path_1))
img_arr_2 = imageToVector(image_to_vector(path_2))

global functions_I = 2
global functions_K = 3

global row1,col1 = get_row_column(path_1)
global row2,col2 = get_row_column(path_2)

global N = row1*col1

global sigma_1 = 0.1
global sigma_2 = 0.1
global theta = 1

global randomize_initial = false                      # this bool must be set to true if you want to randomize the intiial vector
global initialize_with_zi = true                      # this bool must be set to true if you want to initialize the initial vector with the defected images itself
#record_residual = 1 for storing ||x_{n+1} - x_n||^2
global compute_epoch_bool = false
global record_residual = true
global record_func = true
global record_dist = true

# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number
# "2" is used to plot against the number of prox calla
# "3" is used to plot against the wall clock time
global record_method = "0"      
global final_ans = [imageToVector(image_to_vector(path_1)), imageToVector(image_to_vector(path_2))]                                                 # to be declared for finding the various statistic values

function constant_vector(N, c)
    return fill(c, N)
end

global mu_array = constant_vector(N,1.0)

function phi(x)
    y = Wavelets.dwt(x, wavelet(WT.db8))
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

w1 = generate_random_vector(N, sigma_1)
w2 = generate_random_vector(N, sigma_2)

# global z1 = copy(img_arr_1)
# global z2 = copy(img_arr_2)

global z1 = blur(img_arr_1)
global z2 = blur(img_arr_2)

w_1 = generate_random_vector(N,1)
norm_w1 = norm_function(w_1)*2
w_1 = w_1/norm_w1
w_2 = generate_random_vector(N,1)
norm_w2 = norm_function(w_2)*2
w_2 = w_2/norm_w2

for i in 1:N
    z1[i] = z1[i] + w1[i]
end

for i in 1:N
    z2[i] = z2[i] + w2[i]
end

# z1 = masking_left(z1)
# z2 = masking_right(z2)

function shift_image_left(image::Matrix{Float64}, x::Int64)
    width, height = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i - x - 1, width) + 1
            shifted_image[y, i] = image[y, j]
        end
    end
    return -shifted_image
end

function define_D(image::Vector{Float64})
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_left(image1,7)
    return imageToVector(image2)
end

function define_D(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_left(image1,7)
    return imageToVector(image2)
end

function shift_image_right(image::Matrix{Float64}, x::Int)
    width, height = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i + x + 1, width) + 1
            shifted_image[y, i] = image[y, j]
        end
    end

    return -shifted_image
end

function define_D_star(image::Vector{Float64})
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_right(image1,7)
    return imageToVector(image2)
end

function define_D_star(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_right(image1,7)
    return imageToVector(image2)
end

identity_function(x) = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end

# global L_function =     [[masking_l, null_func], [null_func, masking_r], [identity_function, define_D]]
global L_function =     [[blur, null_func], [null_func, blur], [identity_function, define_D]]
# global L_function = [[identity_function , null_func], [null_func, identity_function], [identity_function, define_D]]        #for no masking

# global L_star_function = [[masking_l, null_func], [null_func, masking_r], [identity_function, define_D_star]]
global L_star_function = [[blur, null_func], [null_func, blur], [identity_function, define_D_star]]
# global L_star_function = [[identity_function, null_func], [null_func, identity_function], [identity_function, define_D_star]]    # for no masking
global functions = []

degraded_image_left = matrix_to_image(vectorToImage(row1,col1,z1))
degraded_path_l = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/degraded_image_left.jpeg"   # Replace with the desired path and filename for the image
save(degraded_path_l, degraded_image_left)
degraded_image_right = matrix_to_image(vectorToImage(row1,col1,z2))
degraded_path_r = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/degraded_image_right.jpeg"  # Replace with the desired path and filename for the image
save(degraded_path_r, degraded_image_right)

include("functions.jl")

global dims_I = [N,N]
global dims_K = [N,N,N]
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_constant      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

append!(functions,[phi])
append!(functions,[phi])
append!(functions,[Precompose(SqrNormL2(1/(sigma_1*sigma_1)),1,1,-z1)])
append!(functions,[Precompose(SqrNormL2(1/(sigma_2*sigma_2)),1,1,-z2)])
append!(functions,[SqrNormL2(theta)])
