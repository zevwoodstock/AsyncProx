using LinearAlgebra
using ProximalOperators
using Random
using Images
using ImageView
using Wavelets

global L_function_bool = true  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions below:


# include("functions.jl")
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

global functions_I = 2
global functions_K = 3

global path_1 = "/Users/HP/Documents/ZIB Internship/Image Recovery/L.jpeg"
global path_2 = "/Users/HP/Documents/ZIB Internship/Image Recovery/R.jpeg"

global row1,col1 = get_row_column(path_1)
global row2,col2 = get_row_column(path_2)

global N = row1*col1

global sigma_1 = 0.01
global sigma_2 = 0.01
global theta = 1.0

img_arr_1 = imageToVector(image_to_vector(path_1))
img_arr_2 = imageToVector(image_to_vector(path_2))
global z1 = masking(img_arr_1)
global z2 = masking(img_arr_2)



function constant_vector(N, c)
    return fill(c, N)
end

global mu_array = constant_vector(N,1)



function phi(x)
    # Perform discrete wavelet transform
    y = dwt(x, wavelet(WT.sym4))
    
    # Define Linear operator with mu as coefficients ie Linear(c[]) = < . | c[]>
    linear_op = Linear(mu_array)
    
    # Apply linear operator to y
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

w1 = generate_random_vector(N, sigma_1)
w2 = generate_random_vector(N, sigma_2)

# z1 = imageToVector(z1)
# z1 += w1
for i in 1:N
    # println(typeof(z1))
    z1[i] = z1[i] + w1[i]
end

# z2 = imageToVector(z2)
# z2 += w2
for i in 1:N
    z2[i] = z2[i] + w2[i]
end

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
    image2 = shift_image_left(image1,5)
    return imageToVector(image2)
end

function define_D(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_left(image1,5)
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
    image2 = shift_image_right(image1,5)
    return imageToVector(image2)
end

function define_D_star(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_right(image1,5)
    return imageToVector(image2)
end

identity_function(x) = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end

global L_function =     [[masking, null_func], [null_func, masking], [identity_function, define_D]]

global L_star_function = [[masking, null_func], [null_func, masking], [identity_function, define_D_star]]
global functions = []

include("functions.jl")

global dims_I = [N,N]
global dims_K = [N,N,N]
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_constant      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

append!(functions,[phi])
append!(functions,[phi])
# append!(functions,[SqrNormL2(theta)])
# append!(functions,[SqrNormL2(theta)])
append!(functions,[Precompose(SqrNormL2(1/(sigma_1*sigma_1)),1,1,-z1)])
append!(functions,[Precompose(SqrNormL2(1/(sigma_2*sigma_2)),1,1,-z2)])
append!(functions,[SqrNormL2(theta)])
