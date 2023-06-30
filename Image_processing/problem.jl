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

global path_1 = "/Users/kashishgoel/Downloads/imagel.jpg"
global path_2 = "/Users/kashishgoel/Downloads/imager.jpg"

global row1,col1 = get_row_column(path_1)
global row2,col2 = get_row_column(path_2)

global N = row1*col1

global sigma_1 = 2.0
global sigma_2 = 2.0
global theta = 1.0

img_arr_1 = imageToVector(image_to_vector(path_1))
img_arr_2 = imageToVector(image_to_vector(path_2))
global z1,L1 = masking(row1,col1,img_arr_1)
global z2,L2 = masking(row2,col2,img_arr_2)



function constant_vector(N, c)
    return fill(c, N)
end

global mu_array = constant_vector(N,1)



function phi(x)
    # Perform discrete wavelet transform
    y = dwt(x, wavelet(WT.sym4))
    
    # Define Linear operator with mu as coefficients
    linear_op = LinearOperator(mu_array)
    
    # Apply linear operator to y
    result = linear_op * y
    
    return result
end

function generate_random_vector(N, sigma)
    rng = MersenneTwister(1234)  # Set a random number generator seed for reproducibility
    mu = 0.0  # Mean of the normal distribution (default: 0.0)
    sigma_squared = sigma^2  # Variance of the normal distribution
    
    random_vector = sqrt(sigma_squared) * randn(rng, N) .+ mu
    return random_vector
end

w1 = generate_random_vector(N, 2.0)
w2 = generate_random_vector(N, 2.0)

# z1 = imageToVector(z1)
z1 += w1

# z2 = imageToVector(z2)
z2 += w2



function shift_image_left(image::Matrix{Float64}, x::Int64)
    width, height = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i - x - 1, width) + 1
            shifted_image[i, y] = image[j, y]
        end
    end

    return -shifted_image
end

function define_D(image::Vector{Float64})
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_left(image1,10)
    return imageToVector(image2)
end

function shift_image_right(image::Matrix{Float64}, x::Int)
    width, height = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i + x + 1, width) + 1
            shifted_image[i, y] = image[j, y]
        end
    end

    return -shifted_image
end

function define_D_star(image::Vector{Float64})
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_right(image1,10)
    return imageToVector(image2)
end

identity_function(x) = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end



global L_function =     [[masking, null_func], [null_func, masking], [identity_function, shift_image_left]]

global L_star_function = [[masking, null_func], [null_func, masking], [identity_function, shift_image_right]]
global functions = []

include("functions.jl")

global dims_I = [N,N]
global dims_K = [N,N,N]
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_constant      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

append!(functions,[phi,phi])
append!(functions,[Precompose(SqrNormL2(1/(sigma_1*sigma_1)),Matrix(LinearAlgebra.I, N,N),1,-z1)])
append!(functions,[Precompose(SqrNormL2(1/(sigma_2*sigma_2)),Matrix(LinearAlgebra.I, N,N),1,-z2)])
append!(functions,[SqrNormL2(theta)])


# append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i])])
#solution should be x1 = [0, 1x] , x2 =[0.5, 0.134] and x3 = [0.73, 0.47, 0.2]
#modified