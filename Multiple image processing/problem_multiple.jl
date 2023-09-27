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
# global path_1 = "/Users/HP/Documents/ZIB Internship/Image Recovery/zib_dotted_l.jpeg"
# global path_2 = "/Users/HP/Documents/ZIB Internship/Image Recovery/zib_dotted_r.jpeg"

global path_1 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/1.jpeg"
global path_2 = "//Users/kashishgoel/Desktop/Intern_2023/Image_processing/2.jpeg"
global path_3 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/3.jpeg"
global path_4 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/4.jpeg"
img_arr_1 = imageToVector(image_to_vector(path_1))
img_arr_2 = imageToVector(image_to_vector(path_2))
img_arr_3 = imageToVector(image_to_vector(path_3))
img_arr_4 = imageToVector(image_to_vector(path_4))
img_arr = [img_arr_1,img_arr_2,img_arr_3,img_arr_4]

# println(sum(img_arr_1)/length(img_arr_1))

# for 2 images
# path_1 = "/Users/kashishgoel/Downloads/left.jpeg"
# path_2 = "/Users/kashishgoel/Downloads/right.jpeg"
# img_arr_1 = imageToVector(image_to_vector(path_1))
# img_arr_2 = imageToVector(image_to_vector(path_2))
# img_arr = [img_arr_1,img_arr_2]

# SPECIFY NUMBER OF IMAGES = m HERE:
global num_images = 4

# functions_I = m , functions_K = 2m - 1
global functions_I = num_images
global functions_K = 2*num_images - 1

global row1,col1 = get_row_column(path_1)
global row2,col2 = get_row_column(path_2)
global row3, col3 = get_row_column(path_3)
global row4, col4 = get_row_column(path_4)

# save(save_path_l, matrix_to_image(vectorToImage(row1,col1,img_arr_1)))
# save(save_path_r, matrix_to_image(vectorToImage(row2,col2,img_arr_2)))

global N = row1*col1

global sigma = 0.01
global sigma_1 = 0.001
global sigma_2 = 0.001
global theta_main = 0.1
global randomize_initial = false                      # this bool must be set to true if you want to randomize the intiial vector
global initialize_with_zi = true                      # this bool must be set to true if you want to initialize the initial vector with the defected images itself
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
global final_ans = [imageToVector(image_to_vector(path_1)), imageToVector(image_to_vector(path_2))]                                                 # to be declared for finding the various statistic values

function constant_vector(N, c)
    return fill(c, N)
end
global mu_array = constant_vector(N,0.1)

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

w = []
for i in 1:functions_I
    append!(w,generate_random_vector(N,sigma))
end

# w1 = generate_random_vector(N, sigma_1)
# w2 = generate_random_vector(N, sigma_2)

# global z1 = copy(img_arr_1)
# global z2 = copy(img_arr_2)

z = []
for i in 1:functions_I
    copy_i = copy(img_arr[i])
    # println(copy_i)
    push!(z,copy_i)
    # println(copy(img_arr[i])[1])
    # println(z[i])
end

# println(z[1])

# global z1 = blur(img_arr_1)
# global z2 = blur(img_arr_2)

# w_1 = generate_random_vector(N,1)
# norm_w1 = norm_function(w_1)*2
# w_1 = w_1/norm_w1
# w_2 = generate_random_vector(N,1)
# norm_w2 = norm_function(w_2)*2
# w_2 = w_2/norm_w2

w = []
for i in 1:functions_I
    push!(w,generate_random_vector(N,1))
    norm_wi = norm_function(w[i])*2
    w[i] = w[i]/norm_wi
    # s = sum(w[i])
    # println(s/length(w))
end


# for i in 1:N
#     z1[i] = z1[i] + w1[i]
# end

# for i in 1:N
#     z2[i] = z2[i] + w2[i]
# end


z_blur = []
for i in 1:functions_I
    push!(z_blur,blur(z[i]))
end

for i in 1:functions_I
    for j in 1:N
        z_blur[i] = z_blur[i] + w[i]
    end
end

# z1 = blur(z1)
# z2 = blur(z2)
# z1 = masking_left(z1)
# z2 = masking_right(z2)

function shift_image_left(image::Matrix{Float64}, x::Int64)
    height, width = size(image)
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
    image2 = shift_image_left(image1,20)
    return imageToVector(image2)
end

function define_D(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_left(image1,20)
    return imageToVector(image2)
end

function shift_image_right(image::Matrix{Float64}, x::Int)
    height, width = size(image)
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
    image2 = shift_image_right(image1,20)
    return imageToVector(image2)
end

function define_D_star(image::Matrix{Float64})
    image = vec(image)
    image1 = vectorToImage(row1,col1,image)
    image2 = shift_image_right(image1,20)
    return imageToVector(image2)
end

identity_function(x) = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end

function generate_L(m::Int, degradation::Function)
    L_func = Vector{Vector{Function}}()
    for i = 1:m
        # println("hello ",i)
        row = Vector{Function}()
        for j = 1:m
            # println("in ",j)
            if (j == i)
                push!(row, degradation)
            else
                push!(row, null_func)
            end
        end
        push!(L_func, row)
        # println(row)
        # println(L_func)
        # if i == 1
        #     println(typeof(row))
        # end
    end
    for i = 1:m-1
        # println("hi")
        row = Vector{Function}()
        for j = 1:m
            if j == i 
                push!(row, identity_function)
            elseif j == i+1
                push!(row, define_D)
            else 
                push!(row, null_func)
            end
        end
        # println(row)
        # println(L_func)
        push!(L_func, row)
    end
    # println(L_func)
    return L_func
end

function generate_L_star(m::Int, degradation::Function)
    L_func = []
    for i = 1:m
        row = []
        for j = 1:m
            if (j == i)
                push!(row, degradation)
            else
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    for i = 1:m-1
        row = []
        for j = 1:m
            if j == i 
                push!(row, identity_function)
            elseif j == i+1
                push!(row, define_D_star)
            else 
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    return L_func
end

# global L_function =     [[masking_l, null_func], [null_func, masking_r], [identity_function, define_D]]
global L_function =     generate_L(num_images, blur)
# println(L_function)
# global L_function = [[identity_function , null_func], [null_func, identity_function], [identity_function, define_D]]        #for no masking

# println(L_function)
# println(typeof(L_function))
# global L_star_function = [[masking_l, null_func], [null_func, masking_r], [identity_function, define_D_star]]
global L_star_function = generate_L_star(num_images, blur)
# global L_star_function = [[identity_function, null_func], [null_func, identity_function], [identity_function, define_D_star]]    # for no masking
global functions = []

# degraded_image_left = matrix_to_image(vectorToImage(row1,col1,z1))
# degraded_path_l = "/Users/HP/Documents/ZIB Internship/Image Recovery/degraded_image_left.jpeg"
# save(degraded_path_l, degraded_image_left)
# degraded_image_right = matrix_to_image(vectorToImage(row1,col1,z2))
# degraded_path_r = "/Users/HP/Documents/ZIB Internship/Image Recovery/degraded_image_right.jpeg"
# save(degraded_path_r, degraded_image_right)

# degraded_images = []
for i in 1:functions_I
    deg_image_i = matrix_to_image(vectorToImage(row1,col1,z[i]))
    save("/Users/kashishgoel/Desktop/Intern_2023/Image_processing/deg_$i.jpeg",deg_image_i)
end

include("functions.jl")

global dims_I = fill(N, num_images)
global dims_K = fill(N, 2*num_images - 1)
global block_function = get_block_cyclic             #To be set by user
global generate_gamma = generate_gamma_seq      #To be set by user
global generate_mu = generate_mu_constant            #To be set by user

for i in 1:num_images
    append!(functions, [phi])
end
#Now append m Lx-z functions
for i in 1:num_images
    append!(functions,[Precompose(SqrNormL2(1/(sigma*sigma)),1,1,-z[i])])
end

#Now append m-1 Distance functions
for i in 1:num_images - 1
    append!(functions,[SqrNormL2(theta_main)])
end