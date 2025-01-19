dimensions.num_func_I = params.num_images
dimensions.num_func_K = 2*params.num_images - 1

dimensions.dims_array_I = fill(dimensions.d, dimensions.num_func_I)
dimensions.dims_array_K = fill(dimensions.d, dimensions.num_func_K)
L_matrix = fill(fill(1.0, dimensions.num_func_I), dimensions.num_func_K)

L_function = generate_L(params.num_images, blur)
L_star_function = generate_L_star(params.num_images, blur)

if L_function_bool == true
    L_operator = L_function
else
    L_operator = L_matrix
end

L_operator_transpose = rearrange(L_operator)

if params.record_method == 1
    params.compute_epoch_bool = true
end

params.constant_g = []   # this is defined if for generate_gamma the strategy taken is generate_gamma_constant
params.constant_m = []   # this is defined if for generate_mu the strategy taken is generate_mu_constant
calculate_mu_beta()

img_arr = []
for i in 1:params.num_images
    push!(img_arr, imageToVector(image_to_vector(params.img_path_array[i])))
end

num_rows, num_cols = get_row_column(params.img_path_array[1])
N = num_rows * num_cols
# saving the original images here that will be later degraded and recovered
for i in 1:dimensions.num_func_I
    orig_image_i = matrix_to_image(vectorToImage(num_rows, num_cols, img_arr[i]))
    println("saving the original images here")
    save("/Users/kashishgoel/Desktop/Intern_2023/Multiple_Image_processing/orig_$i.jpeg",orig_image_i)
end

z = []
for i in 1:dimensions.num_func_I
    copy_i = copy(img_arr[i])
    push!(z,copy_i)
end

w = []
for i in 1:dimensions.num_func_I
    push!(w,generate_random_vector(N, params.sigma))
    # norm_wi = norm_function(w[i])*2
    # w[i] = w[i]/norm_wi
end

# adding noise to the image
for i in 1:dimensions.num_func_I
    for j in 1:N
        z[i][j] = z[i][j] + w[i][j]
    end
end

# blurring the image
for i in 1:dimensions.num_func_I
    z[i] = blur(z[i])
end

if params.randomize_initial == true
    for i in 1:dimensions.num_func_I
        res.x[1][i] = w[i]
    end
end

if params.initialize_with_zi == true
    for i in 1:dimensions.num_func_I
        res.x[1][i] = z[i]
    end
end

# saving the degraded images here
for i in 1:dimensions.num_func_I
    deg_image_i = matrix_to_image(vectorToImage(num_rows, num_cols, z[i]))
    save("/Users/kashishgoel/Desktop/Intern_2023/Multiple_Image_processing/deg_$i.jpeg",deg_image_i)
end

functions = []
global theta_main = 0.1
for i in 1:params.num_images
    append!(functions, [phi])
end
for i in 1:params.num_images
    append!(functions,[Precompose(SqrNormL2(1/(params.sigma*params.sigma)),1,1,-z[i])])
end
for i in 1:params.num_images - 1
    append!(functions,[SqrNormL2(theta_main)])
end

zeros_I = []
zeros_K = []

for i in 1:dimensions.num_func_I
    append!(zeros_I, [zeros(dimensions.dims_array_I[i])])
    append!(params.constant_g, params.constant_gamma)
end
for i in 1:dimensions.num_func_K
    append!(zeros_K, [zeros(dimensions.dims_array_K[i])])
    append!(params.constant_m, params.constant_mu)
end

vars = variables(zeros_I, 
                zeros_I, 
                zeros_K, 
                zeros_K,
                zeros_K, 
                zeros_I, 
                zeros_K, 
                zeros_I,
                zeros(dimensions.num_func_I),       
                zeros(dimensions.num_func_K),   
                [[],[]],     
                [0,0],       
                [[],[]],     
                [[],[]],     
                [],
                [],
                0.0,
                0,
                [],
                Vector{Vector{Vector{Float64}}}(undef, dimensions.iters),
                Vector{Vector{Vector{Float64}}}(undef, dimensions.iters),
                [],
                Vector{Vector{Float64}}([]),
                [],
                [],
                [],
                [])

res = result([zeros_I],
            [zeros_K])

for i in 1:dimensions.num_func_I+dimensions.num_func_K
    push!(vars.prox_call,0)
end