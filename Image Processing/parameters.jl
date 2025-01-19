# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 
if length(ARGS)>0
    if ARGS[1]=="false"
        L_function_bool = false
    else
        L_function_bool = true
    end
    dimensions.d = parse(Int, ARGS[2])
    dimensions.iters = parse(Int, ARGS[3])
    params.max_iter_delay = parse(Int, ARGS[4])
    params.alpha_ = parse(Float64, ARGS[5])
    params.beta_ = parse(Float64, ARGS[6])
    if ARGS[7]=="false"
        params.compute_epoch_bool = false
    else
        params.compute_epoch_bool = true
    end
    if ARGS[8]=="false"
        params.record_residual = false
    else
        params.record_residual = true
    end
    if ARGS[9]=="false"
        params.record_func = false
    else
        params.record_func = true
    end
    if ARGS[10]=="false"
        params.record_dist = false
    else
        params.record_dist = true
    end
    params.record_method = parse(Int, ARGS[11])
    params.randomize_initial = parse(Bool, ARGS[12])
    params.initialize_with_zi = parse(Bool, ARGS[13])

    # block_cyclic function
    if ARGS[14] == 1
        params.block_function = get_block_cyclic
    end
    # generate_gamma_function setting
    if ARGS[15] == 1
        params.generate_gamma_function = generate_gamma_constant
    end
    if ARGS[15] == 2
        params.generate_gamma_function = generate_gamma_seq
    end
    if ARGS[15] == 3
        params.generate_gamma_function = generate_gamma_linear_decrease
    end
    if ARGS[15] == 4
        params.generate_gamma_function = generate_gamma_random
    end
    if ARGS[15] == 5
        params.generate_gamma_function = generate_gamma_nonlinear_decrease
    end
    # generate_gamma_function setting
    if ARGS[16] == 1
        params.generate_mu_function = generate_mu_constant
    end
    if ARGS[16] == 2
        params.generate_mu_function = generate_mu_seq
    end
    if ARGS[16] == 3
        params.generate_mu_function = generate_mu_linear_decrease
    end
    if ARGS[16] == 4
        params.generate_mu_function = generate_mu_random
    end
    if ARGS[16] == 5
        params.generate_mu_function = generate_mu_nonlinear_decrease
    end
    
    dimensions.num_images = parse(Int, ARGS[17])
    for i in 1:dimensions.num_images
        push!(params.img_path_array, parse(string, ARGS[17 + i]))
    end
    for i in 1:dimensions.num_images-1
        push!(params.left_shift_pixels, parse(Int, ARGS[17 + dimensions.num_images + i]))
    end
    for i in 1:dimensions.num_images-1
        push!(params.right_shift_pixels, parse(Int, ARGS[17 + 2*dimensions.num_images - 1 + i]))
    end

    params.sigma_blur = parse(Float64, ARGS[16 + 3*dimensions.num_images])
    
else
    L_function_bool = true  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions.

    dimensions.iters = 300
    params.max_iter_delay = 150 # 10/2 for async
    # params.max_task_delay = 1000 #inf for async
    params.alpha_ = 0.5
    params.beta_ = 0.5
    params.sigma_blur = 500.0

    params.compute_epoch_bool = false       # Necessary if record method is "1" - epoch numbers
    params.record_residual = false          # record_residual = 1 for storing ||x_{n+1} - x_n||^2
    params.record_func = true               # record_func = 1 for storing f(x_n)
    params.record_dist = false              # record_dist = 1 for storing ||x_* - x_n||^2

    params.randomize_initial = false
    params.initialize_with_zi = true
    params.block_function = get_block_cyclic
    params.generate_gamma_function = generate_gamma_constant
    params.generate_mu_function = generate_mu_constant

    # the variable record_method indicates the type of variable you wish to use for the x_axis
    # 0 is used for plotting against the number of iterations
    # 1 is used for plotting against the epoch number
    # 2 is used to plot against the number of prox calls
    # 3 is used to plot against the wall clock time
    params.record_method = 0
    params.num_images = 4
    params.img_path_array = ["Image Processing/orig_1.jpeg",
                        "Image Processing/orig_2.jpeg",
                        "Image Processing/orig_3.jpeg",
                        "Image Processing/orig_4.jpeg"]
    dimensions.num_func_I = params.num_images
    dimensions.num_func_K = 2*params.num_images - 1
    params.left_shift_pixels = [8,5,12]
    params.right_shift_pixels = [8,5,12]
    
end
