dimensions = dim_struct(0, 0, [], [], 0, 0)

params = parameters(Int64[],                      # I
                    Int64[],                      # K
                    Float64[],                    # mu_start
                    Float64[],                    # mu_end
                    Float64[],                    # mu_step
                    Float64[],                    # gamma_start
                    Float64[],                    # gamma_end
                    Float64[],                    # gamma_step
                    Float64[],                    # gamma_a
                    Float64[],                    # gamma_b
                    0.01,                         # epsilon
                    Float64[],                    # mu_a
                    Float64[],                    # mu_b
                    0.0,                          # constant_gamma
                    0.0,                          # constant_mu
                    Float64[],                    # constant_g
                    Float64[],                    # constant_m
                    Vector{Vector{Float64}}[],    # mu_k
                    Float64[],                    # beta_k
                    false,                        # compute_epoch_bool
                    false,                        # record_residual
                    false,                        # record_func
                    false,                        # record_dist
                    0,                            # record_method
                    0.5,                          # alpha_
                    0.5,                          # beta_
                    0,                            # max_iter_delay #default sync
                    0,                            # max_task_delay #default sync
                    String[],                     # img_path_array
                    0,                            # num_images
                    Int64[],                      # left_shift_pixels
                    Int64[],                      # right_shift_pixels
                    false,                        # randomize_initial
                    false,                        # initialize_with_zi
                    get_block_cyclic,             # block_function
                    generate_gamma_constant,      # generate_gamma_function
                    generate_mu_constant,         # generate_mu_function
                    0.01,                         # sigma
                    0.0)                          # sigma_blur
