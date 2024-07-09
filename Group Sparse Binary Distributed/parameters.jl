# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 
if length(ARGS)>0
    if ARGS[1]=="false"
        L_function_bool = false
    else
        L_function_bool = true
    end
    dimensions.num_func_I = parse(Int, ARGS[2])
    dimensions.d = parse(Int, ARGS[3])
    dimensions.num_func_K = parse(Int, ARGS[4])
    params.q_datacenters = parse(Int, ARGS[5])
    dimensions.iters = parse(Int, ARGS[6])
    params.max_iter_delay = parse(Int, ARGS[7])
    params.alpha_ = parse(Float64, ARGS[8])
    params.beta_ = parse(Float64, ARGS[9])
    if ARGS[10]=="false"
        params.compute_epoch_bool = false
    else
        params.compute_epoch_bool = true
    end
    if ARGS[11]=="false"
        params.record_residual = false
    else
        params.record_residual = true
    end
    if ARGS[12]=="false"
        params.record_func = false
    else
        params.record_func = true
    end
    if ARGS[13]=="false"
        params.record_dist = false
    else
        params.record_dist = true
    end
    record_method = parse(Int, ARGS[14])
    
else
    L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions.

    dimensions.num_func_I = 1429 # The number of intervals (number of Gi)
    # dimensions.num_func_I = 100 # (m)
    dimensions.d = 10000
    # dimensions.d = 1000 # (d)
    dimensions.num_func_K = 100
    # dimensions.num_func_K = 300 # (p)

    params.q_datacenters = 100
    dimensions.iters = 10
    params.max_iter_delay = 0
    params.max_task_delay = 0
    params.alpha_ = 0.5
    params.beta_ = 0.5

    params.compute_epoch_bool = false       # Necessary if record method is "1" - epoch numbers
    params.record_residual = false          # record_residual = 1 for storing ||x_{n+1} - x_n||^2
    params.record_func = true               # record_func = 1 for storing f(x_n)
    params.record_dist = false              # record_dist = 1 for storing ||x_* - x_n||^2

    # the variable record_method indicates the type of variable you wish to use for the x_axis
    # 0 is used for plotting against the number of iterations
    # 1 is used for plotting against the epoch number
    # 2 is used to plot against the number of prox calls
    # 3 is used to plot against the wall clock time
    record_method = 0 
    
end

block_function = get_block_cyclic
