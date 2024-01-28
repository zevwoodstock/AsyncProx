# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 

L_function_bool = false  #Set this to be true if you want to input L as a Matrix of functions. Need to declare adjoint functions.

# dimensions.num_func_I = 1429 # The number of intervals (number of Gi)
dimensions.num_func_I = 100 # (m)
# dimensions.d = 10000
dimensions.d = 1000 # (d)
# dimensions.num_func_K = 100
dimensions.num_func_K = 300 # (p)

conditions = []
push!(conditions, [Condition() for _ in 1:dimensions.num_func_I])
push!(conditions, [Condition() for _ in 1:dimensions.num_func_K])

params.q_datacenters = 100
dimensions.iters = 10
params.max_task_delay = 40
block_function = get_block_cyclic
params.alpha_ = 0.5
params.beta_ = 0.5

params.compute_epoch_bool = false       # Necessary if record method is "1" - epoch numbers
params.record_residual = false          # record_residual = 1 for storing ||x_{n+1} - x_n||^2
params.record_func = true               # record_func = 1 for storing f(x_n)
params.record_dist = false              # record_dist = 1 for storing ||x_* - x_n||^2

# the variable record_method indicates the type of variable you wish to use for the x_axis
# "0" is used for plotting against the number of iterations
# "1" is used for plotting against the epoch number
# "2" is used to plot against the number of prox calls
# "3" is used to plot against the wall clock time
record_method = "0"  