include("functions.jl")

for j in 1:iters
    global I_n = block_function(j,functions_I) # Keep M to be less than or equal to functions_I
    global K_n = block_function(j,functions_K) # Keep M to be less than or equal to functions_K
    update_params(j)
    check_task_delay(j)
    define_tasks(j)
    compute(j, 1)
    compute(j, 2)
    calc_theta(j)
    write(j)
    update_vars(j)
end
record()
