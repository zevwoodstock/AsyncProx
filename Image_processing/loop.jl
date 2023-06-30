include("functions.jl")

for j in 1:iters
    global I_n = block_function(j,functions_I,1)
    global K_n = block_function(j,functions_K,2)
    update_params(j)
    check_task_delay(j)
    define_tasks(j)
    compute(j, 1)
    compute(j, 2)
    calc_theta(j)
    write(j)
    update_vars(j)
end