vars.epoch_array = []

for j in 1:dimensions.iters
    println("j = ",j)
    params.I  = block_function(j,dimensions.num_func_I,1)
    params.K = block_function(j,dimensions.num_func_K,1)
    # println("block functions made")
    update_params(j)
    # println("parameters updated")
    check_task_delay(j)
    # println("task delay checked")
    define_tasks(j)
    # println("tasks defined")
    compute(j, 1)
    # println("computed j, 1")
    compute(j, 2)
    # println("computed j, 2")
    calc_theta(j)
    # println("calculated theta")
    write(j)
    update_vars(j)
    # println("updated vars(j)\n")

    if params.compute_epoch_bool == true
        if compute_epoch()
            push!(vars.epoch_array,j)
            for i in 1:dimensions.num_func_I+dimensions.num_func_K
                vars.prox_call[i] = 0
            end
        end
    end
end