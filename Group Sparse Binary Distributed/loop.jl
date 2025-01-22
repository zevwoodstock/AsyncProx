vars.epoch_array = []
@everywhere j = 1
while j <= dimensions.iters
    println("j = ",j)
    # set_current_iter(j)
    @everywhere params.I  = block_function(j,dimensions.num_func_I,1)
    @everywhere params.K = block_function(j,dimensions.num_func_K,1)
    @everywhere update_params(j)
    println("params updated")
    check_task_delay(j)
    println("task delay checked")
    define_tasks(j)
    println("tasks defined")
    compute(j, 1)
    compute(j, 2)
    println("compute done")
    calc_theta(j)
    @everywhere write(j)
    @everywhere update_vars(j)
    println("vars updated")
    if params.compute_epoch_bool == true
        if compute_epoch()
            push!(vars.epoch_array,j)
            for i in 1:dimensions.num_func_I+dimensions.num_func_K
                vars.prox_call[i] = 0
            end
        end
    end
    @everywhere j += 1
end