vars.epoch_array = []
@everywhere j = 1
while j <= dimensions.iters
    println("iteration number: ",j)
    @everywhere params.I = params.block_function(j,dimensions.num_func_I,1)
    @everywhere params.K = params.block_function(j,dimensions.num_func_K,2)
    @everywhere update_params(j)
    check_task_delay(j)
    define_tasks(j)
    compute(j, 1)
    compute(j, 2)
    calc_theta(j)
    @everywhere write(j)
    @everywhere update_vars(j)

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
