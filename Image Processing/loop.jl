@everywhere j = 1
while j <= dimensions.iters
    println("iteration number: ",j)
    @everywhere params.I = params.block_function(j,dimensions.num_func_I,1)
    @everywhere params.K = params.block_function(j,dimensions.num_func_K,2)
    println("here 1")
    @everywhere update_params(j)
    println("here 2")
    check_task_delay(j)
    println("here 3")
    define_tasks(j)
    println("here 4")
    compute(j, 1)
    compute(j, 2)
    calc_theta(j)
    println("here 5")
    @everywhere write(j)
    @everywhere update_vars(j)
    println("here 6")

    if params.compute_epoch_bool == true
        if compute_epoch()
            push!(vars.epoch_array,j)
            for i in 1:dimensions.num_func_I+dimensions.num_func_K
                vars.prox_call[i] = 0
            end
        end
    end
    @everywhere j += 1
    println("here 7")
end
