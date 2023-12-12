include("problem.jl")

# global epoch_found = false
global epoch_array = []
global prox_count_array = []

for j in 1:iters
    println("j = ",j)
    # println("theta = ", theta_main)
    global I_n = block_function(j,functions_I,1)
    global K_n = block_function(j,functions_K,1)
    println("block functions made")
    update_params(j)
    println("parameters updated")
    check_task_delay(j)
    println("task delay checked")
    define_tasks(j)
    println("tasks defined")
    compute(j, 1)
    println("computed j, 1")
    compute(j, 2)
    println("computed j, 2")
    calc_theta(j)
    println("calculated theta")
    write(j)
    update_vars(j)
    println("updated vars(j)")
    push!(prox_count_array,prox_call_count)

    if compute_epoch_bool == true
        if compute_epoch()
            # global epoch_found = true
            push!(epoch_array,j)
            for i in 1:functions_I+functions_K
                prox_call[i] = 0
            end
        end
    end
end
global final_ans = res.x[iters]
record()

