include("functions.jl")
include("problem.jl")

global epoch_found = false
global epoch_array = []

for j in 1:iters
    println("j = ",j)
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
    if compute_epoch_bool == true
        # if epoch_found == false
        if compute_epoch()
            global epoch_found = true
            push!(epoch_array,j)
            for i in 1:functions_I+functions_K
                prox_call[i] = 0
            end
        end
        # end
    end
end

record()
