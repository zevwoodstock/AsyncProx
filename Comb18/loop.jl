include("functions.jl")

for j in 1:iters
    update_params(j)
    if j%2==0
        global minibatches = get_minibatch(j)
    end
    check_task_delay(j)
    define_tasks(j)
    compute(j, 1)
    compute(j, 2)
    calc_theta(j)
    write(j)
    update_vars(j)
end