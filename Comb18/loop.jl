using LinearAlgebra
using ProximalOperators
using Random

include("functions.jl")

function execute(vars, gamma_history, mu_history, L, v_history, x_history,functions)
    global mode = "w" #to write to file
    for j in 1:vars.iters
        global lambda, gamma_history, mu_history = update_params(j, gamma_history, mu_history)

        if j%2==0
            global minibatches = get_minibatch(j)
        end

        #println(minibatches)
        check_task_delay(j, vars)
        define_tasks(minibatches, L, v_history, x_history, gamma_history, mu_history, vars, functions, j)
        compute(vars, minibatches, v_history, mu_history, L, j, 1)
        compute(vars, minibatches, v_history, mu_history, L, j, 2)
        theta = calc_theta(vars, x_history, v_history, lambda, j)

        if j>1
            global mode = "a"
        end
        open("x1.txt",mode) do io
            println(io,x_history[j][1])
        end
        open("x2.txt",mode) do io
            println(io,x_history[j][2])
        end

        global x_history, v_history = update_vars(x_history, v_history, j, theta, vars)
        #println(x_history[j])
    end
    open("x1.txt","a") do io
        println(io,x_history[vars.iters][1])
    end
    open("x2.txt",mode) do io
        println(io,x_history[vars.iters][2])
    end

    print("Final ans: ")
    println(x_history[vars.iters])
    return x_history, v_history
end