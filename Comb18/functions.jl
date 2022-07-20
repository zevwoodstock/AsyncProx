using LinearAlgebra
using ProximalOperators
using Random

function rearrange(L::Vector{Vector{Vector}}, vars)
    L_star::Vector{Vector{Vector}} = []
    temp = []
    for i in 1:vars.I
        for k in 1:vars.K
            push!(temp, L[k][i])
        end
    end
    for i in 1:vars.I
        push!(L_star, [])
        for k in 1:vars.K
            push!(L_star[i], temp[vars.K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Int64}}, vars)
    L_star::Vector{Vector{Int64}} = []
    temp::Vector{Int64} = []
    for i in 1:vars.I
        for k in 1:vars.K
            push!(temp, L[k][i])
        end
    end
    for i in 1:vars.I
        push!(L_star, [])
        for k in 1:vars.K
            push!(L_star[i], temp[vars.K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(mat::Matrix, vars)
    return mat'
end

#a function to find the L2 norm of a vector                       
global norm_function = SqrNormL2(1)


function linear_operator_sum(function_array::Vector{Vector}, x, tr, vars)
    n = length(x)
    sum = [0, 0]
    ind = 1
    if tr==true
        ind = 2
    end
    for i in 1:n
        temp = x[i]
        sum = sum + (function_array[i][ind](temp))
    end
    return sum
end

function linear_operator_sum(weights::Vector{Int64},x, tr, vars)                   
    global s = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        global s = s+weights[i]*x[i]
    end
    return s
end

function linear_operator_sum(mat::Matrix, x, tr, vars)
    n = length(x)
    sum = [0;0]
    for i in 1:n
        temp = x[i]
        #each temp corresponds to each x_history
        #each matrix_array corresponds to vector of matrices for a vector of all Xs
        reshape(temp, 1, length(temp))
        sum = sum + mat[:, i*vars.dims-1:i*vars.dims]*temp
    end
    return sum
end

function generate_random(epsilon, ind)
    arr = []
    for i in 1:ind
        append!(arr, [1])
    end
    return arr
end

function get_L(mat::AbstractMatrix, ind, vars)
    return mat[ind*vars.dims-1:ind*vars.dims, :]
end

function get_L(vect::Vector, ind, vars)
    return vect[ind]
end

function get_minibatch(j)
#generate a random vector and its complement random vector. the random vector selects the I's in the jth iteration,
    #and the complement vector selects the I's in the (j+1)th iteration. This was, all I's are covered every two iterations
    minibatches = [[],[]]
    if j%2==0
        minibatches = [get_bitvector_pair(j, I), get_bitvector_pair(j+1, K)]
    end
    return minibatches
end

function get_bitvector_pair(iter, ind)
    random_bitvector = bitrand(MersenneTwister(iter), ind)
    empty_flag = false
    ones_flag = false
    for index in 1:ind
        if random_bitvector[index]==1
            empty_flag = true
        end
    end
    for index in 1:ind
        if random_bitvector[index]==0
            ones_flag = true
        end
    end

    if empty_flag==false
        random_bitvector[1] = 1
    end
    if ones_flag==false
        random_bitvector[1] = 0
    end

    complement_bitvector = []
    for index in 1:ind
        append!(complement_bitvector, [abs(1-random_bitvector[index])])
    end
    return [random_bitvector, complement_bitvector]
end

function check_task_delay(j, vars)
    #Checking if a task has been delayed for too long
    if j>1
        
        for b in 1:vars.tasks_num[1]
            if vars.birthdates[1][b]<j-vars.D
                newvals=fetch(vars.running_tasks[1][b])
            end
        end
        for b in 1:vars.tasks_num[2]
            if vars.birthdates[2][b]<j-D
                newvals=fetch(vars.running_tasks[2][b])
            end
        end
    end
end

function compute(vars, minibatches, v_history, mu_history, L, j, ind)
    birth = 1
    while birth<= vars.tasks_num[ind]
        if istaskdone(vars.running_tasks[ind][birth]) == true
            task = vars.task_number[ind][birth]
            #if (j==1)||(minibatches[ind][j%2+1][task]==1)
                if ind==2
                    vars.b[task],y = fetch(vars.running_tasks[ind][birth])
                    vars.b_star[task] = v_history[j][task] + (vars.l[task]-vars.b[task])./mu_history[j][task]
                else
                    vars.a[task], y = fetch(vars.running_tasks[1][birth])
                    vars.a_star[task] = (x_history[j][task]-vars.a[task])./gamma_history[j][task] - vars.l_star[task]
                end
                delete_task(ind, birth, vars)
            #else
            #    birth = birth+1
            #end
            
            if ind==2
                vars.t[task] = vars.b[task] - linear_operator_sum(get_L(L, task, vars), vars.a, false, vars)
                vars.sum_k[task] = (norm_function(vars.t[task]))*2
            else
                vars.t_star[task] = vars.a_star[task] + linear_operator_sum(get_L(rearrange(L ,vars), task, vars), vars.b_star, true, vars)
                vars.sum_i[task] = (norm_function(vars.t_star[task]))*2
            end
        else
            birth = birth+1
        end
    end
end

function custom_prox(t, f, y, gamma)
    sleep(t)
    a,b = prox(f,y,gamma)
    return a,b
end

function define_tasks(minibatches, L, v_history, x_history, gamma_history, mu_history, vars, functions, j)
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in 1:vars.I
        #if (j==1) || (minibatches[1][j%2+1][i]==1) 
            vars.l_star[i] = linear_operator_sum(get_L(rearrange(L, vars), i, vars), v_history[j], true, vars)
            delay = 0
            if i==1
                delay = 0
            end
            local task = @task custom_prox(delay,functions[i], x_history[j][i]-vars.l_star[i]*gamma_history[j][i] ,gamma_history[j][i])
            add_task(task, 1, j, i, vars)
        #end
    end

    for k in 1:vars.K
        #if (j==1)  || (minibatches[2][j%2+1][k]==1)
            vars.l[k] = linear_operator_sum(get_L(L, k, vars), x_history[j], false, vars)
            delay = 0
            if k==1
                delay = 0
            end

            local task = @task custom_prox(delay, functions[vars.I+k], vars.l[k] + mu_history[j][k]*v_history[j][k], mu_history[j][k])
            add_task(task, 2, j, k, vars)
        #end
    end
end

function calc_theta(vars, x_history, v_history, lambda, j)
    tau = 0
    for i in 1:vars.I
        tau = tau + vars.sum_i[i] 
    end

    for k in 1:vars.K
        tau = tau+vars.sum_k[k]
    end

    theta = 0

    #finding theta
    if tau > 0
        sum = 0
        #finding the sum of the dot products related to the I set
        for i in 1:vars.I
            sum = sum+dot(x_history[j][i], vars.t_star[i])-dot(vars.a[i],vars.a_star[i])
        end
        #finding the sum of the dot products related to the K set
        for k in 1:vars.K
            sum = sum+dot(vars.t[k],v_history[j][k])-dot(vars.b[k],vars.b_star[k])
        end
        #using the 2 sums to find theta according to the formula
        theta = lambda*max(0,sum)/tau
    end
    return theta
end

function update_vars(x_history, v_history, j, theta, vars)
    x = x_history[j]
    for i in 1:vars.I
        x[i] = x_history[j][i] - theta*vars.t_star[i]
    end
    append!(x_history, [x])
    v_star = v_history[j]
    for k in 1:vars.K
        v_star[k] = v_history[j][k] - theta*vars.t[k]
    end
    append!(v_history, [v_star])

    #sleep(0.001)
    #println(x_history[j])
    if j==3
        print("First: ")
        println(x_history[j])
    end
    if j==vars.iters
        print("Final ans: ")
        println(x_history[j])
    end
    return x_history, v_history
end

function update_params(j, gamma_history, mu_history)
    lambda = 1/(j-0.5) + 0.5
    append!(gamma_history, [generate_random(vars.epsilon, vars.I)])
    append!(mu_history, [generate_random(vars.epsilon, vars.K)])

    return lambda, gamma_history, mu_history
end

function delete_task(ind, birth, vars)
    deleteat!(vars.running_tasks[ind], birth)
    deleteat!(vars.birthdates[ind], birth)
    deleteat!(vars.task_number[ind], birth)
    vars.tasks_num[ind] = vars.tasks_num[ind] - 1
end

function add_task(task, ind, j, i, vars)
    schedule(task)
    push!(vars.running_tasks[ind], task)
    push!(vars.birthdates[ind], j)
    push!(vars.task_number[ind], i)
    vars.tasks_num[ind] = vars.tasks_num[ind] + 1
end