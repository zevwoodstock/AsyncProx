using LinearAlgebra
using ProximalOperators
using Random

function rearrange(L::Vector{Vector{Vector}})
    L_star::Vector{Vector{Vector}} = []
    temp = []
    for i in 1:functions_I
        for k in 1:functions_K
            push!(temp, L[k][i])
        end
    end
    for i in 1:functions_I
        push!(L_star, [])
        for k in 1:functions_K
            push!(L_star[i], temp[functions_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Int64}})
    L_star::Vector{Vector{Int64}} = []
    temp::Vector{Int64} = []
    for i in 1:functions_I
        for k in 1:functions_K
            push!(temp, L[k][i])
        end
    end
    for i in 1:functions_I
        push!(L_star, [])
        for k in 1:functions_K
            push!(L_star[i], temp[functions_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(mat::Matrix)
    return mat'
end

#a function to find the L2 norm of a vector                       
global norm_function = SqrNormL2(1)


function linear_operator_sum(function_array::Vector{Vector}, x, tr)
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

function linear_operator_sum(weights::Vector{Int64},x, tr)                   
    global s = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        global s = s+weights[i]*x[i]
    end
    return s
end

function linear_operator_sum(mat::Matrix, x, tr)
    n = length(x)
    sum = [0;0]
    for i in 1:n
        temp = x[i]
        #each temp corresponds to each res.x
        #each matrix_array corresponds to vector of matrices for a vector of all Xs
        reshape(temp, 1, length(temp))
        sum = sum + mat[:, i*dims-1:i*dims]*temp
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

function get_L(mat::AbstractMatrix, ind)
    return mat[ind*dims-1:ind*dims, :]
end

function get_L(vect::Vector, ind)
    return vect[ind]
end

# function get_minibatch(j)
# #generate a random vector and its complement random vector. the random vector selects the I's in the jth iteration,
#     #and the complement vector selects the I's in the (j+1)th iteration. This was, all I's are covered every two iterations
#     minibatches = [[],[]]
#     if j%2==0
#         minibatches = [get_bitvector_pair(j, functions_I), get_bitvector_pair(j+1, functions_K)]
#     end
#     return minibatches
# end

# function get_bitvector_pair(iter, ind)
#     random_bitvector = bitrand(MersenneTwister(iter), ind)
#     empty_flag = false
#     ones_flag = false
#     for index in 1:ind
#         if random_bitvector[index]==1
#             empty_flag = true
#         end
#     end
#     for index in 1:ind
#         if random_bitvector[index]==0
#             ones_flag = true
#         end
#     end

#     if empty_flag==false
#         random_bitvector[1] = 1
#     end
#     if ones_flag==false
#         random_bitvector[1] = 0
#     end

#     complement_bitvector = []
#     for index in 1:ind
#         append!(complement_bitvector, [abs(1-random_bitvector[index])])
#     end
#     return [random_bitvector, complement_bitvector]
# end

function check_task_delay(j)
    #Checking if a task has been delayed for too long
    if j>1
        
        for b in 1:vars.tasks_num[1]
            if vars.birthdates[1][b]<j-D
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

function compute(j, ind)
    # change to be done - once we introduce blocks we need to incorporate the i belonging to I but not I_n step
    birth = 1
    while birth<= vars.tasks_num[ind]
        if istaskdone(vars.running_tasks[ind][birth]) == true
            task = vars.task_number[ind][birth]
            #if (j==1)||(minibatches[ind][j%2+1][task]==1)
                if ind==2
                    vars.b[task],y = fetch(vars.running_tasks[ind][birth])
                    vars.b_star[task] = res.v_star[j][task] + (vars.l[task]-vars.b[task])./vars.mu_history[j][task]
                else
                    vars.a[task], y = fetch(vars.running_tasks[1][birth])
                    vars.a_star[task] = (res.x[j][task]-vars.a[task])./vars.gamma_history[j][task] - vars.l_star[task]
                end
                delete_task(ind, birth)
            #else
            #    birth = birth+1
            #end
            
            if ind==2
                vars.t[task] = vars.b[task] - linear_operator_sum(get_L(L, task), vars.a, false)
                vars.sum_k[task] = (norm_function(vars.t[task]))*2               
            else
                vars.t_star[task] = vars.a_star[task] + linear_operator_sum(get_L(rearrange(L), task), vars.b_star, true)
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

function define_tasks(j)     ## no change required hopefully for the generalisation of n dimensions
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in 1:functions_I                  # future change  - incorporate blocks into this, right now running over entire I
        #if (j==1) || (minibatches[1][j%2+1][i]==1) 
            vars.l_star[i] = linear_operator_sum(get_L(rearrange(L), i), res.v_star[j], true)
            
            ###### doubt - what is the use of this delay thing ######
            delay = 0
            if i==1
                delay = 0
            end
            ####################
            local task = @task custom_prox(delay,functions[i], res.x[j][i]-vars.l_star[i]*vars.gamma_history[j][i] ,vars.gamma_history[j][i])
            add_task(task, 1, j, i)
        #end
    end

    for k in 1:functions_K
        #if (j==1)  || (minibatches[2][j%2+1][k]==1)
            vars.l[k] = linear_operator_sum(get_L(L, k), res.x[j], false)
            delay = 0
            if k==1
                delay = 0
            end

            local task = @task custom_prox(delay, functions[functions_I+k], vars.l[k] + vars.mu_history[j][k]*res.v_star[j][k], vars.mu_history[j][k])
            add_task(task, 2, j, k)
        #end
    end
end

function calc_theta(j)   ## no change required hopefully
    lambda = 1/(j-0.5) + 0.5     # design decision
    tau = 0
    for i in 1:functions_I
        tau = tau + vars.sum_i[i] 
    end

    for k in 1:functions_K
        tau = tau+vars.sum_k[k]
    end

    global theta = 0

    #finding theta
    if tau > 0
        sum = 0
        #finding the sum of the dot products related to the I set
        for i in 1:functions_I
            sum = sum+dot(res.x[j][i], vars.t_star[i])-dot(vars.a[i],vars.a_star[i])
        end
        #finding the sum of the dot products related to the K set
        for k in 1:functions_K
            sum = sum+dot(vars.t[k],res.v_star[j][k])-dot(vars.b[k],vars.b_star[k])
        end
        #using the 2 sums to find theta according to the formula
        global theta = lambda*max(0,sum)/tau
    end
end

function update_vars(j)
    x = res.x[j]
    for i in 1:functions_I
        x[i] = res.x[j][i] - theta*vars.t_star[i]
    end
    push!(res.x, x)
    v_star = res.v_star[j]
    for k in 1:functions_K
        v_star[k] = res.v_star[j][k] - theta*vars.t[k]
    end
    push!(res.v_star, v_star)

    """sleep(0.001)
    println(res.x[j])
    if j==Iters
        print("Final ans: ")
        println(res.x[j])
    end"""
end

function update_params(j)
    # change required - make the choosing of gamma and mu random
    append!(vars.gamma_history, [generate_random(epsilon, functions_I)])
    append!(vars.mu_history, [generate_random(epsilon, functions_K)])
end

function delete_task(ind, birth)
    deleteat!(vars.running_tasks[ind], birth)
    deleteat!(vars.birthdates[ind], birth)
    deleteat!(vars.task_number[ind], birth)
    vars.tasks_num[ind] = vars.tasks_num[ind] - 1
end

function add_task(task, ind, j, i)
    schedule(task)
    push!(vars.running_tasks[ind], task)
    push!(vars.birthdates[ind], j)
    push!(vars.task_number[ind], i)
    vars.tasks_num[ind] = vars.tasks_num[ind] + 1
end

function write(j)
    mode = "a"
    if j==1
        mode = "w"
    end
    for i in 1:functions_I
        open("x" * string(i,base = 10) * ".txt",mode) do io
            println(io,res.x[j][i])
        end
    end
    # open("x1.txt",mode) do io
    #     println(io,res.x[j][1])
    # end
    # open("x2.txt",mode) do io
    #     println(io,res.x[j][2])
    # end
end

function check_feasibility()
    feasible = true
    for i in 1:functions_I
        if(functions[i](res.x[iters][i])==Inf)
            feasible = false
        end
    end

    for k in 1:functions_K
        if(functions[functions_I + k](linear_operator_sum(get_L(L, k), res.x[iters], false))==Inf)
            feasible = false
        end
    end
    return feasible
end
