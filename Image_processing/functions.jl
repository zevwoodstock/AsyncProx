using LinearAlgebra
using ProximalOperators
using Random

if L_function_bool == true
    L = L_function
end

function get_block_cyclic(n::Int64, m::Int64 = 20, M::Int64 = 5) 
    block_size = div(m, M)
    start = (((n%M) - 1) * block_size) % m + 1
    fin = start + block_size - 1

    if n % M == 0
        start = ((M - 1)* block_size)%m + 1
        fin = max(fin, m)
    end

    arr = Int64[]
    for i in start:fin
        push!(arr, i)
        # println(arr)
    end
    return arr
end

function rearrange(L::Vector{Vector{Vector}})
    # println("1")
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

function rearrange(L::Vector{Vector{Matrix{Float64}}})
    # println("yeah")
    L_star::Vector{Vector{Matrix}} = []
    temp = []
    for i in 1:functions_I
        for k in 1:functions_K
            new_matrix = L[k][i]'
            # println(new_matrix)
            push!(temp, new_matrix)
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
    # println("3")
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

function rearrange(L::Vector{Vector{Function}})
    L_star::Vector{Vector{Function}} = []
    temp = []
    for i in 1:functions_I
        for k in 1:functions_K
            new_matrix = L_star_function[k][i]
            println(new_matrix)
            push!(temp, new_matrix)
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
    # println("4")
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

function matrix_dot_product(v::Vector{Matrix}, u::Vector{Vector{Float64}})
    ans =  v[1]*reshape(u[1], length(u[1]), 1)
    ans*=0
    # println("ans = ", ans)
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1 * column_vec
        ans = ans + matrix_product
    end
    # println("ans is ", ans)
    return vec(ans)
end

function matrix_dot_product(v::Vector{Matrix{Float64}}, u::Vector{Vector{Float64}})
    ans =  v[1]*reshape(u[1], length(u[1]), 1)
    ans*=0
    # println("ans = ", ans)
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1 * column_vec
        ans = ans + matrix_product
    end
    # println("ans is ", ans)
    return vec(ans)
end

function matrix_dot_product(v::Vector{Function}, u::Vector{Vector{Float64}})
    ans =  v[1](reshape(u[1], length(u[1]), 1))
    ans*=0
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1(column_vec)
        ans = ans + matrix_product
    end
    # println("ans is ", ans)
    return vec(ans)
end

function generate_gamma_constant(i,j)
    return constant_g[i]
end

function generate_mu_constant(k,j)
    return constant_m[k]
end

function generate_gamma_seq(i,j)
    if j == 1
        return 1/epsilon
    else
        if vars.gamma_history[j-1][i] == epsilon
            return epsilon
        else
            return ((1/epsilon) - 0.1*(j-1))
        end
    end
end

function generate_mu_seq(k,j)
    if j == 1
        return 1/epsilon
    else
        if vars.mu_history[j-1][k] == epsilon
            return epsilon
        else
            return ((1/epsilon) - 0.1*(j-1))   # in future we can also set the subtraction constant different for different i and k using the constant arrays made in main.jl
        end
    end
end

function get_L(mat::AbstractMatrix, ind)
    return mat[ind*dims-1:ind*dims, :]
end

function get_L(vect::Vector, ind)
    return vect[ind]
end

function check_task_delay(j)
    #Checking if a task has been delayed for too long
    if j>1
        for b in 1:vars.tasks_num[1]
            if vars.birthdates[1][b]<j-D
                newvals=fetch(vars.running_tasks[1][b])
                task_no = vars.task_number[1][b]
                vars.a[task_no] , y= newvals 
                vars.a_star[task_no] = (res.x[j][task_no]-vars.a[task_no])./vars.gamma_history[j][task_no] - vars.l_star[task_no]
            end
        end
        for b in 1:vars.tasks_num[2]
            if vars.birthdates[2][b]<j-D
                newvals=fetch(vars.running_tasks[2][b])
                task_no = vars.task_number[2][b]
                vars.b[task_no] , y= newvals 
                vars.b_star[task_no] = res.v_star[j][task_no] + (vars.l[task_no]-vars.b[task_no])./vars.mu_history[j][task_no]
            end
        end
    end
end

function compute(j, ind)
    birth = 1
    while birth<= vars.tasks_num[ind]
        if istaskdone(vars.running_tasks[ind][birth]) == true
            task = vars.task_number[ind][birth]
                if ind==2
                    vars.b[task],y = fetch(vars.running_tasks[ind][birth])
                    vars.b_star[task] = res.v_star[j][task] + (vars.l[task]-vars.b[task])./vars.mu_history[j][task]
                else
                    vars.a[task], y = fetch(vars.running_tasks[1][birth])
                    vars.a_star[task] = (res.x[j][task]-vars.a[task])./vars.gamma_history[j][task] - vars.l_star[task]
                end
                delete_task(ind, birth)
        
            if ind==2
                vars.t[task] = vars.b[task] - matrix_dot_product(get_L(L, task), vars.a)
                vars.sum_k[task] = (norm_function(vars.t[task]))*2               
            else
                vars.t_star[task] = vars.a_star[task] +  matrix_dot_product(get_L(rearrange(L), task), vars.b_star)
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

function define_tasks(j)    
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in I_n                  # change  - incorporated blocks into this, now running over entire I_n
            vars.l_star[i] = matrix_dot_product(get_L(rearrange(L), i), res.v_star[j]) 
            delay = 0
            local task = @task custom_prox(delay,functions[i], res.x[j][i]-vars.l_star[i]*vars.gamma_history[j][i] ,vars.gamma_history[j][i])
            add_task(task, 1, j, i)
    end

    for k in K_n
            vars.l[k] = matrix_dot_product(get_L(L, k), res.x[j])
            delay = 0
            local task = @task custom_prox(delay, functions[functions_I+k], vars.l[k] + vars.mu_history[j][k]*res.v_star[j][k], vars.mu_history[j][k])
            add_task(task, 2, j, k)
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
    # check if this change needs to be done - when appending to the gamma history array, we need to input the ind in generate random corresponding to I_n, 
    # right now it is functions_I cause there are no blocks; basically we need to generate lambda and mu for the i which belong to the block I_n however, 
    # it doesn't harm if it is generated for all i in I
    push!(vars.gamma_history, [])
    push!(vars.mu_history, [])
    for i in 1:functions_I
        push!(vars.gamma_history[j], generate_gamma(i,j))
    end
    for k in 1:functions_K
        push!(vars.mu_history[j], generate_mu(k,j))
    end
    # append!(vars.gamma_history, [generate_random(epsilon, functions_I)])
    # append!(vars.mu_history, [generate_random(epsilon, functions_K)])
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
        if(functions[functions_I + k](matrix_dot_product(get_L(L, k), res.x[iters]))==Inf)
            feasible = false
        end
    end
    return feasible
end