# A function to find the L2 norm of a vector                       
norm_function = SqrNormL2(1)

# Implement the call method for the HingeDot function type
function (f::HingeDot)(x)
    linear_result = Linear(f.mu[f.k])(x)
    hinge_loss_result = HingeLoss([f.beta[f.k]], 10)([linear_result])
    return hinge_loss_result
end

function ProximalOperators.prox!(y, f::HingeDot, x, gamma)
    datacenter_index = params.mapping_ping[f.k]
    sleep_time = params.ping_array[datacenter_index]
    n_error = randn()*0.005 #implies std_dev of normal error is 0.005 
    sleep(abs(n_error) + sleep_time)
    mu2 = SqrNormL2(2)(f.mu[f.k])
    Lx = Linear(f.mu[f.k])(x)
    p,v = prox(HingeLoss([f.beta[f.k]], 10), [Lx], gamma)
    p = p[1]
    p = p - Lx
    p = p*f.mu[f.k] #L*(v \in R^1) = v mu_k
    y .= (x + (1/mu2)*p)
    return f(y)
end

function generate_G_x(m, d)
    g = [Int[] for _ in 1:m]
    x = [Float64[0.0 for _ in 1:d] for _ in 1:m]
    start = 1
    for i in 1:m
        if i == 1
            start = 1
        else
            start += 7
        end
        
        temp = Int[]
        for j in 1:10
            if start + j - 1 > d
                break
            end
            push!(temp, start + j - 1)
            x[i][start + j - 1] = randn()*1000
        end
        g[i]= temp
    end
    return g, x
end

function calculate_mu_beta()
    G, original_x = generate_G_x(dimensions.num_func_I, dimensions.d)
    original_y = sum(original_x, dims=1)[1]
    d_one = fill(1.0, dimensions.d)
    
    for _ in 1:dimensions.num_func_K
        random_vector = randn(Float64, dimensions.d)
        rnorm = NormL2(1)(random_vector)
        random_vector = random_vector/rnorm
        push!(params.mu_k, random_vector)
    end
    w_temp = []
    for i in 1:dimensions.num_func_K
        if i%4==0
            push!(w_temp, -1)
        else
            push!(w_temp, 1)
        end
    end
    w = shuffle(w_temp)

    for i in 1:dimensions.num_func_K
        push!(params.beta_k, w[i]*sign(dot(params.mu_k[i], original_y)))
    end

end

function rearrange(L::Vector{Vector{Vector}})
    L_star::Vector{Vector{Vector}} = []
    temp = []
    for i in 1:dimensions.num_func_I
        for k in 1:dimensions.num_func_K
            push!(temp, L[k][i])
        end
    end
    for i in 1:dimensions.num_func_I
        push!(L_star, [])
        for k in 1:dimensions.num_func_K
            push!(L_star[i], temp[dimensions.num_func_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Matrix{Float64}}})
    L_star::Vector{Vector{Matrix}} = []
    temp = []
    for i in 1:dimensions.num_func_I
        for k in 1:dimensions.num_func_K
            new_matrix = L[k][i]'
            push!(temp, new_matrix)
        end
    end
    for i in 1:dimensions.num_func_I
        push!(L_star, [])
        for k in 1:dimensions.num_func_K
            push!(L_star[i], temp[dimensions.num_func_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Int64}})
    L_star::Vector{Vector{Int64}} = []
    temp::Vector{Int64} = []
    for i in 1:dimensions.num_func_I
        for k in 1:dimensions.num_func_K
            push!(temp, L[k][i])
        end
    end
    for i in 1:dimensions.num_func_I
        push!(L_star, [])
        for k in 1:dimensions.num_func_K
            push!(L_star[i], temp[dimensions.num_func_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Float64}})
    L_star::Vector{Vector{Float64}} = []
    temp::Vector{Float64} = []
    for i in 1:dimensions.num_func_I
        for k in 1:dimensions.num_func_K
            push!(temp, L[k][i])
        end
    end
    for i in 1:dimensions.num_func_I
        push!(L_star, [])
        for k in 1:dimensions.num_func_K
            push!(L_star[i], temp[dimensions.num_func_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(L::Vector{Vector{Function}})
    L_star::Vector{Vector{Function}} = []
    temp = []
    for i in 1:dimensions.num_func_I
        for k in 1:dimensions.num_func_K
            new_matrix = L_star_function[k][i]
            push!(temp, new_matrix)
        end
    end
    for i in 1:dimensions.num_func_I
        push!(L_star, [])
        for k in 1:dimensions.num_func_K
            push!(L_star[i], temp[dimensions.num_func_K*(i-1) + k])
        end
    end
    return L_star
end

function rearrange(mat::Matrix)
    return mat'
end

function define_mu_beta(p::Int64, d::Int64, original_y::Vector{Float64}) 
    mu_temp::Vector{Vector{Float64}} = []
    for _ in 1:p
        random_vector = randn(Float64, d)
        rnorm = NormL2(1)(random_vector)
        random_vector = random_vector/rnorm
        push!(mu_temp, random_vector)
    end

    w_temp = []
    for i in 1:p
        if i%4==0
            push!(w_temp, -1)
        else
            push!(w_temp, 1)
        end
    end
    w = shuffle(w_temp)

    beta_temp = Float64[]
    for i in 1:p
        push!(beta_temp, w[i]*sign(dot(mu_temp[i], original_y)))
    end
    return mu_temp, beta_temp
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
    end
    return arr
end

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
    sum = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        sum = sum + weights[i]*x[i]
    end
    return sum
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
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1 * column_vec
        ans = ans + matrix_product
    end
    return vec(ans)
end

function matrix_dot_product(v::Vector{Matrix{Float64}}, u::Vector{Vector{Float64}})
    ans =  v[1]*reshape(u[1], length(u[1]), 1)
    ans*=0
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1 * column_vec
        ans = ans + matrix_product
    end
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
    return vec(ans)
end

function matrix_dot_product(v::Vector{Float64}, u::Vector{Vector{Float64}})
    ans =  v[1]*(reshape(u[1], length(u[1]), 1))
    ans*=0
    n = length(v)
    for i in 1:n
        v1 = v[i]
        u1 = u[i]
        m = length(u1)
        column_vec = reshape(u1, m, 1)
        matrix_product = v1*(column_vec)
        ans = ans + matrix_product
    end
    return vec(ans)
end

function generate_gamma_constant(i,j)
    return params.constant_g[i]
end

function generate_mu_constant(k,j)
    return params.constant_m[k]
end

function generate_gamma_seq(i,j)
    if j == 1
        return 1/params.epsilon
    else
        if vars.gamma_history[j-1][i] == params.epsilon
            return params.epsilon
        elseif ((1/params.epsilon) - 0.1*(j-1)) <= params.epsilon
            return params.epsilon
        else
            return ((1/params.epsilon) - 0.1*(j-1))
        end
    end
end

function generate_mu_seq(k,j)
    if j == 1
        return 1/params.epsilon
    else
        if vars.mu_history[j-1][k] == params.epsilon
            return params.epsilon
        elseif ((1/params.epsilon) - 0.1*(j-1)) <= params.epsilon
            return params.epsilon
        else
            return ((1/params.epsilon) - 0.1*(j-1))   # in future we can also set the subtraction constant different for different i and k using the constant arrays made in main.jl
        end
    end
end

function generate_gamma_linear_decrease(i, j)
    return max(params.gamma_end[i], (params.gamma_start[i] - params.gamma_step[i] * (j - 1)))
end

function generate_mu_linear_decrease(k, j)
    return max(params.mu_end[k], (params.mu_start[k] - params.mu_step[k] * (j - 1)))
end

function generate_gamma_random(i, j)
    return params.epsilon + ((1/params.epsilon - params.epsilon) * rand())
end

function generate_mu_random(k, j)
    return params.epsilon + ((1/params.epsilon - params.epsilon) * rand())
end

function generate_gamma_nonlinear_decrease(i, j)
    return params.gamma_a[i] + (params.gamma_b[i] * (1/j))
end

function generate_mu_nonlinear_decrease(k, j)
    return params.mu_a[k] + (params.mu_b[k] * (1/j))
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
            if vars.birthdates[1][b]<j-params.max_task_delay
                newvals=fetch(vars.running_tasks[1][b])
                task_no = vars.task_number[1][b]
                vars.a[task_no] , y= newvals 
                vars.a_star[task_no] = (res.x[j][task_no]-vars.a[task_no])./vars.gamma_history[j][task_no] - vars.l_star[task_no]
            end
        end
        for b in 1:vars.tasks_num[2]
            if vars.birthdates[2][b]<j-params.max_task_delay
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
                vars.t[task] = vars.b[task] - matrix_dot_product(get_L(L_operator, task), vars.a)
                vars.sum_k[task] = (norm_function(vars.t[task]))*2               
            else
                vars.t_star[task] = vars.a_star[task] +  matrix_dot_product(get_L(L_operator_transpose, task), vars.b_star)
                vars.sum_i[task] = (norm_function(vars.t_star[task]))*2
            end
        else
            birth = birth+1
        end
    end
end

function soft_threshold(x::Vector{Float64}, gamma::Float64)
    return sign.(x) .* max.(abs.(x) .- gamma, 0)
end

function custom_prox(t, f, y, gamma)
    sleep(t)
    a,b = prox(f,y,gamma)
    return a,b
end

function define_tasks(j)
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in params.I                  # change  - incorporated blocks into this, now running over entire params.I 
            vars.l_star[i] = matrix_dot_product(get_L(L_operator_transpose, i), res.v_star[j]) 
            delay = 0
            local task = @task custom_prox(delay,functions[i], res.x[j][i]-vars.l_star[i]*vars.gamma_history[j][i] ,vars.gamma_history[j][i])
            vars.prox_call[i] = 1
            vars.prox_call_count += 1
            add_task(task, 1, j, i)
    end

    for k in params.K 
            vars.l[k] = matrix_dot_product(get_L(L_operator, k), res.x[j])
            delay = 0
            local task = @task custom_prox(delay, functions[dimensions.num_func_I+k], vars.l[k] + vars.mu_history[j][k]*res.v_star[j][k], vars.mu_history[j][k])
            vars.prox_call[dimensions.num_func_I+k] = 1
            vars.prox_call_count += 1
            add_task(task, 2, j, k)
    end
end

function calc_theta(j)
    lambda = 1/(j-params.alpha_) + params.beta_
    tau = 0
    for i in 1:dimensions.num_func_I
        tau = tau + vars.sum_i[i] 
    end

    for k in 1:dimensions.num_func_K
        tau = tau+vars.sum_k[k]
    end

    vars.theta = 0.0

    # Calculating theta
    if tau > 0
        sum = 0
        # Finding the sum of the dot products related to the I set
        for i in 1:dimensions.num_func_I
            sum = sum+dot(res.x[j][i], vars.t_star[i])-dot(vars.a[i],vars.a_star[i])
        end
        # Finding the sum of the dot products related to the K set
        for k in 1:dimensions.num_func_K
            sum = sum+dot(vars.t[k],res.v_star[j][k])-dot(vars.b[k],vars.b_star[k])
        end
        # Using the 2 sums to find theta according to the formula
        vars.theta = lambda*max(0,sum)/tau
    end
end

function update_vars(j)
    x = res.x[j]
    for i in 1:dimensions.num_func_I
        x[i] = res.x[j][i] - vars.theta*vars.t_star[i]
    end
    push!(res.x, x)
    v_star = res.v_star[j]
    for k in 1:dimensions.num_func_K
        v_star[k] = res.v_star[j][k] - vars.theta*vars.t[k]
    end
    push!(res.v_star, v_star)
end

function update_params(j)
    # change required - make the choosing of gamma and mu random
    # check if this change needs to be done - when appending to the gamma history array, we need to input the index in generate random corresponding to params.I , 
    # right now it is dimensions.num_func_I cause there are no blocks; basically we need to generate lambda and mu for the i which belong to the block params.I  however, 
    # it doesn't harm if it is generated for all i in I
    push!(vars.gamma_history, [])
    push!(vars.mu_history, [])
    for i in 1:dimensions.num_func_I
        # if i in params.I do this
        push!(vars.gamma_history[j], generate_gamma(i,j))
        # else 
        #     push!(vars.gamma_history[j], vars.gamma_history[j-1][i])
    end
    for k in 1:dimensions.num_func_K
        push!(vars.mu_history[j], generate_mu(k,j))
    end
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
    vars.store_x[j] = Vector{Vector{Float64}}(undef, dimensions.num_func_I)
    vars.store_v[j] = Vector{Vector{Float64}}(undef, dimensions.num_func_K)
    for i in 1:dimensions.num_func_I
        vars.store_x[j][i] = res.x[j][i]
    end
    for k in 1:dimensions.num_func_K
        vars.store_v[j][k] = matrix_dot_product(get_L(L_operator, k), res.x[j])
    end
end

function check_feasibility()
    feasible = true
    for i in 1:dimensions.num_func_I
        if(functions[i](res.x[dimensions.iters][i])==Inf)
            feasible = false
        end
    end

    for k in 1:dimensions.num_func_K
        if(functions[dimensions.num_func_I + k](matrix_dot_product(get_L(L_operator, k), res.x[dimensions.iters]))==Inf)
            feasible = false
        end
    end
    return feasible
end

function get_accuracy()
    println()
    print("Final ans: ")

    x_res = []
    for i in 1:dimensions.num_func_I
        push!(x_res,res.x[dimensions.iters][i])
    end
    println(size(x_res))
    y_pred::Vector{Float64} = fill(0.0, dimensions.d)

    for j in 1:length(x_res)
        y_pred += x_res[j]
    end

    beta_res = Float64[]                                             #The predicted beta (classifications)
    corr_pred::Float64 = 0                                           #the correct predictions count

    for i in 1:dimensions.num_func_K
        push!(beta_res, sign(dot(params.mu_k[i], y_pred)))
        if beta_res[i] == params.beta_k[i]
            corr_pred+=1
        end
    end

    println("Correct predictions = ", corr_pred, "\nAccuracy = ", (corr_pred / dimensions.num_func_K))
end

function compute_epoch()
    for i in 1:dimensions.num_func_I+dimensions.num_func_K
        if vars.prox_call[i] == 0
            return false
        end
    end
    return true  
end

function record()
    if record_method == "0"
        if params.record_residual == true
            for j in 2:dimensions.iters
                temp = []
                sum = 0
                for i in 1:dimensions.num_func_I
                    push!(temp, SqrNormL2(1)(vars.store_x[j][i] - vars.store_x[j-1][i]))
                    sum+=SqrNormL2(1)(vars.store_x[j][i] - vars.store_x[j-1][i])
                end 
                push!(vars.x_residuals_avg, sum/dimensions.num_func_I)
                push!(vars.x_residuals, temp)
            end
            println("\nx_residuals_avg is ", vars.x_residuals_avg)
            println("\nx_residuals is ", vars.x_residuals)
        end
        if params.record_dist == true
            for j in 1:dimensions.iters
                push!(vars.dist_to_minima, NormL2(1)(vars.store_x[j] - vars.store_x[dimensions.iters]))
            end
            println("\ndist to minima is ", vars.dist_to_minima)
        end
        if params.record_func == true
            for j in 1:dimensions.iters
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+= (functions[i](vars.store_x[j][i]) - functions[i](vars.store_x[dimensions.iters][i]))
                end
                for k in 1:dimensions.num_func_K
                    sum+= functions[dimensions.num_func_I+k](vars.store_v[j][k]) - functions[dimensions.num_func_I+k](vars.store_v[dimensions.iters][k])            
                end
                push!(vars.f_values, abs(sum))
            end
            for j in 1:dimensions.iters
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+= (functions[i](vars.store_x[j][i]))
                end
                for k in 1:dimensions.num_func_K
                    sum+= (functions[dimensions.num_func_I+k](vars.store_v[j][k]))            
                end
                push!(vars.only_f_values, sum)
            end
            println("\nf_values is ", vars.f_values)
            println("\nonly_f_values is ", vars.only_f_values)
        end
    end
    if record_method == "1"
        if params.record_residual == true
            for j in vars.epoch_array
                if j != 1
                    temp = []
                    sum = 0
                    for i in 1:dimensions.num_func_I
                        push!(temp, SqrNormL2(1)(vars.store_x[j][i] - vars.store_x[j-1][i]))
                        sum+=SqrNormL2(1)(vars.store_x[j][i] - vars.store_x[j-1][i])
                    end
                    push!(vars.x_residuals, temp)
                    push!(vars.x_residuals_avg, sum/dimensions.num_func_I)
                end
            end
            println("\nx_residuals_avg is ", vars.x_residuals_avg)
            println("\nx_residuals is ", vars.x_residuals)
        end
        if params.record_dist == true
            for j in vars.epoch_array
                push!(vars.dist_to_minima, NormL2(1)(vars.store_x[j] - vars.store_x[dimensions.iters]))
            end
            println("\ndist to minima is ", vars.dist_to_minima)
        end
        if params.record_func == true
            for j in vars.epoch_array
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+= (functions[i](vars.store_x[j][i]) - functions[i](vars.store_x[dimensions.iters][i]))
                end
                for k in 1:dimensions.num_func_K
                    sum+= functions[dimensions.num_func_I+k](vars.store_v[j][k]) - functions[dimensions.num_func_I+k](vars.store_v[dimensions.iters][k])            
                end
                push!(vars.f_values, abs(sum))
            end
            for j in vars.epoch_array
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+= functions[i](vars.store_x[j][i])
                end
                for k in 1:dimensions.num_func_K
                    sum+= functions[dimensions.num_func_I+k](vars.store_v[j][k])         
                end
                push!(vars.only_f_values, sum)
            end
            println("\nf_values is ", vars.f_values)
            println("\nonly_f_values is ", vars.only_f_values)
        end
    end
    if record_method == "2"
        if params.record_residual == true
            for j in 1:dimensions.iters
                temp = []
                if j == 1
                    for i in 1:dimensions.num_func_I
                        push!(temp, 0.0)
                    end
                else
                    for i in 1:dimensions.num_func_I
                        push!(temp, SqrNormL2(1)(vars.store_x[j][i] - vars.store_x[j-1][i]))
                    end
                end
                push!(vars.x_residuals, temp)
            end
            println("\nx_residuals_avg is ", vars.x_residuals_avg)
            println("\nx_residuals is ", vars.x_residuals)
        end
        if params.record_dist == true
            for j in 1:dimensions.iters
                push!(vars.dist_to_minima, NormL2(1)(vars.store_x[j] - vars.store_x[dimensions.iters]))
            end
            println("\ndist to minima is ", vars.dist_to_minima)
        end
        if params.record_func == true
            for j in 1:dimensions.iters
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+= (functions[i](vars.store_x[j][i]) - functions[i](vars.store_x[dimensions.iters][i]))
                end
                for k in 1:dimensions.num_func_K
                    sum+= functions[dimensions.num_func_I+k](vars.store_v[j][k]) - functions[dimensions.num_func_I+k](vars.store_v[dimensions.iters][k])            
                end
                push!(vars.f_values, abs(sum))
            end
            for j in 1:dimensions.iters
                sum = 0
                for i in 1:dimensions.num_func_I
                    sum+=(functions[i](vars.store_x[j][i]))
                end
                for k in 1:dimensions.num_func_K
                    sum+=(functions[dimensions.num_func_I+k](vars.store_v[j][k]))            
                end
                push!(vars.only_f_values, sum)
            end
            println("\nf_values is ", vars.f_values)
            println("\nonly_f_values is ", vars.only_f_values)
        end
    end
end