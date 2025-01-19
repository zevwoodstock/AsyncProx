function imageToVector(image::Array{T, 2}) where {T}
    rows, columns = size(image)
    vector = vec(image)
    return vector
end

function vectorToImage(rows::Int64, columns::Int64, vector::Vector{T}) where {T}
    image = reshape(vector, rows, columns)
    return image
end

function get_row_column(img_path)
    img = load(img_path)
    row,column = size(img)
    return (row,column)
end

function image_to_vector(img_path)
    img = load(img_path)
    gray_img = Gray.(img)
    height, width = size(gray_img)
    gray_matrix = Matrix{Float64}(undef, height, width)
    for y in 1:height, x in 1:width
        gray_matrix[y, x] = float(gray_img[y, x])
    end
    return gray_matrix
end

function masking_left(x1::Vector{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x = vectorToImage(row,column,x1)
    iter = rand(1:row)
    global row_array_left = []
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = rand(50:90)
        push!(row_array_left,row_i )
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_right(x1::Vector{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x = vectorToImage(row,column,x1)
    iter = rand(1:row)
    global row_array_right = []
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = rand(140:180)
        push!(row_array_right, row_i)
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_r(x1::Vector{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x = vectorToImage(row,column,x1)
    iter = length(row_array_right)
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_right[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_r(x1::Matrix{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x2 = vec(x1)
    x = vectorToImage(row,column, x2)
    iter = length(row_array_right )
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_right[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_l(x1::Vector{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x = vectorToImage(row,column,x1)
    iter = length(row_array_left)
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_left[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_l(x1::Matrix{Float64})
    row, column = get_row_column(params.img_path_array[1])
    x2 = vec(x1)
    x = vectorToImage(row,column, x2)
    iter = length(row_array_left )
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_left[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function matrix_to_image(X::Matrix{Float64})
    gray_image = Gray.(X)
    image = Gray.(clamp.(gray_image, 0, 1))  # Ensure pixel values are between 0 and 1
    return image
end

function blur(img::Vector{Float64})
    row, column = get_row_column(params.img_path_array[1])
    img = vectorToImage(row, column, img)
    kernel = reflect(Kernel.gaussian((params.sigma_blur,params.sigma_blur), (5,5)))
    img_blurred = imfilter(img, kernel, Fill(0))
    return imageToVector(img_blurred);  
end

function blur(img::Matrix{Float64})
    row, column = get_row_column(params.img_path_array[1])
    img = vec(img)
    img = vectorToImage(row, column, img)
    kernel = reflect(Kernel.gaussian((params.sigma_blur, params.sigma_blur), (5,5)))
    img_blurred = imfilter(img, kernel, Fill(0))
    # img_blurred = imfilter(img, Kernel.gaussian((3, 3), 0.5))
    # img_blurred = imfilter(img, Kernel.gaussian(3, 3, 0.5))
    return imageToVector(img_blurred);  
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

# function generate_gamma_seq(i,j)
#     if j == 1
#         return start_value_gamma
#     else
#         if vars.gamma_history[j-1][i] == end_value_gamma
#             return end_value_gamma
#         elseif (vars.gamma_history[j-1][i] - rate_decrease_gamma) <= end_value_gamma
#             return end_value_gamma
#         else
#             return vars.gamma_history[j-1][i] - rate_decrease_gamma
#         end
#     end
# end

# function generate_mu_seq(k,j)
#     if j == 1
#         return start_value_mu
#     else
#         if vars.mu_history[j-1][k] == end_value_mu
#             return end_value_mu
#         elseif (vars.mu_history[j-1][k] - rate_decrease_mu) <= end_value_mu
#             return end_value_mu
#         else
#             return vars.mu_history[j-1][k] - rate_decrease_mu   # in future we can also set the subtraction constant different for different i and k using the constant arrays made in main.jl
#         end
#     end
# end

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
            if vars.birthdates[1][b]<j-params.max_iter_delay
                task_done = istaskdone(vars.running_tasks[1][b])
                println("Task done: ", task_done)
                newvals=fetch(vars.running_tasks[1][b])
                task_no = vars.task_number[1][b]
                vars.a[task_no] , y= newvals 
                vars.a_star[task_no] = (res.x[j][task_no]-vars.a[task_no])./vars.gamma_history[j][task_no] - vars.l_star[task_no]
            end
        end
        for b in 1:vars.tasks_num[2]
            if vars.birthdates[2][b]<j-params.max_iter_delay
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

function constant_vector(N, c)
    return fill(c, N)
end

function phi(x)
    y = Wavelets.dwt(x, wavelet(WT.sym4))
    n = length(y)
    for i in 1:n
        y[i] = abs(y[i])
    end
    row, column = get_row_column(params.img_path_array[1])
    N = row * column
    mu_array = constant_vector(N, 0.1)
    linear_op = Linear(mu_array)
    result = linear_op(y)
    return result
end

function generate_random_vector(N, sigma = 0.01)
    row, column = get_row_column(params.img_path_array[1])
    N = row * column
    rng = MersenneTwister(1234)  # Set a random number generator seed for reproducibility
    mu = 0.0  # Mean of the normal distribution (default: 0.0)
    sigma_squared = params.sigma^2  # Variance of the normal distribution
    random_vector = sqrt(sigma_squared) * randn(rng, N) .+ mu
    return random_vector
end

function shift_image_left(image::Matrix{Float64}, x::Int64)
    height, width = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i - x - 1, width) + 1
            shifted_image[y, i] = image[y, j]
        end
    end
    return -shifted_image
end

function define_D(shift::Int64)
    row, column = get_row_column(params.img_path_array[1])
    return x -> imageToVector(shift_image_left(vectorToImage(row, column,x),shift))
end

function define_D(shift::Int64)
    row, column = get_row_column(params.img_path_array[1])
    return x -> imageToVector(shift_image_left(vectorToImage(row, column,vec(x)),shift))
end

function shift_image_right(image::Matrix{Float64}, x::Int)
    height, width = size(image)
    shifted_image = similar(image)

    for y = 1:height
        for i = 1:width
            j = mod(i + x + 1, width) + 1
            shifted_image[y, i] = image[y, j]
        end
    end

    return -shifted_image
end

function define_D_star(shift::Int64)
    row, column = get_row_column(params.img_path_array[1])
    return x -> imageToVector(shift_image_right(vectorToImage(row, column, x),shift))
end

function define_D_star(shift::Int64)
    row, column = get_row_column(params.img_path_array[1])
    return x -> imageToVector(shift_image_right(vectorToImage(row, column, vec(x)),shift))
end

identity_function(x) = x

function null_func(input_vector)
    return zeros(eltype(input_vector), length(input_vector))
end

function generate_L(m::Int, degradation::Function)
    L_func = Vector{Vector{Function}}()
    for i = 1:m
        row = Vector{Function}()
        for j = 1:m
            if (j == i)
                push!(row, degradation)
            else
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    for i = 1:m-1
        row = Vector{Function}()
        for j = 1:m
            if j == i 
                push!(row, identity_function)
            elseif j == i+1
                push!(row, define_D(params.left_shift_pixels[i]))
            else 
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    return L_func
end

function generate_L_star(m::Int, degradation::Function)
    L_func = []
    for i = 1:m
        row = []
        for j = 1:m
            if (j == i)
                push!(row, degradation)
            else
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    for i = 1:m-1
        row = []
        for j = 1:m
            if j == i 
                push!(row, identity_function)
            elseif j == i+1
                push!(row, define_D_star(params.right_shift_pixels[i]))
            else 
                push!(row, null_func)
            end
        end
        push!(L_func, row)
    end
    return L_func
end

function custom_prox(t, f, y, gamma)
    sleep(t)
    row, column = get_row_column(params.img_path_array[1])
    N = row * column
    mu_array = constant_vector(N,0.1)
    if f == phi
        dwt = Wavelets.dwt(y, wavelet(WT.sym4))
		#This step assumes that mu_array is constant. mu_array is
		#the constant of coefficients of the l1 norm in the imaging
		#problem. Increasing that coefficient effectively increases
		#the parameter of the prox.
		st = soft_threshold(dwt, gamma*mu_array[1])
        # st = soft_threshold(dwt, 1.0)
        idwt = Wavelets.idwt(st, wavelet(WT.sym4))
        return idwt, phi(idwt)
    end
    a,b = prox(f,y,gamma)
    return a,b
end

function define_tasks(j)    
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in params.I               
        vars.l_star[i] = matrix_dot_product(get_L(L_operator_transpose, i), res.v_star[j]) 
        delay = 0
        local task = @task custom_prox(delay,functions[i], res.x[j][i] - vars.l_star[i]*vars.gamma_history[j][i], vars.gamma_history[j][i])
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
    # wait(conditions[ind][i])
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

function compute_epoch()
    for i in 1:dimensions.num_func_I+dimensions.num_func_K
        if vars.prox_call[i] == 0
                return false
            end
        end
        return true
end

function save_images()
    row, column = get_row_column(params.img_path_array[1])
    x_res = []
    for i in 1:dimensions.num_func_I
        push!(x_res,res.x[dimensions.iters][i])
    end

    ret_images = []
    for i in 1:dimensions.num_func_I
        push!(ret_images,matrix_to_image(vectorToImage(row, column, x_res[i])))
    end

    ret_path = []
    for i in 1:dimensions.num_func_I
        println("saving the recovered images")
        base_path = dirname(params.img_path_array[i])
        push!(ret_path, joinpath(base_path, "ret_$i.jpeg"))
        save(ret_path[i],ret_images[i])
    end
end

function record()
    if params.record_method == 0
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
            # print(vars.store_x[dimensions.iters])
            println("\ndist to minima is ", vars.dist_to_minima)
            mn1 = Inf
            for i in 1:Int64(dimensions.iters/2)
                mn1 = min(mn1, vars.dist_to_minima[i])
            end
            mn2 = mn1
            for i in 1:Int64(dimensions.iters)
                mn2 = min(mn2, vars.dist_to_minima[i])
            end
            println("||x_mn - xinf||^2 / ||x0-finf||^2 = ", SqrNormL2(1)(mn1/vars.dist_to_minima[1]))
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
            
            println("\nf_values is ", vars.f_values) #stores distance from final value 
            println("\nonly_f_values is ", vars.only_f_values) #stores actual function value
            mn = Inf
            for i in 1:Int64(dimensions.iters/2)
                mn = min(mn, vars.f_values[i])
            end
            mn2 = Inf
            for i in 1:Int64(dimensions.iters)
                mn2 = min(mn2, vars.f_values[i])
            end
            println("||f_mn - finf||^2 / ||f0-finf||^2 = ", SqrNormL2(1)((mn - mn2)/(vars.only_f_values[1] - mn2)))
        end
    end
    if params.record_method == 1
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
    if params.record_method == 2
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

function print_params()
    println()
    println("-------------------------------------------------------------------")
    println("Parameters used were:")
    println("L_function_bool = ", L_function_bool)
    println("num_func_I = ", dimensions.num_func_I)
    println("d = ", dimensions.d)
    println("num_func_K = ", dimensions.num_func_K)
    println("iters = ", dimensions.iters)
    println("max_iter_delay = ", params.max_iter_delay)
    println("alpha = ", params.alpha_)
    println("beta = ", params.beta_)
    println("compute_epoch_bool = ", params.compute_epoch_bool)
    println("record_residual = ", params.record_residual)
    println("record_func = ", params.record_func)
    println("record_dist = ", params.record_dist)
    println("record_method = ", params.record_method)
end
