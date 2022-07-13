using LinearAlgebra
using ProximalOperators

#intialising our problem
global I = 2
global K = 3
global dims = 2
global gamma = [1,1]
global mu = [1,1,1]
centers = [[0,1], [0,1.1], [1,1], [1,1.1]]

L = [[1,0], [0,1], [1,-1]]

functions = []
for i in 1:I+K-1
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, dims,dims), 1, -centers[i])])
end

#append!(functions, [Linear([1,0])])
append!(functions, [IndBallL2(0.001)])


zeros_I = []
zeros_K = []
global sum_vector_i = []
global sum_vector_k = []
for i in 1:I
    append!(zeros_I, [zeros(dims)])
    append!(sum_vector_i, [0.0])
end
for i in 1:K
    append!(zeros_K, [zeros(dims)])
    append!(sum_vector_k, [0.0])
end

x_history = [zeros_I]
v_history = [zeros_K]

struct variables                  #these are not really hyperparameters, but it makes the code look mmore organised~
    a::Vector{Vector{Float64}}
    a_star::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
    b_star::Vector{Vector{Float64}}
    t::Vector{Vector{Float64}}
    t_star::Vector{Vector{Float64}}
    l::Vector{Vector{Float64}}
    l_star::Vector{Vector{Float64}}
    iters:: Int32
end


#initialising all the global variables/hyperparameters
vars = variables(zeros_I, zeros_I, zeros_K, zeros_K,           #gamma, mu, a, a star, b and b star
                        zeros_K, zeros_I, zeros_K, zeros_I,40000)          #t and t star, l, l_star, iters

#a function to find the L2 norm of a vector                       
global norm_function = SqrNormL2(1)

#a function to find the weighted sum of all elements of an array like, w[1]v[1] + w[2]v[2] + .....

function weighted_sum_array(weights,x)                   
    global s = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        global s = s+weights[i]*x[i]
    end
    return s
end

function transpose(L)
    L_star = []
    temp = []
    for i in 1:I
        for k in 1:K
            append!(temp, [L[k][i]])
        end
    end
    for i in 1:I
        append!(L_star, [[]])
        for k in 1:K
            append!(L_star[i], [temp[K*(i-1) + k]])
        end
    end
    return L_star
end


global mode = "w" #to write to file



#Variables for asynchronous implementation
running_tasks_i = []
birthdates_i = []
task_number_i = []
global tasks_num_i = 0

running_tasks_k = []
birthdates_k = []
task_number_k = []
global tasks_num_k = 0


#the main loop starts here
for j in 1:vars.iters
    global lambda = 1/(j-0.5) + 0.5
    #the learning rate decreases with iteration number


    #Checking if a task has been delayed for too long
    if j>1
        D = 50
        for b in 1:tasks_num_i
            if birthdates_i[b]<j-D
                newvals=fetch(running_tasks_i[b])
            end
        end
        for b in 1:tasks_num_k
            if birthdates_k[b]<j-D
                newvals=fetch(running_tasks_k[b])
            end
        end
    end

    #the loop running through I
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in 1:I
        vars.l_star[i] = weighted_sum_array(transpose(L)[i], v_history[j])
        local task = @task prox(functions[i], x_history[j][i]-vars.l_star[i]*gamma[i] ,gamma[i])
        schedule(task)
        append!(running_tasks_i, [task])
        append!(birthdates_i, [j])
        append!(task_number_i, [i])
        global tasks_num_i = tasks_num_i + 1
    end

    #for iterations after the first
    b = 1
    while b<= tasks_num_i
        if istaskdone(running_tasks_i[b]) == true
            i = task_number_i[b]
            vars.a[i], y = fetch(running_tasks_i[b])
            vars.a_star[i] = (x_history[j][i]-vars.a[i])./gamma[i] - vars.l_star[i]
            vars.t_star[i] = vars.a_star[i] + weighted_sum_array(transpose(L)[i], vars.b_star)
            global sum_vector_i[i] = (norm_function(vars.t_star[i]))*2
            deleteat!(running_tasks_i, b)
            deleteat!(birthdates_i, b)
            deleteat!(task_number_i, b)
            global tasks_num_i = tasks_num_i - 1
        else
            b = b+1
        end
    end


    for k in 1:K
        vars.l[k] = weighted_sum_array(L[k], x_history[j])
        local task = @task prox(functions[I+k], vars.l[k] + mu[k]*v_history[j][k], mu[k])
        schedule(task)
        append!(running_tasks_k, [task])
        append!(birthdates_k, [j])
        append!(task_number_k, [k])
        global tasks_num_k = tasks_num_k + 1
    end

    #for iterations after the first
    b = 1
    while b<= tasks_num_k
        if istaskdone(running_tasks_k[b]) == true
            k = task_number_k[b]
            vars.b[k],y = fetch(running_tasks_k[b])
            vars.b_star[k] = v_history[j][k] + (vars.l[k]-vars.b[k])./mu[k]
            vars.t[k] = vars.b[k] - weighted_sum_array(L[k], vars.a)
            global sum_vector_k[k] = (norm_function(vars.t[k]))*2
            deleteat!(running_tasks_k, b)
            deleteat!(birthdates_k, b)
            deleteat!(task_number_k, b)
            global tasks_num_k = tasks_num_k - 1
        else
            b = b+1
        end
    end

    tau = 0
    for i in 1:I
        tau = tau + sum_vector_i[i] 
    end

    #the loop running through K
    for k in 1:K
        tau = tau+sum_vector_k[k]
    end

    theta = 0

    #finding theta
    if tau > 0
        sum = 0
        #finding the sum of the dot products related to the I set
        for i in 1:I
            sum = sum+dot(x_history[j][i], vars.t_star[i])-dot(vars.a[i],vars.a_star[i])
        end
        #finding the sum of the dot products related to the K set
        for k in 1:K
            sum = sum+dot(vars.t[k],v_history[j][k])-dot(vars.b[k],vars.b_star[k])
        end
        #using the 2 sums to find theta according to the formula
        theta = lambda*max(0,sum)/tau
    end

    #deciding to write to file or append
    if j>1
        global mode = "a"
    end
    open("x.txt",mode) do io
        println(io,x)
    end

    #updating our variables
    for i in 1:I
        global x[i] = x_history[j][i] - theta*vars.t_star[i]
    end
    append!(x_history, [x])
    for k in 1:K
        global v_star[k] = v_history[j][k] - theta*vars.t[k]
    end
    append!(v_history, [v_star])
    

end

#write to file and store the last x
open("x.txt","a") do io
    println(io,x)
end
println(x_history[vars.iters])


#things left to do 
#minibatches
#L as a function or vector
#threads