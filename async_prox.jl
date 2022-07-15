using LinearAlgebra
using ProximalOperators
using Random


#inputs
#intialising our problem
global I = 2
global K = 3
global dims = 2
global gamma = [1,1]
global mu = [1,1,1]
centers = [[0,1], [0,1.1], [1,1], [1,1.1]]

identity = [1 0;0 1]
null_mat = [0 0;0 0]

L_matrix = [[identity, null_mat], [null_mat, identity], [identity, -identity]]

functions = []
for i in 1:I+K-1
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, dims,dims), 1, -centers[i])])
end

#append!(functions, [Linear([1,0])])
append!(functions, [IndBallL2(0.0001)])




#main program


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

#initialing an array to store the values of our main variables at every iteration
x_history = [zeros_I]
v_history = [zeros_K]

#Variables for asynchronous implementation
running_tasks = [[],[]]
birthdates = [[],[]]
task_number = [[],[]]
global tasks_num = [0,0]
minibatches = [[],[]]

global a = zeros_I

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
                        zeros_K, zeros_I, zeros_K, zeros_I,150000)          #t and t star, l, l_star, iters

#a function to find the L2 norm of a vector                       
global norm_function = SqrNormL2(1)

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

function matrix_sum(matrix_array, x)
    n = length(x)
    sum = [0;0]
    for i in 1:n
        temp = x[i]
        #each temp corresponds to each x_history
        #each matrix_array corresponds to vector of matrices for a vector of all Xs
        reshape(temp, 1, length(temp))
        sum = sum + matrix_array[i]*temp
    end
    return sum
end

#generate random seeds each time



random_bitvector = bitrand(MersenneTwister(0), I)
global empty_flag = false
for i in 1:I
    if random_bitvector[i]==1
        global empty_flag = true
    end
end

if empty_flag==false
    random_bitvector[1] = 1
end

complement_bitvector = []
for i in 1:I
    append!(complement_bitvector, [abs(1-random_bitvector[i])])
end
minibatches[1] = [random_bitvector, complement_bitvector]

random_bitvector = bitrand(MersenneTwister(1), K)
complement_bitvector = []
for k in 1:K
    append!(complement_bitvector, [abs(1-random_bitvector[k])])
end
minibatches[2] = [random_bitvector, complement_bitvector]


#the main loop starts here
for j in 1:vars.iters
    global lambda = 1/(j-0.5) + 0.5
    #the learning rate decreases with iteration number


    #Checking if a task has been delayed for too long
    if j>1
        D = 10
        for b in 1:tasks_num[1]
            if birthdates[1][b]<j-D
                newvals=fetch(running_tasks[1][b])
            end
        end
        for b in 1:tasks_num[2]
            if birthdates[2][b]<j-D
                newvals=fetch(running_tasks[2][b])
            end
        end
    end

    #the loop running through I
    #schedule a new task in each iteration for each i in I, and append it to the running tasks vector
    for i in 1:I
        if (minibatches[1][j%2+1][i]==1) || (j==1) 
            vars.l_star[i] = matrix_sum(transpose(L_matrix)[i], v_history[j])
            local task = @task prox(functions[i], x_history[j][i]-vars.l_star[i]*gamma[i] ,gamma[i])
            schedule(task)
            append!(running_tasks[1], [task])
            append!(birthdates[1], [j])
            append!(task_number[1], [i])
            global tasks_num[1] = tasks_num[1] + 1
        end
    end

    #for iterations after the first
    global birth = 1
    while birth<= tasks_num[1]
        if istaskdone(running_tasks[1][birth]) == true
            i = task_number[1][birth]
            if (minibatches[1][j%2+1][i]==1) || (j==1) 
                vars.a[i], y = fetch(running_tasks[1][birth])
                vars.a_star[i] = (x_history[j][i]-vars.a[i])./gamma[i] - vars.l_star[i]
                deleteat!(running_tasks[1], birth)
                deleteat!(birthdates[1], birth)
                deleteat!(task_number[1], birth)
                global tasks_num[1] = tasks_num[1] - 1
            else
                global birth = birth+1
            end
            vars.t_star[i] = vars.a_star[i] + matrix_sum(transpose(L_matrix)[i], vars.b_star)
            global sum_vector_i[i] = (norm_function(vars.t_star[i]))*2
        else
            global birth = birth+1
        end
    end


    for k in 1:K
        if (minibatches[2][j%2+1][k]==1) || (j==1) 
            vars.l[k] = matrix_sum(L_matrix[k], x_history[j])
            local task = @task prox(functions[I+k], vars.l[k] + mu[k]*v_history[j][k], mu[k])
            schedule(task)
            append!(running_tasks[2], [task])
            append!(birthdates[2], [j])
            append!(task_number[2], [k])
            global tasks_num[2] = tasks_num[2] + 1
        end
    end

    #for iterations after the first
    global b = 1
    while b<= tasks_num[2]
        if istaskdone(running_tasks[2][b]) == true
            k = task_number[2][b]
            if (minibatches[2][j%2+1][k]==1) || (j==1)
                vars.b[k],y = fetch(running_tasks[2][b])
                vars.b_star[k] = v_history[j][k] + (vars.l[k]-vars.b[k])./mu[k]
                deleteat!(running_tasks[2], b)
                deleteat!(birthdates[2], b)
                deleteat!(task_number[2], b)
                global tasks_num[2] = tasks_num[2] - 1
            else
                global b = b+1
            end
            vars.t[k] = vars.b[k] - matrix_sum(L_matrix[k],vars.a)
            global sum_vector_k[k] = (norm_function(vars.t[k]))*2
        else
            global b = b+1
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

    #updating our variables
    global x = x_history[j]
    for i in 1:I
        global x[i] = x_history[j][i] - theta*vars.t_star[i]
    end
    append!(x_history, [x])
    global v_star = v_history[j]
    for k in 1:K
        global v_star[k] = v_history[j][k] - theta*vars.t[k]
    end
    append!(v_history, [v_star])
    

end

println(x_history[vars.iters])


#things left to do 
#L as a function
#threads
#mu and gamma