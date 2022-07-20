using LinearAlgebra
using ProximalOperators
using SparseArrays



#intialising our problem
global I = 2
global K = 4
global dims = 2
global gamma = [1,1]
global mu = [1,1,1,1]
centers = [[0,1], [0,1], [1,1], [1,1]]

L = [[1,0], [0,1], [1,0], [0,1]]

functions = []
for i in 1:I+K-2
    append!(functions, [Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, dims,dims), 1, -centers[i])])
end

append!(functions, [Linear([-1,0])])
append!(functions, [Linear([0,1])])

zeros_I = []
zeros_K = []
for i in 1:I
    append!(zeros_I, [zeros(dims)])
end
for i in 1:K
    append!(zeros_K, [zeros(dims)])
end

x = zeros_I
v_star = zeros_K

global temp_i = [0.0,0.0]

struct variables                  #these are not really hyperparameters, but it makes the code look more organised~
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
                        zeros_K, zeros_I, zeros_K, zeros_I,4000)          #t and t star, l, l_star, iters

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


#the main loop starts here
for j in 1:vars.iters
    global lambda = 1/(j-0.5) + 0.5
    #the learning rate decreases with iteration number

    sum_k = 0
    #the loop running through I
    for i in 1:I
        vars.l_star[i] = weighted_sum_array(transpose(L)[i], v_star)
        vars.a[i], y = prox(functions[i], x[i]-vars.l_star[i]*gamma[i] ,gamma[i])
        vars.a_star[i] = (x[i]-vars.a[i])./gamma[i] - vars.l_star[i]
        vars.t_star[i] = vars.a_star[i] + weighted_sum_array(transpose(L)[i], vars.b_star)
        global temp_i[i] = (norm_function(vars.t_star[i]))*2
    end

    for i in 1:I
       sum_k = sum_k + temp_i[i] 
    end
    
    sum_i = 0
    #the loop running through K
    for k in 1:K
        vars.l[k] = weighted_sum_array(L[k], x)
        vars.b[k],y = prox(functions[I+k], vars.l[k] + mu[k]*v_star[k], mu[k])
        vars.b_star[k] = v_star[k] + (vars.l[k]-vars.b[k])./mu[k]
        vars.t[k] = vars.b[k] - weighted_sum_array(L[k], vars.a)
        sum_i = sum_i+(norm_function(vars.t[k]))*2
    end

    tau =  sum_i + sum_k
    theta = 0

    #finding theta
    if tau > 0
        sum_i = 0
        sum_k = 0
        #finding the sum of the dot products related to the I set
        for i in 1:I
            sum_i = sum_i+dot(x[i], vars.t_star[i])-dot(vars.a[i],vars.a_star[i])
        end
        #finding the sum of the dot products related to the K set
        for k in 1:K
            sum_k = sum_k+dot(vars.t[k],v_star[k])-dot(vars.b[k],vars.b_star[k])
        end
        #using the 2 sums to find theta according to the formula
        theta = lambda*max(0,sum_i+sum_k)/tau
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
        global x[i] = x[i] - theta*vars.t_star[i]
    end
    for k in 1:K
        global v_star[k] = v_star[k] - theta*vars.t[k]
    end
    
    if j==1
        println(x)
    end

end

#write to file and store the last x
open("x.txt","a") do io
    println(io,x)
end
println(x)


#grid search
#bayesian 
#hyperparameter optimisation
#asycnhronous, after a certain limit, terminate the process