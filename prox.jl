using LinearAlgebra
using ProximalOperators
using SparseArrays


"""The problem currently has been tested on the intersection of 4 balls - 
centred at (1,5), (0,4), (2,4), (1,3) so clearly, their point of intersection is (1,4)."""

"""The next function with non zero slope I am trying to incorporate is the Linear function <c|x>. 
I am having trouble generlaising this to higher dimensions and visualising what it will be like"""

#intialising our problem
global I = 1
global K = 3
global dims = 2
centers = [[2,4], [0,4], [1,5], [1,3]]

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


struct hyperparameters                  #these are not really hyperparameters, but it makes the code look mmore organised
    gamma:: Vector{Float64}
    mu::Vector{Float64}
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
params = hyperparameters([1], [1,1,1], zeros_I, zeros_I, zeros_K, zeros_K,           #gamma, mu, a, a star, b and b star
                        zeros_K, zeros_I, zeros_K, zeros_I, 200000)                   #t and t star, l, l_star, iters

#a function to find the L2 norm of a vector                       
global norm_function = SqrNormL2(1)


#a function to find the sum of all elements of an array like, v[1] + v[2] + .....
function sum_array(x)                   
    global s = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        global s = s+x[i]
    end
    return s
end



global mode = "w" #to write to file


#the main loop starts here
for j in 1:params.iters
    global lambda = 1/j + 0.5
    #the learning rate decreases with iteration number

    sum_k = 0
    #the loop running through I
    for i in 1:I
        params.l_star[i] = sum_array(v_star)
        local f = Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i])
        params.a[i], y = prox(f, x[i]-params.l_star[i]*params.gamma[i] ,params.gamma[i])
        params.a_star[i] = (x[i]-params.a[i])./params.gamma[i] - params.l_star[i]
        global params.t_star[i] = params.a_star[i] + sum_array(params.b_star)
        sum_k = sum_k+(norm_function(params.t_star[i]))*2
    end
    
    sum_i = 0
    #the loop running through K
    for k in 1:K
        if((j+k)%2==0)
            params.l[k] = sum_array(x)
            if(k==K)
                local f = Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[k+I])
                params.b[k],y = prox(f, params.l[k] + params.mu[k]*v_star[k], params.mu[k])
            else
                local f = Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[k+I])
                params.b[k],y = prox(f, params.l[k] + params.mu[k]*v_star[k], params.mu[k])
            end
            params.b_star[k] = v_star[k] + (params.l[k]-params.b[k])./params.mu[k]
            params.t[k] = params.b[k] - sum_array(params.a)
            sum_i = sum_i+(norm_function(params.t[k]))*2
        end
    end

    tau =  sum_i + sum_k
    theta = 0

    #finding theta
    if tau > 0
        sum_i = 0
        sum_k = 0
        #finding the sum of the dot products related to the I set
        for i in 1:I
            sum_i = sum_i+dot(x[i], params.t_star[i])-dot(params.a[i],params.a_star[i])
        end
        #finding the sum of the dot products related to the K set
        for k in 1:K
            sum_k = sum_k+dot(params.t[k],v_star[k])-dot(params.b[k],params.b_star[k])
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
        global x[i] = x[i] - theta*params.t_star[1]
    end
    for k in 1:K
        global v_star[k] = v_star[k] - theta*params.t[k]
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