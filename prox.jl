using LinearAlgebra
using ProximalOperators
using SparseArrays

x = [[1.0,1.0]]
v_star = [[5.0, 9.0], [3.0,8.0]]
centers = [[2,4], [0,4], [1,5]]

global norm_function = NormL2(1)
global gamma = [1]
global mu = [1,1]
global a_star = [x[1]]
global a = x
global t_star = [x[1]]
global b = [x[1], x[1]]
global l = [x[1], x[1]]
global l_star = [v_star[1]]
global K = 2
global I = 1
iters = 50000

function sum_array(x)
    global s = zeros(size(x[1], 1))
    for i in 1:size(x, 1)
        global s = s+x[i]
    end
    return s
end

global mode = "w"
for j in 1:iters
    global lambda = 1/j + 0.5

    sum_k = 0
    for i in 1:I
        global l_star[i] = sum_array(v_star)
        local f = Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[i])
        global a[i], y = prox(f, x[i]-l_star[i]*gamma[i] ,gamma[i])
        global a_star[i] = (x[i]-a[i])./gamma[i] - l_star[i]
        global t_star[i] = a_star[i] + sum_array(b_star)
        sum_k = sum_k+(norm_function(t_star[i]))^2
    end
    
    sum_i = 0
    for k in 1:K
        l[k] = sum_array(x)
        local f = Precompose(IndBallL2(1.0), Matrix(LinearAlgebra.I, 2,2), 1, -centers[k+I])
        global b[k], y = prox(f, l[k] + mu[k]*v_star[k], mu[k])
        b_star[k] = v_star[k] + (l[k]-b[k])./mu[k]
        t[k] = b[k] - sum_array(a)
        sum_i = sum_i+(norm_function(t[k]))^2
    end

    tau =  sum_i + sum_k
    theta = 0

    if tau > 0
        sum_i = 0
        sum_k = 0
        for i in 1:I
            sum_i = sum_i+dot(x[i], t_star[i])-dot(a[i],a_star[i])
        end
        for k in 1:K
            sum_k = sum_k+dot(t[k],v_star[k])-dot(b[k],b_star[k])
        end
        theta = lambda*max(0,sum_i+sum_k)/tau
    end

    if j>1
        global mode = "a"
    end
    open("x.txt",mode) do io
        println(io,x)
    end

    for i in 1:I
        global x[i] = x[i] - theta*t_star[1]
    end
    for k in 1:K
        global v_star[k] = v_star[k] - theta*t[k]
    end
    

end

open("x.txt","a") do io
    println(io,x)
end
println(x)


#plot graphs
#grid search
#bayesian 
#hyperparameter optimisation
#asycnhronous, after a certain limit, terminate the process