global v_gamma = 10
global v_mu = 20

arr = []

global v_gamma = 1

eps = 0.01
r = 10
l = 0.1
while r - l > eps
    global v_mu = l + (r-l)/3
    include("main.jl")
    fvaluel = 0
    fvaluer = 0
    for i in 1:functions_I
        fvaluel+= functions[i](final_ans[i])
    end
    for k in 1:functions_K
        fvaluel+= functions[functions_I+k](matrix_dot_product(get_L(L, k), final_ans))         
    end
    global v_mu = r - (r-l)/3
    include("main.jl")
    for i in 1:functions_I
        fvaluer+= functions[i](final_ans[i])
    end
    for k in 1:functions_K
        fvaluer+= functions[functions_I+k](matrix_dot_product(get_L(L, k), final_ans))         
    end
    push!(arr, [fvaluel, l + (r-l)/3])
    push!(arr, [fvaluer, r - (r-l)/3])
    println([fvaluel, l + (r-l)/3])
    println([fvaluer, r - (r-l)/3])
    global v_mu = r - (r-l)/3
    if fvaluel > fvaluer
        global l = l + (r-l)/3
    else
        global r = r - (r-l)/3
    end
end
println("best value is at mu = ", l)
println(arr)
println(sort(arr))