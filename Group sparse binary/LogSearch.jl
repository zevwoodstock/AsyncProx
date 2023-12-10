global v_gamma = 10
global v_mu = 20

arr = []

for ii in -1:4
    for j in -1:4
        global v_gamma = 10.0^ii
        global v_mu = 10.0^j
        include("main.jl")
        fvalue = 0
        for i in 1:functions_I
            println(i)
            println(functions[i])
            fvalue+= functions[i](final_ans[i])
        end
        for k in 1:functions_K
            fvalue+= functions[functions_I+k](matrix_dot_product(get_L(L, k), final_ans))         
        end
        push!(arr, [fvalue, [v_gamma, v_mu]])
    end
end
println(arr)
println(sort(arr))