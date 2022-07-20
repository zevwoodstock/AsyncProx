struct variables
    a::Vector{Vector{Float64}}
    a_star::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
    b_star::Vector{Vector{Float64}}
    t::Vector{Vector{Float64}}
    t_star::Vector{Vector{Float64}}
    l::Vector{Vector{Float64}}
    l_star::Vector{Vector{Float64}}
    sum_i::Vector{Float64}
    sum_k::Vector{Float64}
    birthdates::Vector{Vector{Int32}}
    tasks_num::Vector{Int32}
    task_number::Vector{Vector{Int32}}
    running_tasks::Vector{Vector{Any}}
    gamma_history::Vector{Any}
    mu_history::Vector{Any}
end

struct result
    x::Vector{Vector{Vector{Float64}}}
    v_star::Vector{Vector{Vector{Float64}}}
end

zeros_I = []
zeros_K = []

for i in 1:functions_I
    append!(zeros_I, [zeros(dims)])
end
for i in 1:functions_K
    append!(zeros_K, [zeros(dims)])
end

vars = variables(   zeros_I, 
                    zeros_I, 
                    zeros_K, 
                    zeros_K,
                    zeros_K, 
                    zeros_I, 
                    zeros_K, 
                    zeros_I,
                    [0,0], 
                    [0,0,0,0], 
                    [[],[]],
                    [0,0],
                    [[],[]],
                    [[],[]],
                    [],
                    []
                    )

res = result(   [zeros_I],
                [zeros_K])
