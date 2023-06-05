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

for i in 1:functions_I                     # I believe this is already generalised according to dimensions
    append!(zeros_I, [zeros(dims)])
end
for i in 1:functions_K                     # I believe this is already generalised according to dimensions
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
                    [0,0],       # here sum_i has two zeroes because of functions_i being 2 right and not because of 2 dimensions?
                    [0,0,0,0],   # here sum_k has 4 zeroes because of functions_k being 4 right and not because of 2 dimensions?
                    [[],[]],     # do we need to generalise the birthdates variable for n dimensions ? Idts cause it has 2 elements corresponding to I_n and K_n
                    [0,0],       # same doubt as birthdates for tasks_num
                    [[],[]],     # same doubt as above for task_number
                    [[],[]],     # same doubt as above for running tasks
                    [],
                    []
                    )

res = result(   [zeros_I],
                [zeros_K])
