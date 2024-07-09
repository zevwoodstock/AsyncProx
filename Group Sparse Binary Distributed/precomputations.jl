# -- X -- X -- X -- X -- X -- X -- X -- X -- Some Precomputations -- X -- X -- X -- X -- X -- X -- X -- X 

params.mapping_ping = rand(1:params.q_datacenters, dimensions.num_func_K)
params.ping_array = rand(Float64, params.q_datacenters) .* 0.5

dimensions.dims_array_I = fill(dimensions.d, dimensions.num_func_I)
dimensions.dims_array_K = fill(dimensions.d, dimensions.num_func_K)
L_matrix = fill(fill(1.0, dimensions.num_func_I), dimensions.num_func_K)

conditions = []
push!(conditions, [Condition() for _ in 1:dimensions.num_func_I])
push!(conditions, [Condition() for _ in 1:dimensions.num_func_K])

if L_function_bool == true
    L_operator = L_function
else
    L_operator = L_matrix
end

L_operator_transpose = rearrange(L_operator)

if record_method == 1
    params.compute_epoch_bool = true
end

params.constant_g = []   # this is defined if for generate_gamma the strategy taken is generate_gamma_constant
params.constant_m = []   # this is defined if for generate_mu the strategy taken is generate_mu_constant
calculate_mu_beta()

functions = []
for i in 1:dimensions.num_func_I
    append!(functions,[SqrNormL2(2)])
    append!(params.constant_g, params.constant_gamma)
end
for k in 1:dimensions.num_func_K
    append!(functions,[HingeDot(params.beta_k, params.mu_k, k)])
    append!(params.constant_m, params.constant_mu)
end

zeros_I = []
zeros_K = []

for i in 1:dimensions.num_func_I
    append!(zeros_I, [zeros(dimensions.dims_array_I[i])])
end
for i in 1:dimensions.num_func_K
    append!(zeros_K, [zeros(dimensions.dims_array_K[i])])
end

vars = variables(   zeros_I, 
                    zeros_I, 
                    zeros_K, 
                    zeros_K,
                    zeros_K, 
                    zeros_I, 
                    zeros_K, 
                    zeros_I,
                    zeros(dimensions.num_func_I),       
                    zeros(dimensions.num_func_K),   
                    [[],[]],     
                    [0,0],       
                    [[],[]],     
                    [[],[]],     
                    [],
                    [],
                    0.0,
                    0,
                    [],
                    Vector{Vector{Vector{Float64}}}(undef, dimensions.iters),
                    Vector{Vector{Vector{Float64}}}(undef, dimensions.iters),
                    [],
                    Vector{Vector{Float64}}([]),
                    [],
                    [],
                    [],
                    [])

res = result(   [zeros_I],
                [zeros_K])

for i in 1:dimensions.num_func_I+dimensions.num_func_K
    push!(vars.prox_call,0)
end
