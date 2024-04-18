mutable struct variables
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
    theta::Float64
    prox_call_count::Int32
    epoch_array::Vector{Int32}
    store_x::Vector{Vector{Vector{Float64}}}
    store_v::Vector{Vector{Vector{Float64}}}
    prox_call::Vector{Int32}
    x_residuals::Vector{Vector{Float64}}
    x_residuals_avg::Vector{Float64}
    f_values::Vector{Float64}
    only_f_values::Vector{Float64}
    dist_to_minima::Vector{Float64}
end

mutable struct parameters
    I::Vector{Int64}
    K::Vector{Int64}
    mu_start::Vector{Float64}
    mu_end::Vector{Float64}
    mu_step::Vector{Float64}
    gamma_start::Vector{Float64}
    gamma_end::Vector{Float64}
    gamma_step::Vector{Float64}
    gamma_a::Vector{Float64}
    gamma_b::Vector{Float64}
    epsilon::Float64
    mu_a::Vector{Float64}
    mu_b::Vector{Float64}
    constant_gamma::Float64
    constant_mu::Float64
    constant_g::Vector{Float64}
    constant_m::Vector{Float64}
    mu_k::Vector{Vector{Float64}}
    beta_k::Vector{Float64}
    compute_epoch_bool::Bool
    record_residual::Bool       
    record_func::Bool           
    record_dist::Bool     
    alpha_::Float64
    beta_::Float64   
    max_task_delay::Int64 
    mapping_ping::Vector{Int64}
    ping_array::Vector{Float64}  
    q_datacenters::Int64
end                       

struct result
    x::Vector{Vector{Vector{Float64}}}
    v_star::Vector{Vector{Vector{Float64}}}
end

mutable struct dim_struct
    num_func_I::Int64
    num_func_K::Int64
    dims_array_I::Vector{Int64}
    dims_array_K::Vector{Int64}
    d::Int64
    iters::Int64
end

# Define the custom HingeDot function type
struct HingeDot
    beta::Vector{Float64}
    mu::Vector{Vector{Float64}}
    k::Int
    
    function HingeDot(beta::Vector{Float64}, mu::Vector{Vector{Float64}}, k::Int)
        if k < 1 || k > length(beta) || k > length(mu)
            error("Invalid index k")
        else
            new(beta, mu, k)
        end
    end
end