generate_gamma = generate_gamma_linear_decrease
generate_mu = generate_mu_linear_decrease

# For generate_gamma_constant
params.constant_gamma = 1

# For generate_mu_constant
params.constant_mu = 0.35

# For generate_gamma and mu random
params.epsilon = 0.01

# For generate_gamma_linear_decrease
params.gamma_start = []
params.gamma_end = []
params.gamma_step = []
for i in 1:dimensions.num_func_I
    append!(params.gamma_start, 10)
    append!(params.gamma_end, 0.5)
    append!(params.gamma_step, 1)
end

# For generate_mu_linear_decrease

# start - step*j
params.mu_start = []
params.mu_end = []
params.mu_step = []
for k in 1:dimensions.num_func_K
    append!(params.mu_start, 10)
    append!(params.mu_end, 0.25)
    append!(params.mu_step, 1)
end

# For generate_gamma_nonlinear_decrease

# a + b/j
params.gamma_a = []
params.gamma_b = []
for i in 1:dimensions.num_func_I
    append!(params.gamma_a, 0.5)
    append!(params.gamma_b, 5)
end

# For generate_mu_nonlinear_decrease
params.mu_a = []
params.mu_b = []
for i in 1:dimensions.num_func_K
    append!(params.mu_a, 0)
    append!(params.mu_b, 5)
end