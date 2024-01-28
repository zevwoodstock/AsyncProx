#!/bin/bash

# -- X -- X -- X -- X -- X -- X -- X -- X -- USER-DEFINED Parameters -- X -- X -- X -- X -- X -- X -- X -- X 

# L_function_bool
# num_func_I
# d
# num_func_K
# q_datacenters
# iters
# max_task_delay
# alpha
# beta
# compute_epoch_bool
# record_residual
# record_func
# record_dist
# record_method

julia --project main.jl \
    false \
    100 \
    1000 \
    300 \
    100 \
    10 \
    40 \
    0.5 \
    0.5 \
    false \
    false \
    true \
    false \
    0
