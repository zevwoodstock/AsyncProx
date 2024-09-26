# Sparse Linear Classifier Training

This problem focuses on the binary classification problem related to latent group lasso or lasso group overlap. The objective is to solve an optimization problem to find a sparse linear classifier by minimizing a specific objective function under given constraints. 

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Structure](#code-structure)
- [Running the Code](#running-the-code)
- [Parameters](#parameters)
- [Output](#output)

## Prerequisites

Ensure you have the following installed:
- Julia (version 1.5 or later)
- SLURM (for job scheduling)
- Julia packages: `LinearAlgebra`, `ProximalOperators`, `Random`, `TickTock`

## Getting Started

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   cd Group/ Sparse/ Binary/ Distributed
   ```
3. Install the required Julia packages:
   ```julia
   using Pkg
   Pkg.add("LinearAlgebra")
   Pkg.add("ProximalOperators")
   Pkg.add("Random")
   Pkg.add("TickTock")
   ```

## Code Structure

- **`main.jl`**: The main script to be executed, orchestrating the entire computation process.
- **`params.jl`**: Handles user-defined or default parameters used throughout the execution.
- **`loop.jl`**: Contains the iterative loop for computation, updating variables, and recording results.
- **`hyperparameters.jl`**: Defines the hyperparameters for the computation, including gamma and mu values.
- **`variables.jl`, `functions.jl`, `init.jl`, `parameters.jl`, `precomputations.jl`**: Supporting scripts that define necessary functions, initialize variables, and handle precomputations.

## Changes to the problem

Change the parameters section to run in async mode accordingly by changing
```
   params.max_iter_delay = 5 # iters/2 for async, 0 for sync
   params.max_task_delay = 10000 #inf or very large for async, 0 for sync
```


## Running the Code

To run the code, submit a SLURM job using the provided bash script:

```bash
#!/bin/bash
#SBATCH --partition=big   # Specify the desired partition
#SBATCH --mem=4G          # Memory allocation for the job

julia -p auto main.jl
```
You can change the number of processors here by replacing 'auto' with the number n (number of processors).

1. Submit the job:
   ```bash
   sbatch <script-name>.sh
   ```
   The job will run on the specified partition with the allocated memory.
   Optionally, you can use the bash file already provided in the library (run.sh).
2. You can also simply run the julia file from the command line as
   ```bash
   julia -p n main.jl
   ```
where n is the number of processors.

### Command-Line Arguments

The code can be run with or without command-line arguments:

- **Without Arguments**: Defaults will be used for all parameters.
- **With Arguments**: The following format is expected:
  ```bash
  julia -p auto main.jl <L_function_bool> <num_func_I> <d> <num_func_K> <q_datacenters> <iters> <max_iter_delay> <alpha_> <beta_> <compute_epoch_bool> <record_residual> <record_func> <record_dist> <record_method>
  ```

### Example:
```bash
julia -p auto main.jl true 1429 10000 100 50 12 5 0.5 0.5 true true true true 0
```

## Parameters

Parameters can be defined within the script (`params.jl`) or passed as arguments:

- **`L_function_bool`**: Boolean, determines if L is input as a Matrix of functions.
- **`num_func_I`**: Integer, number of functions f_i (for our problem, also the number of intervals (number of Gi)).
- **`d`**: Integer, dimensionality of x.
- **`num_func_K`**: Integer, number of functions g_k.
- **`q_datacenters`**: Integer, number of data centers.
- **`iters`**: Integer, number of iterations.
- **`max_iter_delay`**: Integer, maximum iteration delay allowed before yielding.
- **`alpha_`, `beta_`**: Float, hyperparameters for the optimization.
- **`compute_epoch_bool`**: Boolean, flag to compute epochs.
- **`record_residual`, `record_func`, `record_dist`**: Boolean flags to record various metrics.
- **`record_method`**: Integer, determines the x-axis variable for plotting (iterations, epoch number, prox calls, wall clock time).

## Output

The code will output results depending on the `record_method` and flags provided:
- Residuals
- Function values
- Distance metrics
- Feasibility checks
- Parameter settings

The outputs will be saved or printed based on the configurations in the code.

