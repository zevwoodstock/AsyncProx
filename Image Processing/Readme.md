# Image Recovery

This problem focuses on Sterescopic Images' recovery of images that are degraded using Gaussian blurring. The objective is to retrieve original images by minimizing a specific objective function under the given constraints.

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
- Julia packages: `LinearAlgebra`, `ProximalOperators`, `Random`, `Images`, `FileIO`, `Colors`, `Distributed`

## Getting Started

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   cd Image/ Processing
   ```
3. Install the required Julia packages:
   ```julia
   using Pkg
   Pkg.add("LinearAlgebra")
   Pkg.add("ProximalOperators")
   Pkg.add("Random")
   Pkg.add("Images")
   Pkg.add("FileIO")
   Pkg.add("Colors")
   Pkg.add("Distributed")
   ```

## Code Structure

- **`main.jl`**: The main script to be executed, orchestrating the entire computation process.
- **`params.jl`**: Handles user-defined or default parameters used throughout the execution.
- **`loop.jl`**: Contains the iterative loop for computation, updating variables, and recording results.
- **`hyperparameters.jl`**: Defines the hyperparameters for the computation, including gamma and mu values.
- **`variables.jl`, `functions.jl`, `init.jl`, `parameters.jl`, `precomputations.jl`**: Supporting scripts that define necessary functions, initialize variables, and handle precomputations.

## Changes to the problem

Change the parameters section to run in async mode accordingly by changing the following inside parameters.jl - 
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
  julia -p auto main.jl <L_function_bool> <d> <iters> <max_iter_delay> <alpha_> <beta_> <compute_epoch_bool> <record_residual> <record_func> <record_dist> <record_method> <randomize_initial> <initialize_with_zi> <block_function> <generate_gamma_function> <generate_mu_function> <num_images> <img_path_1> ... <img_path_n> <left_shift_pixel_1> ... <left_shift_pixel_n1> <right_shift_pixel_1> .. <right_shift_pixel_n1> <sigma_blur>
  ```

### Example:
```bash
julia -p auto main.jl true 10000 12 5 0.5 0.5 true true true true 0 false true 1 1 1 4 "img1.jpeg" "img2.jpeg" "img3.jpeg" "img4.jpeg" 8 12 15 8 12 15 500.0
```

## Parameters

Parameters can be defined within the script (`params.jl`) or passed as arguments:

- **`iters`**: Integer, number of iterations.

- **`L_function_bool`**: Boolean, determines if L is input as a Matrix of functions.
- **`num_images`**: Integer, number of images you want to process
- **`num_func_I`**: Integer, number of functions f_i (for our problem, also the number of intervals (number of Gi)).
- **`d`**: Integer, dimensionality of x.
- **`num_func_K`**: Integer, number of functions g_k.
- **`img_path_array`**: Vector{String}, contains the path of the images you want to degrade and retrieve
- **`left_shift_pixels`**: Vector{Int64}, contains the pixels by which each image is shifted towards the left
- **`right_shift_pixels`**: Vector{Int64}, contains the pixels by which each image is shifted towards the right
- **`randomize_initial`**: Boolean, this tells if we want to initialise our result x vectors randomly, else they are initialised with 0s
- **`initialize_with_zi`**: Boolean, this tells if we want to initialise our result x with the blurred images, else they are initialised with 0s
- **`block_function`**: Function, Gives the user the choice to select the method to asyncronously choose the Blocks (as of now, get_block_cyclic() is the function, if a user wishes to use some other function, that may be defined in the functions.jl file)
- **`generate_gamma_function`**: Function, determines the method used to choose the gamma values out of the pre-defined functions
- **`generate_mu_function`**: Function, determines the method used to choose the mu values out of the pre-defined functions
- **`sigma_blur`**: Float, sigma used to perform blurring
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