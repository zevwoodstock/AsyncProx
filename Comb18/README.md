Instructions to run:
1. Traverse to the folder in your terminal.
2. Type “export JULIA_NUM_THREADS = 4”
3. Open Julia REPL. (I have made a bash script for doing the above- julia.sh)
4. Type - include(“async_prox.jl”)



Folder contents:-
# async_prox.jl - the main file which calls all the other files. It contains the main parameters which we can change.

# optimisation_problem.jl - The file that contains the functions we have to optimise.

# variables.jl - The file that initialises all the temporary variables.

# functions.jl - The file that contains all the helper functions needed to carry out the optimisation.

# loop.jl - The file that runs the loop over the desired number of iterations

# x1.txt, x2.txt - The values of x that have been exported from the Julia file

# prox_plot.ipynb - A jupyter notebook to plot the iteration curves for x.

# test.jl - a dummy file used for rough testing

# sync.jl - a program that can work out the same optimisation problem synchronously.



Tasks to be implemented:
# Show the running statistics
# Check the feasibility
# Plot for all components of x