# Instructions to run:
1. Traverse to the folder in your terminal.
2. Type “export JULIA_NUM_THREADS = 4”
3. Open Julia REPL. (I have made a bash script for doing the above- julia.sh)
4. Type - include(“async_prox.jl”)

OR

1. Open your terminal and type: <path_to_julia> <path_to_file.jl>


# Folder contents:-
1.  async_prox.jl - the main file which calls all the other files. It contains the main parameters which we can change.

2.  optimisation_problem.jl - The file that contains the functions we have to optimise.

3.  variables.jl - The file that initialises all the temporary variables.

4.  functions.jl - The file that contains all the helper functions needed to carry out the optimisation.

5.  loop.jl - The file that runs the loop over the desired number of iterations

6.  x1.txt, x2.txt - The values of x that have been exported from the Julia file

7.  prox_plot.ipynb - A jupyter notebook to plot the iteration curves for x.

8.  test.jl - a dummy file used for rough testing

9.  sync.jl - a program that can work out the same optimisation problem synchronously.


# Tasks to be implemented:
1.  Show the running statistics
2.  Plot for all components of x