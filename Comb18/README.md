# Instructions to run:
1. Traverse to the folder in your terminal.
2. Type “export JULIA_NUM_THREADS = 4”
3. Open Julia REPL. (I have made a bash script for doing the above- julia.sh)
4. Type - include(“async_prox.jl”)

OR

1. Find the path to julia on your computer. For example, in mine it is '/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia'
2. Open 'run_async.sh' and replace my path with your path.
1. Open your terminal in this directory and type: 'sh run_async.sh'


# Folder contents:-
1.  main.jl
2.  optimisation_problem.jl
3.  variables.jl
4.  functions.jl
5.  loop.jl
6.  x1.txt, x2.txt
7.  prox_plot.ipynb
8.  test.jl
9.  sync.jl

# Tasks to be implemented:
1.  Show the running statistics
2.  Plot for all components of x


# Tasks implemented so far
1. The algorithm works asynchronously and synchronously
2. The number of threads have been set to 4 in the asynchronous setting
3. The functions to be optimised are to be entered in functions.jl. They can be any functions from the ProximalOperators library in Julia. 
4. "L" can now be entered as a matrix of linear operators(functions), a matrix of matrices(a bigger matrix), or a matrix of integers.


# Brief Description of the files
1. main.jl
    >It is the main file which connects every other file. It imports the content of the other julia files.
    >It contains:

    1. D - The delay after which every unfinished task is to be forcibly fetched
    2. dims - The number of dimensions of the solution vector
    3. functions_I - The number of functions which require only 1 independent variable in our problem
    4. functions_K - The number of functions which require a linear combination of independent variables in our problem.
    5. iters - The number of iterations the optimisation loop will run
    6. epsilon - The hyperparameter that plays a part in choosing the gammas and mus, which decide the values of the proxes.

    

2. optimisation_problem.jl
    >This is the file that contains the functions we have to optimise.
    >The functions to be optimised are initialised here, and then stored in a vector of functions known as functions[].
    >The L matrix is also initialised here, in either of the 3 forms as mentioned above.

3.  variables.jl
    >This is the file that initialises all the temporary variables into a struct known as variables, and the solution variables into a struct known as result. 
    >It initialises every temporary variable to either a zero array or an empty array, as per the need.

4.  functions.jl
    >This is the file that contains all the helper functions needed to carry out the optimisation.
    >The functions contained are:
    1. rearrange(L) - Shifts the rows and columns of L
    2. norm_function() - Finds the L2 norm of a vector
    3. linear_operator_sum(L, x, transpose) - Finds the L[i]*x vector multiplication.
    4. generate_random(epsilon, index) - Finds new random values of gamma and mu
    5. get_L(matrix, index) - Finds the given index from the given matrix
    6. get_minibatch(j) - Takes the result of get_bitvector_pair(j, index) and stores it in an array
    7. get_bitvector_pair(j, index) - finds a random bitvector and its complement and stores them in a vector
    8. check_task_delay(j) - Forcibly fetches a task if it takes too long to complete.
    9. compute(j) - Computes the other operations in the algorithm, after the calculation of the prox. 
    10. custom_prox(t,f,y,gamma) - Performs the prox operation, but after a "t" second time delay.
    11. define_tasks(j) - Schedules new tasks in each iteration
    12. calc_theta(j) - Uses the temporary variables and calculates theta.
    13. update_vars(j) - Updates both the solution variables in the result struct.
    14. update_params(j) - Updates the values of gamma and mu
    15. delete_task(ind, birth) - Deletes a task from the vector of tasks
    16. add_task(task, index, j, i) - Adds a task to the vector of tasks
    17. write(j) - Writes "x1.txt"/"x2.txt" to file.
    18. check_feasibility() - Returns a simple true or false based on the feasibility of the returned solution. 

5.  loop.jl
    >The file that runs the loop over the desired number of iterations
    >It calls the functions in functions.jl from the body of the for loop

6.  x1.txt, x2.txt
    >The values of x that have been exported from the Julia file

7.  prox_plot.ipynb
    >A jupyter notebook to plot the iteration curves for x.

8.  test.jl
    >A dummy file used for rough testing

9.  sync.jl
    >A program that can work out the same optimisation problem synchronously.
