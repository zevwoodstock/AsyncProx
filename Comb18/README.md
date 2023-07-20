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
6.  x.txt, x1.txt, x2.txt
7.  prox_plot.ipynb
8.  test.jl
9.  sync.jl
10. start_repl.sh
11. run_async.sh

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
    2. iters - The number of iterations the optimisation loop will run
    3. epsilon - The hyperparameter that plays a part in choosing the gammas and mus, which decide the values of the proxes.  

2. optimisation_problem.jl
    >This is the file that contains the functions we have to optimise.
    >The functions to be optimised are initialised here, and then stored in a vector of functions known as functions[].\
    >The L matrix is also initialised here, in either of the 3 forms as mentioned above.

3.  variables.jl
    >This is the file that initialises all the temporary variables into a struct known as variables, and the solution variables into a struct known as result. 
    >It initialises every temporary variable to either a zero array or an empty array, as per the need.
    >The variable, prox_call, which is an array is used to keep a track of whether the prox call for a function has been made or not by using the boolean values, 0 and 1 for each function.
    >The variable prox_call_count is used to keep a count of the number of prox calls that have been made uptil that point
    >The variables store_x and store_v are used to store the values of x and v calculated at each iteration
    >The variable, x_residuals, is used to store the L2 norm of the difference of the value of the variable, x calculated at iteration n+1 with the value of variable, x calculated at iteration n
    >The variable, f_values, is used to store the absolute difference of the function value at each iteration and the optimal function value
    >The variable, only_f_values, is used to store the function values at each iteration
    >The variable, dist_to_minima, is used to store the L2 norm of the difference of the value of x at each iteration with the optimal x value

4.  functions.jl
    >This is the file that contains all the helper functions needed to carry out the optimisation.
    >The functions contained are:
    1. rearrange(L) - Shifts the rows and columns of L; it has been defined according to different type settings of L
    2. norm_function() - Finds the L2 norm of a vector
    3. linear_operator_sum(L, x, transpose) - Finds the L[i]*x vector multiplication ; it too has been defined for the various type settings for L
    4. matrix_dot_product(v, u) - This function takes as input two vectors of different types and returns their product
    5. generate_random(epsilon, index) - Finds new random values of gamma and mu
    6. get_L(matrix, index) - Finds the given index from the given matrix
    7. get_minibatch(j) - Takes the result of get_bitvector_pair(j, index) and stores it in an array
    8. get_bitvector_pair(j, index) - finds a random bitvector and its complement and stores them in a vector
    9. check_task_delay(j) - Forcibly fetches a task if it takes too long to complete.
    10. compute(j) - Computes the other operations in the algorithm, after the calculation of the prox. 
    11. custom_prox(t,f,y,gamma) - Performs the prox operation, but after a "t" second time delay.
    12. define_tasks(j) - Schedules new tasks in each iteration
    13. calc_theta(j) - Uses the temporary variables and calculates theta.
    14. update_vars(j) - Updates both the solution variables in the result struct.
    15. update_params(j) - Updates the values of gamma and mu
    16. delete_task(ind, birth) - Deletes a task from the vector of tasks
    17. add_task(task, index, j, i) - Adds a task to the vector of tasks
    18. write(j) - Writes "x1.txt"/"x2.txt" to file.
    19. check_feasibility() - Returns a simple true or false based on the feasibility of the returned solution. 
    20. phi(x) -
    21. soft_threshold(x, lambda) -  
    22. compute_epoch() - This function after every iteration checks if prox call for every function has been made atleast once ; returns true if the same has been done
    23. record() - This function is used to record the various experimental values according to different x-axis plotting methods which maybe chosen by the user in optimisation_problem.jl
    24. get_block_cyclic(n, m, M) - 
    25. generate_gamma_constant(i,j) - This function is designed to set the values of gamma as a constant for each iteration. The random is either randomly assigned or is set as 1 in the main.jl file
    26. generate_mu_constant(k,j) - This function is designed to set the values of mu as a constant for each iteration. The random is either randomly assigned or is set as 1 in the main.jl file
    27. generate_gamma_seq(i,j) - This function is designed to set the value of gamma as a reducing sequence which begins with the value 1/epsilon and decreases with the common difference of 0.1 and once it reaches the value of epsilon it becomes constant
    28. generate_mu_seq(k,j) - This function is designed to set the value of mu as a reducing sequence which begins with the value 1/epsilon and decreases with the common difference of 0.1 and once it reaches the value of epsilon it becomes constant
    
5.  loop.jl
    >The file that runs the loop over the desired number of iterations
    >It calls the functions in functions.jl from the body of the for loop
    >The loop defines the blocks for the particular iteration for the two sets - I and K as I_n and K_n respectively
    >At the end of each iteration it checks if the prox calls for all functions have been completed once and if yes then it pushes the iteration number on the epoch array
    >There is also a provision to calculate the total number of prox calls uptil a particular itertion number and is used to make plots against number of prox calls
    >After the for loop ends, if the user has set the option to record the values for plotting as true then it calculates the various variable values

6.  x.txt, x(i).txt
    >The values of x that have been exported from the Julia file
    >x.txt is the result of the synchronous version 
    >x(i).txt are the results of the asynchronous version of the ith dimension of x

7.  prox_plot.ipynb
    >A jupyter notebook to plot the iteration curves for x.

8.  test.jl
    >A dummy file used for rough testing

9.  sync.jl
    >A program that can work out the same optimisation problem synchronously.

10. start_repl.sh
    >A bash script file that opens the Julia REPL.
    >Be sure to edit this scipt and replace my Julia path with yours.

11. run_async.sh
    >A bash script file that runs the asynchronous version of the algorithm if you run it from the terminal inside this directory.
    >Be sure to edit this scipt and replace my Julia path with yours.
