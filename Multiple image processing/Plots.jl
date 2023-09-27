using Plots
gr()

# plot(f_values, title = "Plot of Function Values", xlabel = "Iteration", ylabel = "sigma(f_i, g_k)")


# for plotting against number of iterations

# x_residuals_t = [[x_residuals[j][i] for j in 1:length(x_residuals)] for i in 1:length(x_residuals[1])]
# plot(x_residuals_t[1], title = "Plot of |x_" * string(1) * " - x*|", xlabel = "number of iterations", ylabel = "Distance from solution")


# f_values_t = [[f_values[j][i] for j in 1:length(f_values)] for i in 1:length(f_values[1])]
# plot(f_values_t[1], title = "Plot of |f(x_n) - f(x*)|", xlabel = "number of iterations", ylabel = "Difference of function values",yscale = :log10)


# dist_to_minima_t = [[dist_to_minima[j][i] for j in 1:length(dist_to_minima)] for i in 1:length(dist_to_minima[1])]
# plot(dist_to_minima_t[1], title = "Plot of |x_n - x*|", xlabel = "number of iterations", ylabel = "Distance from solution",yscale = :log10)


# for plotting against epoch number

# x_residuals_t = [[x_residuals[j][i] for j in 1:length(x_residuals)] for i in 1:length(x_residuals[1])]
# plot(x_residuals_t[1], title = "Plot of |x_" * string(1) * " - x*|", xlabel = "epoch number", ylabel = "Distance from solution")


# f_values_t = [[f_values[j][i] for j in 1:length(f_values)] for i in 1:length(f_values[1])]
# plot(f_values_t[1], title = "Plot of |f(x_n) - f(x*)|", xlabel = "epoch number", ylabel = "Difference of function values",yscale = :log10)


# dist_to_minima_t = [[dist_to_minima[j][i] for j in 1:length(dist_to_minima)] for i in 1:length(dist_to_minima[1])]
# plot(dist_to_minima_t[1], title = "Plot of |x_n - x*|", xlabel = "epoch number", ylabel = "Distance from solution",yscale = :log10)


# for plotting against number of prox calls

# x_residuals_t = [[x_residuals[j][i] for j in 1:length(x_residuals)] for i in 1:length(x_residuals[1])]
# plot(prox_count_array,x_residuals_t[1], title = "Plot of |x_" * string(1) * " - x*|", xlabel = "number of prox calls", ylabel = "Distance from solution")


# f_values_t = [[f_values[j][i] for j in 1:length(f_values)] for i in 1:length(f_values[1])]
# plot(prox_count_array,f_values_t[1], title = "Plot of |f(x_n) - f(x*)|", xlabel = "number of prox calls", ylabel = "Difference of function values",yscale = :log10)


# dist_to_minima_t = [[dist_to_minima[j][i] for j in 1:length(dist_to_minima)] for i in 1:length(dist_to_minima[1])]
# plot(prox_count_array,dist_to_minima_t[1], title = "Plot of |x_n - x*|", xlabel = "number of prox calls", ylabel = "Distance from solution",yscale = :log10)


# to plot function values against real function value 
# img_arr_1 = imageToVector(image_to_vector(path_1))
# img_arr_2 = imageToVector(image_to_vector(path_2))
# global sum11 = 0
# for i in 1:functions_I
#     global sum11 +=functions[i]([img_arr_1, img_arr_2][i])
# end
# for k in 1:functions_K
#     global sum11 += functions[functions_I+k](matrix_dot_product(get_L(L, k), [img_arr_1, img_arr_2]))
# end

# global sum12 = 0
# for i in 1:functions_I
#     global sum11 +=functions[i]([img_arr_1, img_arr_2][i])
# end
# for k in 1:functions_K
#     global sum11 += functions[functions_I+k](matrix_dot_product(get_L(L, k), [img_arr_1, img_arr_2]))
# end
# y2 = sum11
# plot(x = range(50, 500, length=100), [only_f_values, constant_vector(500, y2)], title = "Plot of of objective function value", xlabel = "Iteration", ylabel = "f_i(x_i) + g_k(y_k)", yscale = :log10)


# for (c, xi) in enumerate(x_residuals)
#     local c  # Declare c as a local variable
#     plot(xi, title = "Plot of |x_" * string(c) * " - x*|", xlabel = "xi", ylabel = "Distance from solution")
# end


arr_a_t = [[arr_a[j][i] for j in 1:length(arr_a)] for i in 1:length(arr_a[1])]
arr_b_t = [[arr_b[j][i] for j in 1:length(arr_b)] for i in 1:length(arr_b[1])]
plot([arr_b_t[1],arr_a_t[1]], title = "Plot of |x_n - x*|", xlabel = "number of iterations", ylabel = "Distance from solution")
