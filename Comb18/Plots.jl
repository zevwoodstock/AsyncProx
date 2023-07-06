using Plots
gr()

# plot(f_values, title = "Plot of Function Values", xlabel = "Iteration", ylabel = "sigma(f_i, g_k)")

x_residuals_t = [[x_residuals[j][i] for j in 1:length(x_residuals)] for i in 1:length(x_residuals[1])]
plot(x_residuals_t[1], title = "Plot of |x_" * string(1) * " - x*|", xlabel = "xi", ylabel = "Distance from solution")


# for (c, xi) in enumerate(x_residuals)
#     local c  # Declare c as a local variable
#     plot(xi, title = "Plot of |x_" * string(c) * " - x*|", xlabel = "xi", ylabel = "Distance from solution")
# end
