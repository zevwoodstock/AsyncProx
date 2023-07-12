using LinearAlgebra
using ProximalOperators
using Random
include("masking.jl")
include("problem.jl")

#intialising our problem
global D = 40
global iters = 500
global epsilon = 0.5

global constant_g = []   # this is being defined if for generate_gamma the strategy being taken is generate_gamma_constant
global constant_m = []   # this is being defined if for generate_mu the strategy being taken is generate_mu_constant

for i in 1:functions_I
    constant1 = epsilon + ((1/epsilon) * rand())
    # constant1 = 1
    append!(constant_g, constant1)
end
for i in 1:functions_K
    constant2 = epsilon + ((1/epsilon) * rand())
    # constant2 = 1
    append!(constant_m, constant2)
end

include("problem.jl")
include("variables.jl")
include("loop.jl")

println()
# print("Final ans: ")

println("reached here")

x1 = res.x[iters][1]
x2 = res.x[iters][2]


println("reached here too")

ret_image_left = matrix_to_image(vectorToImage(row1,col1,x1))
ret_image_right = matrix_to_image(vectorToImage(row2,col2,x2))

println("reached here tooooo")

image_path_left = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_image_left.jpeg"  # Replace with the desired path and filename for the image
save(image_path_left, ret_image_left)  

println("reached here tooooo")

image_path_right = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_image_right.jpeg"  # Replace with the desired path and filename for the image
save(image_path_right, ret_image_right)  

println("reached here tooooo")

println(check_feasibility())
println()
