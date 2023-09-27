using LinearAlgebra
using ProximalOperators
using Random
include("masking.jl")
include("problem_multiple.jl")

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
    # constant2 = epsilon + ((1/epsilon) * rand())
    constant2 = 1
    append!(constant_m, constant2)
end

include("problem_multiple.jl")
include("variables.jl")
include("loop.jl")

println()
# print("Final ans: ")

# println("reached here")

# x1 = res.x[iters][1]
# x2 = res.x[iters][2]

x_res = []
for i in 1:functions_I
    push!(x_res,res.x[iters][i])
end

# println("reached here too")

# ret_image_left = matrix_to_image(vectorToImage(row1,col1,x1))
# ret_image_right = matrix_to_image(vectorToImage(row2,col2,x2))

ret_images = []
for i in 1:functions_I
    push!(ret_images,matrix_to_image(vectorToImage(row1,col1,x_res[i])))
end

# println("reached here tooooo")

# image_path_left = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_image_left.jpeg"  # Replace with the desired path and filename for the image
# save(image_path_left, ret_image_left)  

# global ret_path_1 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_1.jpeg"
# global ret_path_2 = "//Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_2.jpeg"
# global ret_path_3 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/3.jpeg"
# global ret_path_4 = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/4.jpeg"

ret_path = []
for i in 1:functions_I
    push!(ret_path,"/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_$i.jpeg")
    save(ret_path[i],ret_images[i])
end

# println("reached here tooooo")

# image_path_right = "/Users/kashishgoel/Desktop/Intern_2023/Image_processing/ret_image_right.jpeg"  # Replace with the desired path and filename for the image
# save(image_path_right, ret_image_right)  

# println("reached here tooooo")
println("mu = ",mu_array[1])
println("theta = ",theta_main)
println("sigma of noise = ",sigma_1)
println("sigma of gaussian kernel = ", sigma_blur)


println(check_feasibility())
println()