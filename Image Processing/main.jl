
@everywhere using LinearAlgebra
@everywhere using ProximalOperators
@everywhere using Random
@everywhere using Images, FileIO, LinearAlgebra, Colors, ImageView, Wavelets
using TickTock
println()
println("-------------------------------------------------------------------")
println("execution started")

@everywhere include("variables.jl")
@everywhere include("functions.jl")
@everywhere include("init.jl")
@everywhere include("parameters.jl")
@everywhere include("hyperparameters.jl")
@everywhere include("precomputations.jl")

tick()
include("loop.jl")
tock()

x_res = []
for i in 1:dimensions.num_func_I
    push!(x_res,res.x[iters][i])
end

ret_images = []
for i in 1:dimensions.num_func_I
    push!(ret_images,matrix_to_image(vectorToImage(row1,col1,x_res[i])))
end

ret_path = []
for i in 1:dimensions.num_func_I
    println("saving the recovered images")
    push!(ret_path,"/Users/kashishgoel/Desktop/Intern_2023/Multiple_Image_processing/ret_$i.jpeg")
    save(ret_path[i],ret_images[i])
end

record()
get_accuracy()
println(check_feasibility())
print_params()
println()

