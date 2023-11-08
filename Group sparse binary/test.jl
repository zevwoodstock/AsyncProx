include("hinge_dot.jl")
m = 7  # The number of intervals (number of Gi)
d = 10

global mu1::Vector{Vector{Float64}} = []
for _ in 1:p
    random_vector = randn(Float64, d)
    rnorm = NormL2(1)(random_vector)
    random_vector = random_vector/rnorm
    # println(random_vector)
    push!(mu1, random_vector)
end
gamma = 1
beta = [1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,-1.0]
x = [0.01, 0.02, 0.03, .04, 0.05, 0.66, 0.07, 0.06, 0.05, 0.02]
k = 3
f = HingeDot(beta, mu1, k)



function proxmin(x, gamma, z)
    y = f(z) + SqrNormL2(1)(x-z)/(gamma)
    return y
end



println(f(x))
y, fy = prox(f, x, gamma)
eps = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
z = y-eps*1000
mn = 100000000
for z1 in 1:2000
    z2 = z + eps*z1
    global mn = min(mn, proxmin(x, gamma, z2))
    # if mn == proxmin(x, gamma, z2)
    #     println(z2)
    # end
end


println(y)
println(mn) #coordinate y at which f(y) + 1/2gamma * ||x-y||^2 is minimum
println(proxmin(x, gamma, y))