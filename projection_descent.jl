using LinearAlgebra

struct Ball
    centre:: Vector{Float64}
    radius::Float64
end

struct Plane
    a:: Vector{Float64}
    b:: Float64
end

struct HalfPlane
    a:: Vector{Float64}
    b:: Float64
end

struct function_params
    A:: Matrix{Float64}
    b:: Vector{Float64}
end




function project(x, set::Ball)

    v = x - set.centre
    mod = sqrt(dot(v,v))
    factor = set.radius/mod
    if factor < 1
        return (factor*v + set.centre)
    else
        return x
    end
    end

function project(x, set::Plane)
    dot_prod = dot(set.a, x)
    mod = dot(set.a, set.a)
    factor = (dot_prod - set.b)/mod
    final = x - factor*set.a
    return final
    end

function project(x, set::HalfPlane)
    condn = dot(set.a, x) > set.b
    if condn == true
        dot_prod = dot(set.a, x)
        mod = dot(set.a, set.a)
        factor = (dot_prod - set.b)/mod
        final = x - factor*set.a
        return final
    else
        return x
    end
    end


function descent(set, alpha, params)
    x = [0;0;0]
    x = project(x, set)
    for i in 1:100
        y = x - alpha*gradf(params, x)
        x = project(y, set)
        print("x = ")
        println(x)
        print("y = ")
        println(y)
        println()
    end
    return x
end

function f(params, x)
    return x'*params.A*x + params.b'*x
end

function gradf(params, x)
    return 2*params.A*x + params.b
end


set1 = Ball([2, 2, 2], 1)           #2 dimensional ball of radius 1 centred at the origin
set2 = Plane([1, 1, 1], 1)              # a 2-dimensional Plane, x+y =1
set3 = HalfPlane([1, 1], 1)         # an inequality , x + y <=1

params = function_params([1 0 0; 0 1 0; 0 0 1], [1;1;1])
x = [1;1]

print(descent(set2, 0.001, params))