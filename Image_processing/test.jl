using LinearAlgebra
using ProximalOperators

"""struct sleepInd(I)
inner ::I
end
po.prox(y,f::IS)
sleep(time)
prox(y,f)
"""

function custom_prox(t, f, y, gamma)
    sleep(t)
    a,b = prox(f,y,gamma)
    return a,b
end