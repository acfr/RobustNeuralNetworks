cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux: gradient
using LinearAlgebra
import RobustNeuralNetworks: tril_eq_layer


D = [0.  0.         0.         0.         0.        ;
     0.27276897 0.0  0.         0.         0.       ;
     0.8973534  0.45088673 0.0 0.         0.        ;
     0.94310784 0.02125645 0.44761765 0.0 0.        ;
     0.24344909 0.17582    0.18456626 0.40024185 0.0 ]

b =[0.5338     0.9719182  0.61623883 0.868845   0.6309322 ;
    0.20438278 0.7415488  0.15026295 0.21696508 0.32493377;
    0.7355863  0.79253435 0.3715024  0.1306243  0.04838264]

b = permutedims(b, (2,1))
σ = tanh

# @btime solve_tril_layer(σ, D, b)
w_eq = tril_eq_layer(σ, D, b)
println(round.(transpose(w_eq), digits=3))

function loss(D, b)
    w_eq = tril_eq_layer(σ, D, b)
    return sum(w_eq.^2)
end

gs = gradient(loss, D, b)

println(loss(D,b))
println("Gradient for D: ", gs[1])
println("Gradient for b: ", gs[2])
