cd(@__DIR__)
using Pkg
Pkg.activate(".")

using BenchmarkTools
using ChainRulesCore: NoTangent, rrule
using LinearAlgebra
import RobustNeuralNetworks: solve_tril_layer

function solve_tril_layer(σ::F, D11, b) where F
    z_eq  = similar(b)
    Di_zi = typeof(b)(zeros(Float32, 1, size(b,2))) 
    for i in axes(b,1)
        Di = @view D11[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]

        mul!(Di_zi, Di, zi)
        z_eq[i:i,:] .= σ.(Di_zi .+ bi)  
    end
    return z_eq
end


D = [0.  0.         0.         0.         0.        ;
     0.27276897 0.0  0.         0.         0.       ;
     0.8973534  0.45088673 0.0 0.         0.        ;
     0.94310784 0.02125645 0.44761765 0.0 0.        ;
     0.24344909 0.17582    0.18456626 0.40024185 0.0 ]

b =[0.5338     0.9719182  0.61623883 0.868845   0.6309322 ;
    0.20438278 0.7415488  0.15026295 0.21696508 0.32493377;
    0.7355863  0.79253435 0.3715024  0.1306243  0.04838264]
# b =[0.5338     0.9719182  0.61623883 0.868845   0.6309322]

b = permutedims(b, (2,1))
σ = tanh

# @btime solve_tril_layer(σ, D, b)
w_eq = solve_tril_layer(σ, D, b)
println(round.(transpose(w_eq), digits=3))


function tril_layer_back(σ::F, D11, v, w_eq::AbstractVecOrMat{T}) where {F,T}
    return w_eq
end

function testing(σ::F, D11, v, w_eq::AbstractVecOrMat{T}) where {F,T}

    # Forwards pass
    y = tril_layer_back(σ, D11, v, w_eq)

    # Reverse mode
    function tril_layer_back_pullback(Δy)

        Δf = NoTangent()
        Δσ = NoTangent()
        ΔD11 = NoTangent()
        Δb = NoTangent()

        # Get gradient of σ(v) wrt v evaluated at v = D₁₁w + b
        _back(σ, v) = rrule(σ, v)[2]
        backs = _back.(σ, v)
        j = map(b -> b(one(T))[2], backs)

        # Compute gradient from implicit function theorem
        Δw_eq = v
        for i in axes(Δw_eq, 2)
            ji = @view j[:, i]
            Δyi = @view Δy[:, i]
            Δw_eq[:,i] = (I - (ji .* D11))' \ Δyi
        end
        return Δf, Δσ, ΔD11, Δb, Δw_eq
    end

    return y, tril_layer_back_pullback
end

v = D*w_eq + b
y, back = testing(σ, D, v, w_eq)

out = back(ones(size(y)))
println(permutedims(out[5], (2,1)))