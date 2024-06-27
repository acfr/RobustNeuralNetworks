cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux: gradient
using LinearAlgebra
using Random
using RobustNeuralNetworks


# DirectRENParams
ρ = [2.828427]
X = [-2.7931416e-01  3.7577868e-01 -2.0261779e-01 -6.3366574e-01 -3.4560490e-01 -3.5987407e-01  5.2713498e-02 -2.9403549e-01;
     -7.4159384e-01 -2.2106767e-02  3.7575787e-01  2.9376900e-01 -3.9702398e-01  2.4195537e-01  6.9321260e-02 -3.2860145e-02;
      1.5940517e-01 -5.2873248e-01  4.7898936e-01 -2.0917925e-01 -2.6140118e-01 -3.4947243e-01 -4.7762558e-01 -5.7075225e-02;
      3.9345527e-01  2.7090281e-01  7.6781586e-02 -2.0530844e-01 -4.7005671e-01  6.6543192e-01 -2.3814479e-01  5.7374619e-02;
      5.2424037e-04  6.7654616e-01  4.2310429e-01  1.3311568e-01 2.0024508e-01 -2.8737074e-01 -3.2319719e-01  3.4410989e-01;
     -4.3575013e-01 -1.4213189e-01 -3.1522024e-01 -2.9877990e-01 3.4246916e-01  2.6901555e-01 -5.8823276e-01  2.5615335e-01;
     -2.2377064e-02 -1.6549540e-01  6.1953943e-02 -3.2520643e-01 -1.6300470e-01 -6.2262207e-02  4.1818678e-01  8.1066978e-01;
     -3.7762433e-02 -7.1110860e-03  5.4791129e-01 -4.6207291e-01 4.9969169e-01  2.9696691e-01  2.8674400e-01 -2.5436604e-01]
B2 = reshape([0.5770632, 0.14482902], 2, 1)
D12 = reshape([-0.39930755,  0.23696136, -0.01424979, -0.43245763], 4, 1)
Y1 = [-0.80490816 -0.13022006;
       0.8404852   0.00755759]
C2 = [ 0.74835783 -0.5171219 ]
D21 = [ 0.5457871 -0.5266131  1.3562537  0.36085203]
D22 = reshape([-0.42949265], 1, 1)
X3 = zeros(0, 0)
Y3 = zeros(0, 0)
Z3 = zeros(0, 0)
bx = [-0.4647214,  0.8140386]
bv = [ 0.31041098, -0.13666147,  0.39765638,  0.44175062]
by = [0.25065297]

T = Float32
ϵ = T(1e-12)

direct_ps = DirectRENParams{T}(
    X, 
    Y1, X3, Y3, Z3, 
    B2, C2, D12, D21, D22,
    bx, bv, by, ϵ, ρ,
    true, true, false,
    false
)
nu, nx, nv, ny = 1, 2, 4, 1
ren_ps = ContractingRENParams{T}(tanh, nu, nx, nv, ny, direct_ps, T(1))
ren = REN(ren_ps)

batches = 4
rng = Xoshiro(42)
x0 = init_states(ren, batches; rng) .+ 1
u0 = ones(ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)
println(x1)
println(y1)

function loss(states, inputs)
    model = REN(ren_ps)
    nstate, out = model(states, inputs)
    return sum(nstate.^2) + sum(out.^2)
end

gs = gradient(loss, x0, u0)
println(loss(x0, u0))
println("States grad: ", gs[1])
println("Output grad: ", gs[2])
