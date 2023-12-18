include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Statistics
using Base.Threads
PyPlot.rc("axes", grid=true)

include("hamiltonian/CoupledHarmonicOscillatorX2.jl")
φ = (1 + √5)/2 
α = 300
α = sqrt(α) * exp(im * φ)
quant_system = BosonQuantumSystem(Hs, Hc, V/abs(α)^2);
function χ(t::Float64)
    return coherentstate(bclc, exp(-im * Ω * t)*α)
end

N_samp = 2^11 - 1
t0 = 0
tmax = 1
Ts = tmax / N_samp
# time coordinate
T = t0:Ts:tmax
c1_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
c2_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local c1 = Vector{ComplexF64}(undef, length(T))
    local c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = tensor(identityoperator(Hs), dagger(χ(t))) * quant_system.Ψ
        if abs(norm(ϕ)) == 0
            ϕ = ϕ
        else
            ϕ = ϕ / norm(ϕ)
        end
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    global c1_var[index] = var(abs2.(c1))
    global c2_var[index] = var(abs2.(c2))
end
figure(figsize=(6, 8))
hist2D(quant_system.GLOB_EIG_E, c1_var, bins=(300, 300), cmap="plasma", cmin=1)
xlabel("Energy")
ylabel(L"$\sigma^2_{|c_1|^2}$")
title("CoupledHarmonicOscillator with Xs x (Xc/α)^2 coupling at\n CutOff N = $N_ \n |α|^2 = $(abs(α)^2)")
colorbar(orientation="horizontal") 
PyPlot.savefig("data/VarianceCoeff.png", dpi=300)
