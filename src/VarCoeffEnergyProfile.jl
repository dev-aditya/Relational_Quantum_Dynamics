include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Statistics
using FFTW # for fft
using Base.Threads
PyPlot.rc("axes", grid=true)

include("hamiltonian/powerLawCoupling.jl")

HC_EIG_E_loc, HC_EIG_V_loc = eigenstates(dense(Hc));
function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end

quant_system = SpinQuantSystem(Hs, Hc, V, χ);

N_samp = 2^12 - 1
t0 = 0
tmax = 2π
Ts = tmax / N_samp
# time coordinate
T = t0:Ts:tmax

freqs = fftfreq(length(T), 1.0/Ts) |> fftshift; ## Equivalent to fftshift(fftfreq(N, 1.0/Ts))
c1_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
c2_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local c1 = Vector{ComplexF64}(undef, length(T))
    local c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = tensor(identityoperator(Hs), dagger(χ(t, quant_system.EΨ))) * quant_system.Ψ
        ϕ = ϕ / norm(ϕ)
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    global c1_var[index] = var(abs2.(c1))
    global c2_var[index] = var(abs2.(c2))
end

figure(figsize=(6, 8))
hist2D(quant_system.GLOB_EIG_E, c1_var, bins=(80, 80), cmap="plasma", cmin=1)
xlabel("Energy")
ylabel(L"$\sigma^2_{|c_1|^2}$")
title("Variance of the coefficient for N = $N spins")
colorbar(orientation="horizontal") 
PyPlot.savefig("data/VarianceCoeff_$N-spins.svg")
