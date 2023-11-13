include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Base.Threads
using Statistics
using LaTeXStrings
using DelimitedFiles
using FFTW

include("hamiltonian/powerLawCoupling.jl")

HC_EIG_E_loc, HC_EIG_V_loc = eigenstates(dense(Hc));
function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end

quant_system = SpinQuantSystem(Hs, Hc, V, χ);
## Calculate Degeneracy
energy_levels = quant_system.GLOB_EIG_E
energy_levels = round.(energy_levels, digits=8)
degeneracy = Dict{Float64,Int}()
for E in energy_levels
    if haskey(degeneracy, E)
        degeneracy[E] += 1
    else
        degeneracy[E] = 1
    end
end

outfile = "data/energy_degeneracy_$N-powerLawCoupling_$l.txt"
writedlm(outfile, degeneracy, '\t')

T = Base._linspace(0.0, 2 * π, 1000);
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    global c1 = Vector{ComplexF64}(undef, length(T))
    global c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = φ_λ(t, quant_system)
        ϕ = ϕ / norm(ϕ)
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    ene = quant_system.EΨ
    c1 = abs2.(c1) .- mean(abs2.(c1))
    #c2 = abs2.(c2) .- mean(abs2.(c2))
    Y1 = fft(c1) |> fftshift
    freqs1 = fftfreq(length(T), T[2] - T[1]) |> fftshift
    entan = real(entanglement_entropy(quant_system.Ψ, [i for i = 2:N])) / log(2)
    fig, axs = subplots(2, 1, figsize=(20, 10),)
    axs[1].plot(T, log.(c1))
    axs[1].set_title("C1^2 coefficient", fontsize=15)
    axs[1].set_xlabel("t", fontsize=15)
    axs[1].set_ylabel(L"log(|c_1|^2 - \overline{|c_1|^2})", fontsize=15)
    axs[2].plot(freqs1, log.(abs.(Y1)))
    axs[2].set_title("Freq Spectrum", fontsize=15)
    axs[2].set_xlabel("Freq(Hz)", fontsize=15)
    axs[2].set_ylabel("log(Power)", fontsize=15)
    fig.suptitle("Energy: $(ene) \n Entanglement: $(entan)", fontsize=15)
    PyPlot.savefig("data/index_$index-powerLaw.png",)
    PyPlot.close()
end