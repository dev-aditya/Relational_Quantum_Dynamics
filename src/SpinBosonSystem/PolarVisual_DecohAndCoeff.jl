include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
PyPlot.rc("axes", grid=true)
using LaTeXStrings
using Base.Threads

include("hamiltonian/SpinBoson.jl")
quant_system = BosonQuantumSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;
φ = (1 + √5)/2 
α = quant_system.GLOB_EIG_E[100] / (ħ*Ω) - 1/2
α = sqrt(α) * exp(im * φ)
function χ(t::Float64)
    return coherentstate(clck_basis, exp(-im * Ω * t)*α)
end

Energy_expect = expect(Hc, χ(0.0))

N_samp = 2^10 - 1
t0 = 0
tmax = 1
Ts = tmax / N_samp
# time coordinate
global T = t0:Ts:tmax
for index in eachindex(quant_system.GLOB_EIG_E)
    local r_t_rel = Vector{ComplexF64}(undef, length(T))
    local r_t = Vector{ComplexF64}(undef, length(T))
    local c_1 = Vector{ComplexF64}(undef, length(T))
    local c_2 = Vector{ComplexF64}(undef, length(T))
    UpdateIndex(quant_system, index)
    energy = quant_system.GLOB_EIG_E[index]
    Φ = Vector{Ket}(undef, length(T))
    Ec = real(expect(Hc, χ(0.0)))
    for i in eachindex(T)
        Φ[i] = tensor(identityoperator(Hs), dagger(χ(T[i]))) * quant_system.Ψ
        Φ[i] = Φ[i] / norm(Φ[i])
        c_1[i] = Φ[i].data[1]
        c_2[i] = Φ[i].data[2]
        r_t[i] = (dm(Φ[i]).data)[1, 2] / (dm(Φ[1]).data)[1, 2]
    end
    entan = real(entanglement_entropy(quant_system.Ψ, [2])) / log(2)
    # Your code...
    fig = figure(figsize=(15, 10))
    axs = Array{Any}(undef, 3, 2)
    # Create the subplots
    axs[1] = subplot2grid((2, 2), (0, 0), rowspan=2, polar=true)
    axs[2] = subplot2grid((2, 2), (0, 1), polar=true)
    axs[3] = subplot2grid((2, 2), (1, 1), polar=true)

        # Create the polar plots
    # Your code...

    # Create the polar plots
    axs[1].plot(angle.(r_t), abs.(r_t), label="r_t", linewidth=1.0)
    axs[1].set_title(L"r_t")
    axs[1].grid(linewidth=0.5)  # Adjust the grid lines

    axs[2].plot(angle.(c_1), abs.(c_1), label="c_1", linewidth=1.0)
    axs[2].set_title(L"c_1")
    axs[2].grid(linewidth=0.5)  # Adjust the grid lines

    axs[3].plot(angle.(c_2), abs.(c_2), label="c_2", linewidth=1.0)
    axs[3].set_title(L"c_2")
    axs[3].grid(linewidth=0.5)  # Adjust the grid lines

    fig.suptitle("SpinBoson \n EnergyDiff: $(abs(quant_system.EΨ - Ec)); Entanglement: $(entan)")
    # Save the figure
    savefig("data/SpinBoson/DecohFact/polar_r_t_index_$index.png")
    close(fig)
end