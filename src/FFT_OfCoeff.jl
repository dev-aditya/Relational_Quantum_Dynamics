include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using PyPlot
PyPlot.rc("axes", grid=true)
using LaTeXStrings
using Base.Threads
using Statistics
using DelimitedFiles
using FFTW # for fft
#= using PyCall
mpl = pyimport("matplotlib.pyplot")
mpl.style.use("seaborn") =#

#= 
1. Import the required Hamiltonians 
2. Don't forgert to update the titles and labels of the plots. 
=#
include("hamiltonian/highlyCoupled_spinx_spinx.jl")

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

outfile = "data/energy_degeneracy_$N-sigx_sigx_coupling.txt"
writedlm(outfile, degeneracy, '\t')

T = Base._linspace(0.0, 2 * π, 1000);
freqs = fftfreq(length(T), T[2] - T[1]) |> fftshift;
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
    c1_var = (abs2.(c1) .- mean(abs2.(c1)))
    c2_var = (abs2.(c2) .- mean(abs2.(c2)))
    Y1 = fft(c1_var) |> fftshift
    entan = real(entanglement_entropy(quant_system.Ψ, [i for i = 2:N])) / log(2)
    fig, axs = subplots(2, 2, figsize=(45, 27),)
    ##==============
    axs[1, 1].plot(T, abs2.(c1), label=L"|c_1|^2", color="red",)
    axs[1, 1].plot(T, abs2.(c2), label=L"|c_2|^2", color="blue",)
    axs[1, 1].set_title("Coefficients together", fontsize=25)
    axs[1, 1].set_xlabel("T", fontsize=25)
    axs[1, 1].set_ylabel(L"|c_1|^2, |c_2|^2", fontsize=25)
    axs[1, 1].tick_params(axis="x", labelsize=18)  # Increase x-ticks font size
    axs[1, 1].tick_params(axis="y", labelsize=18)  # Increase y-ticks font size
    axs[1, 1].legend(fontsize=25)
    ##======##======
    fft_c1 = fft(abs2.(c1)) |> fftshift
    axs[1, 2].plot(freqs, log.(abs.(fft_c1)), linewidth=1.2)
    axs[1, 2].set_title(L"log|FFT(|c_1|^2)|", fontsize=25)
    axs[1, 2].set_xlabel("Freq(Hz)", fontsize=25)
    axs[1, 2].set_ylabel(L"log|FFT(|c_1|^2)|", fontsize=25)
    axs[1, 2].set_xlim(-0.002, 0.002)
    axs[1, 2].tick_params(axis="x", labelsize=18)  # Increase x-ticks font size
    axs[1, 2].tick_params(axis="y", labelsize=18)  # Increase y-ticks font size
    ##======##======
    axs[2, 1].plot(T, c1_var, linewidth=1.2)
    axs[2, 1].set_title(L"var(|c_1|^2) = (|c_1|^2 - \overline{|c_1|^2})", fontsize=25)
    axs[2, 1].set_xlabel("T", fontsize=25)
    axs[2, 1].set_ylabel(L"var(|c_1|^2)", fontsize=25)
    axs[2, 1].tick_params(axis="x", labelsize=14)  # Increase x-ticks font size
    axs[2, 1].tick_params(axis="y", labelsize=14)  # Increase y-ticks font size
    ##======##======
    axs[2, 2].plot(freqs, log.(abs.(Y1)), linewidth=1.2)
    axs[2, 2].set_title(L"log|FFT(var(|c_1|^2))|", fontsize=25)
    axs[2, 2].set_xlabel("Freq(Hz)", fontsize=25)
    axs[2, 2].set_ylabel(L"log|FFT(var(|c_1|^2)))|", fontsize=25)
    axs[1, 2].set_xlim(-0.002, 0.002)
    axs[2, 2].tick_params(axis="x", labelsize=18)  # Increase x-ticks font size
    axs[2, 2].tick_params(axis="y", labelsize=18)  # Increase y-ticks font size

    fig.suptitle("Sigx-Sigx Coupling with $N spins\n Energy: $(ene); Entanglement: $(entan)", fontsize=30)
    PyPlot.savefig("data/index_$index-sigx-sigx.svg",)
    PyPlot.close()
end