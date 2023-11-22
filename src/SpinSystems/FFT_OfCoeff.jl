include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
PyPlot.rc("axes", grid=true)
using LaTeXStrings
using Base.Threads
using Statistics
using DelimitedFiles
using FFTW # for fft
using Peaks
#= 
1. Import the required Hamiltonians 
2. Don't forgert to update the titles and labels of the plots. 
=#
include("hamiltonian/powerLawCoupling.jl")
quant_system = SpinQuantSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;

function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end

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

outfile = "data/energy_degeneracy_$N.txt"
writedlm(outfile, degeneracy, '\t')

N_samp = 2^11 - 1
t0 = 0
tmax = 4pi
Ts = tmax / N_samp
# time coordinate
T = t0:Ts:tmax

freqs = fftfreq(length(T), 1.0 / Ts) |> fftshift; ## Equivalent to fftshift(fftfreq(N, 1.0/Ts))
KetList = Array{Ket}(undef, length(quant_system.GLOB_EIG_E), length(T))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local c1 = Vector{ComplexF64}(undef, length(T))
    local c2 = Vector{ComplexF64}(undef, length(T))
    @threads for i in eachindex(T)
        t = T[i]
        ϕ = tensor(identityoperator(Hs), dagger(χ(t, quant_system.EΨ))) * quant_system.Ψ
        ϕ = ϕ / norm(ϕ)
        KetList[index, i] = ϕ
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    ene = quant_system.EΨ
    local c1_var = (abs2.(c1) .- mean(abs2.(c1)))
    local c2_var = (abs2.(c2) .- mean(abs2.(c2)))
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
    max_indices, max_powers = findmaxima(abs.(fft_c1))
    axs[1, 2].semilogy(freqs, abs.(fft_c1), linewidth=1.2)
    max_freqs = freqs[max_indices]
    for (max_freq, max_power) in zip(max_freqs, max_powers)
        axs[1, 2].plot(max_freq, max_power, marker="o", color="red")
        axs[1, 2].annotate(
            string(round(max_freq, digits=2)),
            (max_freq, max_power),
            textcoords="offset points",
            xytext=(0, 10),  # Adjust this value to move the text vertically
            rotation=90,
            ha="center",
            fontsize=9,
        )
    end
    #axs[1, 2].plot(freqs, log.(abs.(fft_c1)), linewidth=1.2)
    axs[1, 2].set_title(L"log|FFT(|c_1|^2)|", fontsize=25)
    axs[1, 2].set_xlabel("Freq(Hz)", fontsize=25)
    axs[1, 2].set_ylabel(L"log|FFT(|c_1|^2)|", fontsize=25)
    axs[1, 2].set_xlim(0, 20)
    axs[1, 2].tick_params(axis="x", labelsize=18)  # Increase x-ticks font size
    axs[1, 2].tick_params(axis="y", labelsize=18)  # Increase y-ticks font size
    ##======##======
    axs[2, 1].plot(T, c1_var, linewidth=1.2)
    axs[2, 1].set_title(L"std(|c_1|^2) = (|c_1|^2 - \overline{|c_1|^2})", fontsize=25)
    axs[2, 1].set_xlabel("T", fontsize=25)
    axs[2, 1].set_ylabel(L"std(|c_1|^2)", fontsize=25)
    axs[2, 1].tick_params(axis="x", labelsize=14)  # Increase x-ticks font size
    axs[2, 1].tick_params(axis="y", labelsize=14)  # Increase y-ticks font size
    ##======##======
    axs[2, 2].semilogy(freqs, abs.(Y1), linewidth=1.2)
    max_indices, max_powers = findmaxima(abs.(Y1))
    max_freqs = freqs[max_indices]
    for (max_freq, max_power) in zip(max_freqs, max_powers)
        axs[2, 2].plot(max_freq, max_power, marker="o", color="red")
        axs[2, 2].annotate(
            string(round(max_freq, digits=2)),
            (max_freq, max_power),
            textcoords="offset points",
            xytext=(0, 10),  # Adjust this value to move the text vertically
            rotation=90,
            ha="center",
            va="bottom",  # Adjust this value to change the vertical alignment
            fontsize=9,
        )
    end
    #axs[2, 2].plot(freqs, log.(abs.(Y1)), linewidth=1.2)
    axs[2, 2].set_title(L"log|FFT(std(|c_1|^2))|", fontsize=25)
    axs[2, 2].set_xlabel("Freq(Hz)", fontsize=25)
    axs[2, 2].set_ylabel(L"log|FFT(std(|c_1|^2)))|", fontsize=25)
    axs[2, 2].set_xlim(0, 20)
    axs[2, 2].tick_params(axis="x", labelsize=18)  # Increase x-ticks font size
    axs[2, 2].tick_params(axis="y", labelsize=18)  # Increase y-ticks font size

    fig.suptitle("PowerLaw Coupling; $N spins and Gamma $γ \n Energy: $(ene); Entanglement: $(entan)", fontsize=30)
    PyPlot.savefig("data/index_$index.svg",)
    PyPlot.close()
end

OVERLAP = Array{Float64}(undef, length(quant_system.GLOB_EIG_E), length(quant_system.GLOB_EIG_E))
for i in eachindex(quant_system.GLOB_EIG_E)
    for j in eachindex(quant_system.GLOB_EIG_E)
        overlap = Vector{Float64}(undef, length(T))
        for k in eachindex(T)
            overlap[k] = abs2(dagger(KetList[i, k]) * KetList[j, k])
        end
        OVERLAP[i, j] = mean(overlap)
    end
end
figure(figsize=(10, 10))
imshow(OVERLAP, cmap="plasma")

# Add a color bar
colorbar()

# Add labels and title
xlabel("Index j")
ylabel("Index i")
title("Overlap Matrix for PowerLaw Coupling; \n $N spins and Gamma $γ")

# Save the figure
savefig("data/overlap_matrix_$N.svg")

# Close the figure
close()