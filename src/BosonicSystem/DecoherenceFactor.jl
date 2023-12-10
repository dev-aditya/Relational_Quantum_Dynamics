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

include("hamiltonian/CoupledHarmonicOscillator.jl")
quant_system = BosonQuantumSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;
α = (1 + √5)/2
function χ(t::Float64, E::Float64)
    return coherentstate(bclc, exp(-im*E*t)*exp(-im * Ω * t)*α)
end

N_samp = 2^14 - 1
t0 = 0
tmax = 2pi
Ts = tmax / N_samp
# time coordinate
global T = t0:Ts:tmax
global freqs = fftfreq(length(T), 1.0 / Ts) |> fftshift
for index in eachindex(quant_system.GLOB_EIG_E)
    local r_t_rel = Vector{ComplexF64}(undef, length(T))
    local r_t = Vector{ComplexF64}(undef, length(T))
    UpdateIndex(quant_system, index)
    energy = quant_system.GLOB_EIG_E[index]
    Φ = Vector{Ket}(undef, length(T))
    for i in eachindex(T)
        Φ[i] = tensor(identityoperator(Hs), dagger(χ(T[i], quant_system.EΨ))) * quant_system.Ψ
        Φ[i] = Φ[i] / norm(Φ[i])
        r_t[i] = (dm(Φ[i]).data)[1, 2] / (dm(Φ[1]).data)[1, 2]
    end
    entan = real(entanglement_entropy(quant_system.Ψ, [2])) / log(2)
    # Create a new figure
    r_t_abs2 = abs2.(r_t)
    fig = figure(figsize=(15, 10))
    axs = Array{Any}(undef, 3, 2)

    # Create the subplots
    axs[1] = subplot2grid((3, 2), (0, 0))
    axs[2] = subplot2grid((3, 2), (0, 1))
    axs[3] = subplot2grid((3, 2), (1, 0), colspan=2)
    axs[4] = subplot2grid((3, 2), (2, 0), colspan=2)

    axs[1].plot(T, r_t_abs2 , label=L"|r(t)|^2", linewidth=1.0)
    #axs[1].semilogy(T, abs2.(r_t_rel) , label=L"|r(t)_{rel}|^2", linewidth=1.0)
    axs[1].set_ylabel(L"|r(t)|^2")
    axs[1].grid(true)
    axs[1].legend()

    axs[2].plot(T, abs2.([Φ[i].data[1] for i in eachindex(Φ)]), label=L"|c_1|^2", linewidth=1.0)
    axs[2].plot(T, abs2.([Φ[i].data[2] for i in eachindex(Φ)]), label=L"|c_2|^2", linewidth=1.0)
    axs[2].set_ylabel(L"|c|^2")
    axs[2].grid(true)
    axs[2].set_xlabel(L"t")
    axs[2].legend()

    for time in 0:pi/2:tmax
        axs[1].axvline(x=time, color="red", linestyle="--")
        axs[2].axvline(x=time, color="red", linestyle="--")
    end

    rt_ifft = fft(r_t_abs2 .- mean(r_t_abs2)) |> fftshift
    axs[3].plot(freqs, abs2.(rt_ifft), label=L"\eta|E|", linewidth=1.0)
    max_indices, max_powers = findmaxima(abs2.(rt_ifft))
    max_freqs = freqs[max_indices]
    for (max_freq, max_power) in zip(max_freqs, max_powers)
        axs[3].plot(max_freq, max_power, marker="o", color="red")
        axs[3].annotate(
            string(round(max_freq, digits=2)),
            (max_freq, max_power),
            textcoords="offset points",
            xytext=(0, 10),  # Adjust this value to move the text vertically
            rotation=90,
            ha="center",
            va="bottom",  # Adjust this value to change the vertical alignment
        )
    end
    axs[3].set_ylabel(L"FFT(|r(t)|^2)")
    axs[3].set_xlabel(L"f")
    axs[3].legend()
    axs[3].set_xlim(0, 10)

    c1 = [Φ[i].data[1] for i in eachindex(Φ)]
    c1_fft = fft(abs2.(c1) .- mean(abs2.(c1))) |> fftshift
    axs[4].plot(freqs, abs2.(c1_fft), label=L"FFT(|c_1|^2)", linewidth=1.0)
    max_indices, max_powers = findmaxima(abs2.(c1_fft))
    max_freqs = freqs[max_indices]
    for (max_freq, max_power) in zip(max_freqs, max_powers)
        axs[4].plot(max_freq, max_power, marker="o", color="red")
        axs[4].annotate(
            string(round(max_freq, digits=2)),
            (max_freq, max_power),
            textcoords="offset points",
            xytext=(0, 10),  # Adjust this value to move the text vertically
            rotation=90,
            ha="center",
            va="bottom",  # Adjust this value to change the vertical alignment
        )
    end
    axs[4].set_ylabel(L"FFT(|c_1|^2)")
    axs[4].set_xlabel(L"f")
    axs[4].legend()
    axs[4].set_xlim(0, 10)

    fig.suptitle("Coupled Harmonic Oscillator \n Energy: $(quant_system.EΨ); Entanglement: $(entan)")
    # Save the figure
    savefig("data/DecohFact/r_t_index_$index.png")
    close(fig)
end