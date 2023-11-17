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

include("hamiltonian/powerLawCoupling.jl")
quant_system = SpinQuantSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;

function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end

Ket0 = spinup(b)
Ket1 = spindown(b)
N_samp = 2^14 - 1
t0 = 0
tmax = 6pi
Ts = tmax / N_samp
# time coordinate
global T = t0:Ts:tmax
global freqs = fftfreq(length(T), 1.0 / Ts) |> fftshift
for index in eachindex(quant_system.GLOB_EIG_E)
    local r_t = Vector{ComplexF64}(undef, length(T))
    UpdateIndex(quant_system, index)
    energy = quant_system.GLOB_EIG_E[index]
    local a_ = tensor(dagger(Ket0), identityoperator(Hc)) * quant_system.Ψ
    local b_ = tensor(dagger(Ket1), identityoperator(Hc)) * quant_system.Ψ
    Φ = Vector{Ket}(undef, length(T))
    @threads for i in eachindex(T)
        Φ[i] = tensor(identityoperator(Hs), dagger(χ(T[i], quant_system.EΨ))) * quant_system.Ψ
        Φ[i] = Φ[i] / norm(Φ[i])
        αt = dagger(Ket0) * Φ[i]
        βt = dagger(Ket1) * Φ[i]
        r_t[i] = (dagger(b_) * a_)/(conj(βt) *αt)
    end
    # Create a new figure
    fig, axs = subplots(4, 1, figsize=(20, 10))
    axs[1].semilogy(T, abs2.(r_t) , label=L"|r(t)|^2", linewidth=1.0)
    axs[1].set_ylabel(L"|r(t)|^2")
    axs[1].grid(true)
    axs[1].legend()

    axs[2].semilogy(T, abs2.(real.(r_t)), label=L"Re(r(t))^2", linewidth=1.0)   
    axs[2].semilogy(T, abs2.(imag.(r_t)), label=L"Im(r(t))^2", linewidth=1.0)
    axs[2].grid(true)   
    axs[2].legend()

    axs[3].plot(T, abs2.([Φ[i].data[1] for i in eachindex(Φ)]), label=L"|c_1|^2", linewidth=1.0)
    axs[3].plot(T, abs2.([Φ[i].data[2] for i in eachindex(Φ)]), label=L"|c_2|^2", linewidth=1.0)
    axs[3].set_ylabel(L"|c|^2")
    axs[3].grid(true)
    axs[3].set_xlabel(L"t")
    axs[3].legend()

    rt_ifft = fft(r_t) |> fftshift
    axs[4].plot(freqs, abs2.(rt_ifft), label=L"\eta|E|", linewidth=1.0)
    axs[4].set_ylabel(L"|E|^2")
    axs[4].set_xlabel(L"f")
    axs[4].legend()
    axs[4].set_xlim(-2, 2)

    fig.suptitle("Eigenindex $index: Energy = $energy \n powerLawCoupling with γ = $γ and l = $l")
    # Save the figure
    savefig("data/r_t_index_$index.png")
end