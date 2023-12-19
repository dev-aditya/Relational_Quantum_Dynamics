include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Base.Threads
using Statistics
using PyCall
using LaTeXStrings
pyimport("scienceplots")
mpl = pyimport("matplotlib")
mpl.style.use(["science"])
include("hamiltonian/SpinBosonLinear.jl")

φ = (1 + √5)/2 
α = 1
α = sqrt(α) * exp(im * φ)
quant_system = BosonQuantumSystem(Hs, Hc, V/abs(α));
function χ(t::Float64)
    return coherentstate(clck_basis, exp(-im * Ω * t)*α)
end
T_ = LinRange(0, 1, 1000)
## Semiclassical Hamiltonian for quant_system
over_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
H_semi(t, ψ) = Hs + g*(sqrt(2)*cos(Ω*t- φ))*sigmax(spin_basis)
EnergyDiff = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local overlap = Vector{ComplexF64}(undef, length(T_))
    local psi_0 = tensor(identityoperator(Hs), dagger(χ(0.0))) * quant_system.Ψ
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    c1_semi = Vector{Float64}(undef, length(T_))
    c1_rqm  = Vector{Float64}(undef, length(T)) 
    Ec = real(expect(Hc, χ(0.0)))
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0
            c1_semi[i] = abs2(ψt_.data[1])
        else
            ψt_ = normalize(ψt_)
            psi_semi_t[i] = ψt_
            c1_semi[i] = abs2(ψt_.data[1])
        end
    end
    @threads for i in eachindex(T)
        Χt = χ(T[i])
        ϕ = tensor(identityoperator(Hs), dagger(Χt)) * quant_system.Ψ
        if abs(norm(ϕ)) == 0
            c1_rqm[i] = abs2(ϕ.data[1])
        else
            normalize!(ϕ)
            c1_rqm[i] = abs2(ϕ.data[1])
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    over_var[index] = mean(abs.(overlap))
    EnergyDiff[index] = quant_system.GLOB_EIG_E[index] - Ec
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    fig, ax = subplots(2, 1, figsize=(10, 7), sharex=true)
    fig.subplots_adjust(hspace=0.5)
    ax[1].plot(T, c1_semi, label=L"|c_1({semi})|^2", linestyle="--",alpha=0.4, linewidth=0.8)
    ax[1].plot(T, c1_rqm, label=L"|c_1({RQM})|^2", color="red", linestyle=":", linewidth=0.8)
    ax[1].legend()
    ax[1].set_xlabel(L"t")
    ax[1].set_ylabel(L"|c_1|^2")
    ax[1].grid(true, linestyle=":")
    ax[2].plot(T, abs.(overlap), color="black", linestyle="-.", linewidth=1.0, alpha=0.8)
    ax[2].set_xlabel(L"t")
    ax[2].set_ylabel(L"|\langle \psi(t)|\psi_{semi}(t)\rangle|")
    ax[2].set_title("Overlap")
    ax[1].grid(true, linestyle=":")
    ax[2].grid(true, linestyle=":")
    fig.suptitle("SpinBoson with linear coupling (x/" * L"\alpha" *") at CutOff N = $N_cutoff" 
      * "\n S = $(round(entan, digits=3)) E = $(round(quant_system.GLOB_EIG_E[index], digits=4)) Ec = $(round(Ec, digits=4))")
    savefig("data/SpinBoson/alpha2=1/QuantVsClass/index_$(index).png", dpi=300)
    close(fig)
end
#Create a new figure
figure(figsize=(7, 10))
over_var = replace(over_var, NaN => 0)
over_var = replace(over_var, Inf => 0)
title("SpinBoson with linear coupling (x/" * L"\alpha" *") at \n CutOff N = $N_cutoff")
xlabel(L"E_{glob} - E_{clock}")
ylabel(L"mean(|⟨ψ(t)|ψ_{semi}(t)⟩|)")
# Plot the 2D histogram
hist2D(EnergyDiff, over_var, bins=(300, 300), cmap="plasma", cmin=1)
grid(true, linestyle=":")
colorbar(orientation="horizontal")
PyPlot.savefig("data/SpinBoson/alpha2=1/Overlap_mean_dist.png", dpi=300)
println("done")
