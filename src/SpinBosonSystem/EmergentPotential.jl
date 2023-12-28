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
#[1, 6, 10, 100, 600, 900]
φ = (1 + √5)/2 
α = 600
α = sqrt(α) * exp(im * φ)
quant_system = BosonQuantumSystem(Hs, Hc, V/abs(α));
function χ(t::Float64)
    #return quant_system.HC_EIG_V[100] * exp(-im * quant_system.HC_EIG_E[100] * t)
    return coherentstate(clck_basis, exp(-im * Ω * t)*α)
end
T = LinRange(0, 1, 1000)
Ec = real(expect(Hc, χ(0.0)))
mean_NORM = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local sigz = Vector{Float64}(undef, length(T))
    local VGLOB = V*projector(quant_system.Ψ)
    VGLOB = VGLOB + dagger(VGLOB)
    NORM_lis = Vector{Float64}(undef, length(T))
    @threads for i in eachindex(T)
        Χt = χ(T[i])
        NORM = real(expect(tensor(identityoperator(Hs), projector(Χt)), quant_system.Ψ))
        local V_ = tensor(identityoperator(Hs), dagger(Χt)) * VGLOB * tensor(identityoperator(Hs), Χt) / abs(α) ## Note the normalization factor
        if  abs(NORM) == 0
            println("NORM = 0")
        else
            V_ = V_ / NORM
        end
        NORM_lis[i] = NORM
        sigz[i] = real(tr(sigmax(spin_basis) * V_) * 0.5)
    end
    mean_NORM[index] = abs(mean(NORM_lis))
    if (quant_system.GLOB_EIG_E[index] > 4000.0) && (quant_system.GLOB_EIG_E[index] < 6000.0) || (quant_system.GLOB_EIG_E[index] > 10000.0) && (quant_system.GLOB_EIG_E[index] < 11000.0)
        fig, ax = subplots(1, 1, figsize=(10, 7), sharex=true)
    fig.subplots_adjust(hspace=0.5)
    ax.plot(T, sigz, label=L"coeff_{\sigma _x}", linestyle="--",alpha=0.4, linewidth=0.8)
    ax.legend()
    ax.set_xlabel(L"t")
    ax.set_ylabel(L"\text{Coeff of } \sigma_x")
    ax.grid(true, linestyle=":")
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    fig.suptitle("SpinBoson with linear coupling" * L"\alpha \sigma)^2" *" at CutOff N = $N_cutoff" 
        * "\n S = $(round(entan, digits=3)) E = $(round(quant_system.GLOB_EIG_E[index], digits=4)) Ec = $(round(Ec, digits=4))")
    savefig("data/SpinBoson/alpha2=600/Potential/index_$(index).png", dpi=300)
    close(fig)
    end
end
#Create a new figure
figure(figsize=(7, 7))
mean_NORM = replace(mean_NORM, NaN => 0)
mean_NORM = replace(mean_NORM, Inf => 0)
title("SpinBoson with linear coupling at \n CutOff N = $N_cutoff")
xlabel(L"E_{glob}")
ylabel(L"mean(NORM)")
# Plot the 2D histogram
#hist2D(EnergyDiff, over_var, bins=(300, 300), cmap="plasma", cmin=1)
scatter(quant_system.GLOB_EIG_E, mean_NORM, s=1.0, alpha=0.5, color="blue")
axvline(Ec, color="red", linestyle="--", linewidth=0.8)
grid(true, linestyle=":")
PyPlot.savefig("data/SpinBoson/alpha2=600/scatter_meanPotanyialCoeff.png", dpi=600)
println("done")