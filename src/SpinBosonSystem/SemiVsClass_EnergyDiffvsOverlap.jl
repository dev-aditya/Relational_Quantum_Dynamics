include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Base.Threads
using Statistics
using LaTeXStrings
PyPlot.plt.style.use("seaborn")

include("hamiltonian/SpinBosonLinear.jl")

quant_system = BosonQuantumSystem(Hs, Hc, V);
φ = (1 + √5)/2 
α = quant_system.GLOB_EIG_E[368] / (ħ*Ω) - 1/2
α = sqrt(α) * exp(im * φ)
function χ(t::Float64)
    return coherentstate(clck_basis, exp(-im * Ω * t)*α)
end
T_ = LinRange(0, 1, 5000)
## Semiclassical Hamiltonian for quant_system
over_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
EnergyDiff = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
H_semi(t, ψ) = Hs + g*(sqrt(2)*abs(α)*cos(Ω*t- φ))*sigmax(spin_basis)
#Xclc2 = xclc^2
#H_semi(t, ψ) = Hs + λ*expect(Xclc2, χ(t))*xsys
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local overlap = Vector{ComplexF64}(undef, length(T_))
    local psi_0 = tensor(identityoperator(Hs), dagger(χ(0.0))) * quant_system.Ψ
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    Ec = real(expect(Hc, χ(0.0)))
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0
            psi_semi_t[i] = ψt_
        else
            ψt_ = normalize(ψt_)
            psi_semi_t[i] = ψt_
        end
    end
    @threads for i in eachindex(T)
        Χt = χ(T[i])
        ϕ = tensor(identityoperator(Hs), dagger(Χt)) * quant_system.Ψ
        if abs(norm(ϕ)) == 0
            c1_rqm[i] = abs2(ϕ.data[1])
        else
            normalize!(ϕ)
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    over_var[index] = mean(abs.(overlap))
    EnergyDiff[index] = quant_system.GLOB_EIG_E[index] - Ec
end
#Create a new figure
figure(figsize=(7, 10))
over_var = over_var
over_var = replace(over_var, NaN => 0)
over_var = replace(over_var, Inf => 0)
title("SpinBoson, CutOff at N = $N_cutoff")
xlabel(L"E_{glob} - E_{clock}")
ylabel(L"mean|⟨ψ(t)|ψ_{semi}(t)⟩|")
ylim([0, 1])
# Plot the 2D histogram
hist2D(EnergyDiff, over_var, bins=(300, 300), cmap="plasma", cmin=1)
axvline(0, color="black", linestyle="--")
colorbar(orientation="horizontal")
PyPlot.savefig("data/SpinBoson/EnergyDiffVsOverlap.png", dpi=300)
println("done")