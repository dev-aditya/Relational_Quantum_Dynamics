include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
using Base.Threads
using Statistics
using LaTeXStrings

include("hamiltonian/CoupledHarmonicOscillator.jl")

quant_system = BosonQuantumSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;
α = (1 + √5)/2 + (1 - √5)/2 * im
φ = angle(α)
function χ(t::Float64)
    return coherentstate(bclc, exp(-im * Ω * t)*α)
end

T_ = LinRange(0, 2π, 1000)
## Semiclassical Hamiltonian for quant_system
over_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
H_semi(t, ψ) = Hs + λ*sqrt(2)*abs(α)*cos(Ω*t- φ)*xsys
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    #local overlap = Vector{ComplexF64}(undef, length(T_))
    local psi_0 = tensor(identityoperator(Hs), dagger(χ(0.0))) * quant_system.Ψ
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    c1_semi = Vector{Float64}(undef, length(T_))
    c1_rqm  = Vector{Float64}(undef, length(T))
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0
            println("norm is zero for semi")
            c1_semi[i] = abs2(ψt_.data[1])
        else
            ψt_ = normalize(ψt_)
            psi_semi_t[i] = ψt_
            c1_semi[i] = abs2(ψt_.data[1])
        end
    end
    @threads for i in eachindex(T)
        ϕ = tensor(identityoperator(Hs), dagger(χ(T[i]))) * quant_system.Ψ
        if abs(norm(ϕ)) == 0
            println("norm is zero for RQM")
            c1_rqm[i] = abs2(ϕ.data[1])
        else
            normalize!(ϕ)
            c1_rqm[i] = abs2(ϕ.data[1])
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    figure(figsize=(10, 7))
    plot(T, abs.(overlap))
    xlabel(L"t")
    ylabel(L"|⟨ψ(t)|ψ_{semi}(t)⟩|")
    title("CoupledHarmonicOscillator, \n Energy = $(quant_system.EΨ) and Entropy = $entan ", fontsize=9)
    savefig("data/ClassVsQuant/Overlap_index_$index" * "CoupledHarmonicOscillator" * ".png")
    over_var[index] = mean(overlap)
    figure(figsize=(10, 7))
    plot(T, c1_semi, label=L"|c_1({semi})|^2")
    plot(T, c1_rqm, label=L"|c_1({RQM})|^2")
    legend()
    xlabel(L"t")
    ylabel(L"|c_1|^2")
    title("CoupledHarmonicOscillator, \n Energy = $(quant_system.EΨ) and Entropy = $entan ", fontsize=9)
    savefig("data/ClassVsQuant/index_$index" * "CoupledHarmonicOscillator" * ".png")
    close()
end
# Create a new figure
figure(figsize=(6, 6))
over_var = replace(over_var, NaN => 0)
over_var = replace(over_var, Inf => 0)
# Plot the 2D histogram
hist2D(quant_system.GLOB_EIG_E, over_var, bins=(300, 300), cmap="plasma", cmin=1)
colorbar(orientation="horizontal")
# Set labels and title
ylabel(L"mean(|⟨ψ(t)|ψ_{semi}(t)⟩|)")
xlabel("Energy")
grid(true)
title("CoupledHarmonicOscillator")
PyPlot.savefig("data/Overlap_mean_dist.png", dpi=300)
println("done")
