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

T_ = LinRange(0, 1, 1000)
## Semiclassical Hamiltonian for quant_system
over_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
H_semi(t, ψ) = Hs + λ*sqrt(2)*abs(α)*cos(Ω*t- φ)*xsys
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local overlap = Vector{ComplexF64}(undef, length(T_))
    local ζ = Vector{ComplexF64}(undef, length(T_))
    local psi_0 = tensor(identityoperator(Hs), dagger(χ(0.0))) * quant_system.Ψ
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    c1_semi = Vector{Float64}(undef, length(T_))
    c1_rqm  = Vector{Float64}(undef, length(T))
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
        PΧt = tensor(identityoperator(Hs), projector(Χt))
        NORM_ = expect(PΧt,  quant_system.Ψ)
        if abs(norm(ϕ)) == 0
            c1_rqm[i] = abs2(ϕ.data[1])
        else
            normalize!(ϕ)
            c1_rqm[i] = abs2(ϕ.data[1])
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
        if abs(NORM_) == 0
            ζ[i] = expect(V*PΧt, quant_system.Ψ)
        else
            ζ[i] = expect(V*PΧt, quant_system.Ψ) / NORM_
        end
    end
    over_var[index] = mean(abs.(overlap))
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    fig, ax = subplots(3, 1, figsize=(10, 7))
    ax[1].plot(T, c1_semi, label=L"|c_1({semi})|^2", color="black")
    ax[1].plot(T, c1_rqm, label=L"|c_1({RQM})|^2", color="red")
    ax[1].legend()
    ax[1].set_xlabel(L"t")
    ax[1].set_ylabel(L"|c_1|^2")
    ax[2].plot(T, abs.(overlap))
    ax[2].set_xlabel(L"t")
    ax[2].set_ylabel(L"|⟨ψ(t)|ψ_{semi}(t)⟩|")
    ax[3].plot(T, real.(ζ), label=L"Re(ζ)")
    ax[3].plot(T, imag.(ζ), label=L"Im(ζ)")
    ax[3].legend()
    ax[3].set_xlabel(L"t")
    ax[3].set_ylabel(L"ζ")
    fig.suptitle("CoupledHarmonicOscillator, CutOff at N = $N \n Energy = $(quant_system.EΨ) , Entaglement = $entan")
    savefig("data/ClassVsQuant/index_$(index).png", dpi=300)
    close(fig)
end
#Create a new figure
figure(figsize=(6, 6))
over_var = over_var
over_var = replace(over_var, NaN => 0)
over_var = replace(over_var, Inf => 0)
# Plot the 2D histogram
hist2D(quant_system.GLOB_EIG_E, over_var, bins=(300, 300), cmap="plasma", cmin=1)
colorbar(orientation="horizontal")

ylabel(L"mean(|⟨ψ(t)|ψ_{semi}(t)⟩|)")
xlabel("Energy")
grid(true)
title("CoupledHarmonicOscillator")
PyPlot.savefig("data/Overlap_mean_dist.png", dpi=300)
println("done")
