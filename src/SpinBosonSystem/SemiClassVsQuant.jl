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
quant_system = BosonQuantumSystem(Hs, Hc, V);
function χ(t::Float64)
    #return quant_system.HC_EIG_V[100] * exp(-im * quant_system.HC_EIG_E[100] * t)
    return coherentstate(clck_basis, exp(-im * Ω * t)*α)
end
T_ = LinRange(0, 1, 2000)
Ec = real(expect(Hc, χ(0.0)))
## Semiclassical Hamiltonian for quant_system
over_mean = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
over_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
H_semi(t, ψ) = Hs + g*sigmax(spin_basis)*(sqrt(2)*abs(α)*cos(Ω*t- φ))
Overlap = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    Overlap[index] = real(expect(ptrace(quant_system.GLOB_EIG_V[index], [1]), χ(0.0)))
    local overlap = Vector{ComplexF64}(undef, length(T_))
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
        if abs(norm(ϕ)) == 0
            c1_rqm[i] = abs2(ϕ.data[1])
        else
            normalize!(ϕ)
            c1_rqm[i] = abs2(ϕ.data[1])
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    over_mean[index] = mean(overlap)
    over_var[index] = var(overlap)
    #=
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    fig, ax = subplots(2, 1, figsize=(10, 7), sharex=true)
    fig.subplots_adjust(hspace=0.5)
    ax[1].plot(T, c1_semi, label=L"|c_1({semi})|^2", linestyle="--",alpha=0.4, linewidth=0.5)
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
    fig.suptitle("SpinBoson with linear coupling at CutOff N = $N_cutoff" 
        * "\n S = $(round(entan, digits=3)) E = $(round(quant_system.GLOB_EIG_E[index], digits=4)) Ec = $(round(Ec, digits=4))")
    savefig("data/SpinBoson/alpha2=600/QuantVsClass/index_$(index).png")
    close(fig)
    =#
end
# Create a new figure
fig, ax = subplots(1, 2, figsize=(10, 5))

ax[1].scatter(quant_system.GLOB_EIG_E, over_mean, s=0.1, alpha=0.5)
ax[1].set_title("SpinBoson with linear potential at \n CutOff N = $N_cutoff")
ax[1].set_xlabel(L"E_{glob}")
ax[1].set_ylabel(L"mean(|\langle \psi(t)|\psi_{semi}(t)\rangle|)")
ax[1].set_ylim(0.5, 1)
ax[1].grid(true, linestyle=":")
ax[1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1].axvline(Ec + 2500.0, color="green", linestyle="--", linewidth=0.5)
ax[1].axvline(Ec - 2500.0, color="green", linestyle="--", linewidth=0.5)

ax[2].scatter(quant_system.GLOB_EIG_E, over_var, s=0.1, alpha=0.5)
ax[2].set_title("Variance of Overlap for linear Coupling at \n CutOff N = $N_cutoff")
ax[2].set_xlabel(L"E_{glob}")
ax[2].set_ylabel(L"var(|\langle \psi(t)|\psi_{semi}(t)\rangle|)")
ax[2].grid(true, linestyle=":")
ax[2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
PyPlot.savefig("data/SpinBoson/alpha2=600/Overlap.pdf", dpi=600)
