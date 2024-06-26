include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Base.Threads
using Statistics
using PyCall
using LaTeXStrings
using PyPlot
using JLD
pyimport("scienceplots")
mpl = pyimport("matplotlib")
mpl.style.use(["science"])
function set_size(width, fraction::Int64, subplots::Tuple)
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predefined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis"
        width_pt = 434.90039
    else
        width_pt = width
    end

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5^.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[1] / subplots[2])

    return (fig_width_in, fig_height_in)
end
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
T_ = LinRange(0, π, 2000)
Ec = real(expect(Hc, χ(0.0)))
## Semiclassical Hamiltonian for quant_system
H_semi(t, ψ) = Hs + g*sigmax(spin_basis)*(sqrt(2)*abs(α)*cos(Ω*t- φ))
E_semi_mean = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_semi_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_rqm_mean = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_rqm_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    println("index = $index")
    local Ψ = quant_system.GLOB_EIG_V[index]
    local psi_0 = normalize!((tensor(one(Hs), dagger(χ(0.0))) * Ψ))
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    local V_Ψ = V * Ψ
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0
            continue
        else
            normalize!(ψt_)
            psi_semi_t[i] = ψt_
        end
    end
    local E_semi = real.([expect(H_semi(t, 0.0), ψ) for (t, ψ) in zip(T_, psi_semi_t)])
    local E_RQM = Vector{Float64}(undef, length(T_))
    @threads for i in eachindex(T)
        Χt = χ(T[i])
        cond_proj_t = tensor(one(Hs), χ(T[i]))
        ϕ =  dagger(cond_proj_t) * Ψ
        if abs(norm(ϕ)) == 0
            continue
        else
            normalize!(ϕ)
        end
        #E_RQM[i] = tr(quant_system.H * tensor(one(Hs), projector(Χt)) * projector(Ψ)) / tr(tensor(one(Hs), projector(Χt)) * projector(Ψ))    
        V_ϕ = projector(dagger(cond_proj_t) * V_Ψ, dagger(ϕ)) 
        V_ϕ = V_ϕ + dagger(V_ϕ)
        E_RQM[i] = real(expect(Hs + V_ϕ, ϕ))
    end
    E_semi_mean[index] = mean(E_semi)
    E_semi_var[index] = var(E_semi)
    E_rqm_mean[index] = mean(E_RQM)
    E_rqm_var[index] = var(E_RQM)
    save("data/SpinBoson/alpha2=600/Potential/E_semi_$index.jld", "data", E_semi)
    save("data/SpinBoson/alpha2=600/Potential/E_RQM_$index.jld", "data", E_RQM)
    #=
    fig, ax = subplots(2, 1, figsize=set_size("thesis", 1, (2, 1)), sharex=true)
    fig.subplots_adjust(hspace=0.5)
    ax[1].plot(T, E_semi, label=L"E_{semi}", linewidth=0.5, color="red", linestyle="--", alpha=0.5, )
    ax[1].set_ylabel(L"E", fontsize=8)
    ax[1].grid(true, linestyle=":")
    ax[1].tick_params(axis="both", which="major", labelsize=8)
    ax[1].legend(fontsize=8)

    ax[2].plot(T, E_RQM, label=L"E_{RQM}", linewidth=0.5, color="blue", linestyle="--", alpha=0.5, )
    ax[2].set_ylabel(L"E", fontsize=8)
    ax[2].set_xlabel(L"t", fontsize=8)
    ax[2].grid(true, linestyle=":")
    ax[2].tick_params(axis="both", which="major", labelsize=8)
    ax[2].legend(fontsize=8)
    fig.suptitle("SpinBoson with linear potential at  CutOff N = $N_cutoff", fontsize=10)
    PyPlot.savefig("data/SpinBoson/alpha2=600/Potential/index_$index.png", dpi=300, bbox_inches="tight")
    close(fig)
    =#
end

fig, ax = subplots(2, 2, figsize=set_size("thesis", 2, (2, 2)),)
ax[1, 1].scatter(quant_system.GLOB_EIG_E, E_rqm_mean, s=0.1, alpha=0.5)
ax[1, 1].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 1].set_ylabel(L"\langle E_{RQM} \rangle", fontsize=8)
ax[1, 1].grid(true, linestyle=":")
ax[1, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 1].tick_params(axis="both", which="major", labelsize=8)
ax[1, 2].scatter(quant_system.GLOB_EIG_E, E_rqm_var, s=0.1, alpha=0.5)
ax[1, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 2].set_ylabel(L"\mathrm{var}\langle E_{RQM} \rangle", fontsize=8)
ax[1, 2].grid(true, linestyle=":")
ax[1, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 2].tick_params(axis="both", which="major", labelsize=8)
ax[2, 1].scatter(quant_system.GLOB_EIG_E, E_semi_mean, s=0.1, alpha=0.5)
ax[2, 1].set_xlabel(L"E_{glob}")
ax[2, 1].set_ylabel(L"\langle E_{semi} \rangle", fontsize=8)
ax[2, 1].grid(true, linestyle=":")
ax[2, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 1].tick_params(axis="both", which="major", labelsize=8)
ax[2, 2].scatter(quant_system.GLOB_EIG_E, E_semi_var, s=0.1, alpha=0.5)
ax[2, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[2, 2].set_ylabel(L"\mathrm{var}\langle E_{semi} \rangle", fontsize=8)
ax[2, 2].grid(true, linestyle=":")
ax[2, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 2].tick_params(axis="both", which="major", labelsize=8)
fig.suptitle("SpinBoson with Quad potential at  CutOff N = $N_cutoff", fontsize=10)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
PyPlot.savefig("data/SpinBoson/alpha2=600/SystemCondEnergy.pdf", dpi=600, bbox_inches="tight")
close(fig)