include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Base.Threads
using Statistics
using PyCall
using LaTeXStrings
using PyPlot
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
include("hamiltonian/SpinBosonQuad.jl")
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
H_semi(t, ψ) = Hs + g*sigmax(spin_basis)*(sqrt(2)*abs(α)*cos(Ω*t- φ))^2
Sys_energy_RQM = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
Sys_energy_semi = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
Sys_energy_RQM_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
Sys_energy_semi_var = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local overlap = Vector{ComplexF64}(undef, length(T_))
    local psi_0 =(tensor(identityoperator(Hs), dagger(χ(0.0))) * quant_system.Ψ)
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    c1_semi = Vector{Float64}(undef, length(T_))
    c2_semi = Vector{Float64}(undef, length(T_))
    c1_rqm  = Vector{Float64}(undef, length(T)) 
    c2_rqm  = Vector{Float64}(undef, length(T))
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0
            c1_semi[i] = abs2(ψt_.data[1])
            c2_semi[i] = abs2(ψt_.data[2])
        else
            ψt_ = normalize(ψt_)
            psi_semi_t[i] = ψt_
            c1_semi[i] = abs2(ψt_.data[1])
            c2_semi[i] = abs2(ψt_.data[2])
        end
    end
    @threads for i in eachindex(T)
        Χt = χ(T[i])
        ϕ = tensor(identityoperator(Hs), dagger(Χt)) * quant_system.Ψ
        if abs(norm(ϕ)) == 0
            c1_rqm[i] = abs2(ϕ.data[1])
            c2_rqm[i] = abs2(ϕ.data[2])
        else
            normalize!(ϕ)
            c1_rqm[i] = abs2(ϕ.data[1])
            c2_rqm[i] = abs2(ϕ.data[2])
        end
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    over_mean[index] = mean(overlap)
    over_var[index] = var(overlap)
    sys_energy_rqm = c1_rqm - c2_rqm
    sys_energy_semi = c1_semi - c2_semi
    Sys_energy_RQM[index] = 1/2*ħ*ω*mean(sys_energy_rqm)
    Sys_energy_semi[index] = 1/2*ħ*ω*mean(sys_energy_semi)
    Sys_energy_RQM_var[index] = 1/2*ħ*ω*var(sys_energy_rqm)
    Sys_energy_semi_var[index] = 1/2*ħ*ω*var(sys_energy_semi)
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
fig, ax = subplots(1, 2, figsize=set_size("thesis", 1, (1, 2)),)

ax[1].scatter(quant_system.GLOB_EIG_E, over_mean, s=0.1, alpha=0.5)
ax[1].set_xlabel(L"E_{glob}", fontsize=8)
ax[1].set_ylabel(L"mean(|\langle \psi(t)|\psi_{semi}(t)\rangle|)", fontsize=8)
ax[1].set_ylim(0.5, 1)
ax[1].grid(true, linestyle=":")
ax[1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1].axvline(Ec + 2500.0, color="green", linestyle="--", linewidth=0.5)
ax[1].axvline(Ec - 2500.0, color="green", linestyle="--", linewidth=0.5)
ax[1].tick_params(axis="both", which="major", labelsize=8)
ax[2].scatter(quant_system.GLOB_EIG_E, over_var, s=0.1, alpha=0.5)
ax[2].set_xlabel(L"E_{glob}", fontsize=8)
ax[2].set_ylabel(L"var(|\langle \psi(t)|\psi_{semi}(t)\rangle|)", fontsize=8)
ax[2].grid(true, linestyle=":")
ax[2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2].tick_params(axis="both", which="major", labelsize=8)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.suptitle("SpinBoson with linear potential at  CutOff N = $N_cutoff", fontsize=10)
PyPlot.savefig("data/SpinBoson/alpha2=600/Overlap.pdf", dpi=600, bbox_inches="tight")

fig, ax = subplots(2, 2, figsize=set_size("thesis", 1, (2, 2)),)
ax[1, 1].scatter(quant_system.GLOB_EIG_E, Sys_energy_RQM, s=0.1, alpha=0.5)
ax[1, 1].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 1].set_ylabel(L"\frac{\hbar \omega}{2}\langle E_{sys} \rangle _{RQM}", fontsize=8)
ax[1, 1].grid(true, linestyle=":")
ax[1, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 1].tick_params(axis="both", which="major", labelsize=8)
ax[1, 2].scatter(quant_system.GLOB_EIG_E, Sys_energy_semi, s=0.1, alpha=0.5)
ax[1, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 2].set_ylabel(L"\frac{\hbar \omega}{2}\mathrm{var}\langle E_{sys} \rangle _{semi}", fontsize=8)
ax[1, 2].grid(true, linestyle=":")
ax[1, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 2].tick_params(axis="both", which="major", labelsize=8)
ax[2, 1].scatter(quant_system.GLOB_EIG_E, Sys_energy_RQM_var, s=0.1, alpha=0.5)
ax[2, 1].set_xlabel(L"E_{glob}")
ax[2, 1].set_ylabel(L"\frac{\hbar \omega}{2}\langle E_{sys} \rangle _{RQM}", fontsize=8)
ax[2, 1].grid(true, linestyle=":")
ax[2, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 1].tick_params(axis="both", which="major", labelsize=8)
ax[2, 2].scatter(quant_system.GLOB_EIG_E, Sys_energy_semi_var, s=0.1, alpha=0.5)
ax[2, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[2, 2].set_ylabel(L"\frac{\hbar \omega}{2}\mathrm{var}\langle E_{sys} \rangle _{semi}", fontsize=8)
ax[2, 2].grid(true, linestyle=":")
ax[2, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 2].tick_params(axis="both", which="major", labelsize=8)
fig.suptitle("SpinBoson with linear potential at  CutOff N = $N_cutoff", fontsize=10)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
PyPlot.savefig("data/SpinBoson/alpha2=600/SystemEnergy.pdf", dpi=600, bbox_inches="tight")