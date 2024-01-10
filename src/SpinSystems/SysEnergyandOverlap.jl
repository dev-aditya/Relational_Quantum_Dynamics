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
include("hamiltonian/powerLawCoupling.jl")
quant_system = SpinQuantSystem(Hs, Hc, V);

θ, ϕ = π/4, π/6
function χ(t::Float64)
    #return quant_system.HC_EIG_V[100] * exp(-im * quant_system.HC_EIG_E[100] * t)
    return tensor([coherentspinstate(b, θ, (ϕ + t)) for i in 2:N]...)
end
T_ = LinRange(0, 3π, 3000)
Ec = real(expect(Hc, χ(0.0)))
## Semiclassical Hamiltonian for quant_system
H_semi(t, ψ) = Hs + SIGX * sum([1/abs(l*min(abs(1 - i), N - abs(1 - i)))^γ * (0.5*sin(θ)*cos(ϕ + t)) for i in 2:N])
E_semi_mean  = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_semi_var   = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_rqm_mean   = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
E_rqm_var    = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
over_mean    = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
over_var     = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    println("index = $index")
    local Ψ = quant_system.GLOB_EIG_V[index]
    local overlap = Vector{Float64}(undef, length(T_))
    local psi_0 = normalize!((tensor(one(Hs), dagger(χ(0.0))) * Ψ))
    T, psi_semi_t = timeevolution.schroedinger_dynamic(T_, psi_0, H_semi)
    local V_Ψ = V * Ψ
    @threads for i in eachindex(T_)
        ψt_ = psi_semi_t[i]
        if abs(norm(ψt_)) == 0.0
            continue
        else
            normalize!(ψt_)
            psi_semi_t[i] = ψt_
        end
    end
    local E_semi = real.([expect(H_semi(t, 0), ψ) for (t, ψ) in zip(T_, psi_semi_t)])
    local E_RQM = Vector{Float64}(undef, length(T_))
    @threads for i in eachindex(T)
        cond_proj_t = tensor(one(Hs), χ(T[i]))
        ϕ =  dagger(cond_proj_t) * Ψ
        if abs(norm(ϕ)) == 0.0
            continue
        else
            ϕ = normalize!(ϕ)
        end
        #E_RQM[i] = tr(quant_system.H * tensor(one(Hs), projector(Χt)) * projector(Ψ)) / tr(tensor(one(Hs), projector(Χt)) * projector(Ψ))    
        V_ϕ = projector(dagger(cond_proj_t) * V_Ψ, dagger(ϕ)) 
        V_ϕ = V_ϕ + dagger(V_ϕ)
        E_RQM[i] = real(expect(Hs + V_ϕ, ϕ))
        overlap[i] = abs(dagger(ϕ) * psi_semi_t[i])
    end
    E_semi_mean[index] = mean(E_semi)
    E_semi_var[index] = var(E_semi)
    E_rqm_mean[index] = mean(E_RQM)
    E_rqm_var[index] = var(E_RQM)
    over_mean[index] = mean(overlap)
    over_var[index] = var(overlap)
    #save("data/SpinSpin/alpha2=600/Potential/E_semi_$index.jld", "data", E_semi)
    #save("data/SpinSpin/alpha2=600/Potential/E_RQM_$index.jld", "data", E_RQM)
end

fig, ax = subplots(3, 2, figsize=set_size("thesis", 2, (3, 2)),)
ax[1, 1].scatter(quant_system.GLOB_EIG_E, E_rqm_mean, s=0.1, alpha=0.5)
ax[1, 1].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 1].set_ylabel(L"\langle H_{RQM} \rangle", fontsize=8)
ax[1, 1].grid(true, linestyle=":")
ax[1, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 1].tick_params(axis="both", which="major", labelsize=8)
ax[1, 2].scatter(quant_system.GLOB_EIG_E, E_rqm_var, s=0.1, alpha=0.5)
ax[1, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[1, 2].set_ylabel(L"\mathrm{var}\langle H_{RQM} \rangle", fontsize=8)
ax[1, 2].grid(true, linestyle=":")
ax[1, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[1, 2].tick_params(axis="both", which="major", labelsize=8)
ax[2, 1].scatter(quant_system.GLOB_EIG_E, E_semi_mean, s=0.1, alpha=0.5)
ax[2, 1].set_xlabel(L"E_{glob}")
ax[2, 1].set_ylabel(L"\langle H_{semi} \rangle", fontsize=8)
ax[2, 1].grid(true, linestyle=":")
ax[2, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 1].tick_params(axis="both", which="major", labelsize=8)
ax[2, 2].scatter(quant_system.GLOB_EIG_E, E_semi_var, s=0.1, alpha=0.5)
ax[2, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[2, 2].set_ylabel(L"\mathrm{var}\langle H_{semi} \rangle", fontsize=8)
ax[2, 2].grid(true, linestyle=":")
ax[2, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[2, 2].tick_params(axis="both", which="major", labelsize=8)
ax[3, 1].scatter(quant_system.GLOB_EIG_E, over_mean, s=0.1, alpha=0.5)
ax[3, 1].set_xlabel(L"E_{glob}", fontsize=8)
ax[3, 1].set_ylabel(L"\langle |\langle \psi(t)|\psi_{\mathrm{semi}}(t)\rangle| \rangle", fontsize=8)
ax[3, 1].grid(true, linestyle=":")
ax[3, 1].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[3, 1].tick_params(axis="both", which="major", labelsize=8)
ax[3, 2].scatter(quant_system.GLOB_EIG_E, over_var, s=0.1, alpha=0.5)
ax[3, 2].set_xlabel(L"E_{glob}", fontsize=8)
ax[3, 2].set_ylabel(L"\mathrm{var}(|\langle \psi(t)|\psi_{\mathrm{semi}}(t)\rangle|)", fontsize=8)
ax[3, 2].grid(true, linestyle=":")
ax[3, 2].axvline(Ec, color="red", linestyle="--", linewidth=0.5)
ax[3, 2].tick_params(axis="both", which="major", labelsize=8)
fig.suptitle("SpinSpin with N = $N", fontsize=10)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
PyPlot.savefig("data/SpinSpin/Allplots.pdf", dpi=600, bbox_inches="tight")
close(fig)