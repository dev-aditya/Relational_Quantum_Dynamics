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
include("hamiltonian/JyaCumModel.jl")
#[1, 6, 10, 100, 600, 900]
φ = (1 + √5)/2 
α = 400
α = sqrt(α) * exp(im * φ)
quant_system = BosonQuantumSystem(Hs, Hc, V);
function χ(t::Float64)
    #return quant_system.HC_EIG_V[100] * exp(-im * quant_system.HC_EIG_E[100] * t)
    return coherentstate(clck_basis, exp(-im * ω0 * t)*α)
end
T_ = LinRange(0, 3π, 3000)
Ec = real(expect(Hc, χ(0.0)))
## Semiclassical Hamiltonian for quant_system
H_semi(t, ψ) = Hs + ħ*g*(((α*exp(-im * ω0 * t))*sigmap(spin_basis)) + conj((α*exp(-im * ω0 * t)))*(sigmam(spin_basis)))
global OPER = sigmax(spin_basis)
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
    local exp_semi = real.(expect(OPER, psi_semi_t))
    local exp_RQM = Vector{Float64}(undef, length(T))
    @threads for i in eachindex(T)
        cond_proj_t = tensor(one(Hs), χ(T[i]))
        ϕ =  dagger(cond_proj_t) * Ψ
        if abs(norm(ϕ)) == 0.0
            continue
        else
            ϕ = normalize!(ϕ)
        end
        exp_RQM[i] = real(expect(OPER, ϕ))
        #E_RQM[i] = tr(quant_system.H * tensor(one(Hs), projector(Χt)) * projector(Ψ)) / tr(tensor(one(Hs), projector(Χt)) * projector(Ψ))    
        #V_ϕ = projector(dagger(cond_proj_t) * V_Ψ, dagger(ϕ)) 
        #V_ϕ = V_ϕ + dagger(V_ϕ)
    end
    if abs(Ec - quant_system.GLOB_EIG_E[index]) < 100
    fig, ax = subplots(1, 1, figsize=set_size("thesis", 1, (1, 1)), sharex=true)
    ax.plot(T, exp_RQM, label=L"\text{RQM}")
    ax.plot(T_, exp_semi, label=L"\text{Semi}")
    ax.set_xlabel(L"t")
    ax.set_ylabel(L"\langle \sigma_z \rangle")
    ax.legend()
    ax.grid(true, linestyle=":")
    entan = real(entanglement_entropy(Ψ, [2]))/log(2)
    fig.suptitle("Jaynes Cummings at CutOff N = $N_cutoff" 
        * "\n S = $(round(entan, digits=3)) E = $(round(quant_system.GLOB_EIG_E[index], digits=4)) Ec = $(round(Ec, digits=4))")
    savefig("data/LZsys/dynamics/index_$(index).png", dpi=300)
    close(fig)
    end
end
