include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Base.Threads
using Statistics
using PyCall
using LaTeXStrings
using PyPlot
using JLD
using DelimitedFiles
pyimport("scienceplots")
mpl = pyimport("matplotlib")
mpl.style.use(["science", "no-latex"])
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

hamiltonian = "JyaCumModel"
hamiltonian = "SpinBosonLinear"
if !isdir("data/SpinBoson/$hamiltonian")
    mkdir("data/SpinBoson/$hamiltonian")
end
if !isdir("data/SpinBoson/$hamiltonian/dynamics")
    mkdir("data/SpinBoson/$hamiltonian/dynamics")
end
if !isdir("data/SpinBoson/$hamiltonian/dynamics/semi")
    mkdir("data/SpinBoson/$hamiltonian/dynamics/semi")
end
if !isdir("data/SpinBoson/$hamiltonian/dynamics/quant")
    mkdir("data/SpinBoson/$hamiltonian/dynamics/quant")
end
if !isdir("data/SpinBoson/$hamiltonian/dynamics/overlap")
    mkdir("data/SpinBoson/$hamiltonian/dynamics/overlap")
end
include("hamiltonian/$hamiltonian.jl")
#[1, 6, 10, 100, 600, 900]
φ = (1 + √5)/2 
α = 100
α = sqrt(α) * exp(im * φ)
g = 2π*0.01/abs(α)
V = ħ*g*(tensor(sigmax(spin_basis), x))
#V = ħ*g*((sigmap(spin_basis)⊗a) + (sigmam(spin_basis)⊗ad))/abs(α)
quant_system = BosonQuantumSystem(Hs, Hc, V);
function χ(t::Float64)
    #return quant_system.HC_EIG_V[100] * exp(-im * quant_system.HC_EIG_E[100] * t)
    return coherentstate(clck_basis, exp(-im * ω0 * t)*α)
end

Ec = real(expect(Hc, χ(0.0)))
## Semiclassical Hamiltonian for quant_system
p = spinup(spin_basis)
m = spindown(spin_basis)
pp = projector(p, dagger(p))
mm = projector(m, dagger(m))
pm = projector(p, dagger(m))
mp = projector(m, dagger(p))
Ω = g*abs(α)
T = LinRange(0, 25, 2000)
#U(t) = cos(t*Ω)*(pp + mm) - im*sin(t*Ω)*(pm + mp)
U(t, p) = Hs + ħ*g*sigmax(spin_basis)*(α * exp(-im * ω0 * t) + conj(α * exp(-im * ω0 * t)))
global OPER = (sigmax(spin_basis) + sigmay(spin_basis))
for index in eachindex(quant_system.GLOB_EIG_E)
    println("index = $index")
    local Ψ = quant_system.GLOB_EIG_V[index]
    local psi_0 = (tensor(one(Hs), dagger(χ(0.0))) * Ψ)
    if abs(norm(psi_0)) == 0.0
        continue
    else
        psi_0 = normalize!(psi_0)
    end
    local overlap = zeros(length(T))
    local expect_Over_semi = zeros(length(T))
    local expect_Over_quant = zeros(length(expect_Over_semi))
    T_, psi_semi_t = timeevolution.schroedinger_dynamic(T, psi_0, U)
    @threads for i in eachindex(T)
        cond_proj_t = tensor(one(Hs), χ(T[i]))
        ϕ =  dagger(cond_proj_t) * Ψ
        #psi_semi = U(T[i]) * psi_0
        psi_semi = psi_semi_t[i]
        if abs(norm(ϕ)) == 0.0
            continue
        else
            ϕ = normalize!(ϕ)
        end
        if abs(norm(psi_semi)) == 0.0
            continue
        else
            psi_semi = normalize!(psi_semi)
        end
        expect_Over_quant[i] = real(expect(OPER, ϕ))
        expect_Over_semi[i] = real(expect(OPER, psi_semi))
        overlap[i] = abs2(dagger(ϕ)* psi_semi)
        #exp_RQM[i] = real(expect(OPER, ϕ))
        #E_RQM[i] = tr(quant_system.H * tensor(one(Hs), projector(Χt)) * projector(Ψ)) / tr(tensor(one(Hs), projector(Χt)) * projector(Ψ))    
        #V_ϕ = projector(dagger(cond_proj_t) * V_Ψ, dagger(ϕ)) 
        #V_ϕ = V_ϕ + dagger(V_ϕ)
    end
    writedlm("data/SpinBoson/$hamiltonian/dynamics/semi/index_$(index)_OPER.txt", expect_Over_semi)
    writedlm("data/SpinBoson/$hamiltonian/dynamics/quant/index_$(index)_OPER.txt", expect_Over_quant)
    writedlm("data/SpinBoson/$hamiltonian/dynamics/overlap/index_$(index)_overlap.txt", overlap)
end

# Find the index where the absolute difference between Ec and E is the smallest
_, min_index = findmin(abs.(Ec .- quant_system.GLOB_EIG_E))

# Define the range of indices for the plot
index_range = max(1, min_index - 200):min(length(quant_system.GLOB_EIG_E), min_index + 500)

# Create the subplots
x, y = set_size("thesis", 1, (3, 2))
fig, axs = subplots(3, 2, figsize=(x, y), sharex=true,)
function find_nearest_index(lst, target)
    return argmin(abs.(lst .- target))
end
#T = LinRange(0, 4, 2000)
#T = T[1:find_nearest_index(T, 1.0)]
# Plot abs2(c1_rqm_index) and abs2(c1_semi_index) vs T for each index in the range
for (i, index) in enumerate([index_range[1], min_index, index_range[end]])
    @show i, index
    c1_rqm = readdlm("data/SpinBoson/$hamiltonian/dynamics/quant/index_$(index)_OPER.txt")
    c1_semi = readdlm("data/SpinBoson/$hamiltonian/dynamics/semi/index_$(index)_OPER.txt")
    overlap = round.(readdlm("data/SpinBoson/$hamiltonian/dynamics/overlap/index_$(index)_overlap.txt"), digits=3)
    #c1_rqm = c1_rqm[1:length(T)]
    #c1_semi = c1_semi[1:length(T)]
    #overlap = overlap[1:length(T)]
    axs[i, 1].plot(T, (c1_rqm), label="RQM", linestyle="-")
    axs[i, 1].plot(T, (c1_semi), label="SEMI", linestyle="-")
    axs[i, 1].set_xlabel(L"t", fontsize=12)
    axs[i, 1].set_ylabel(L"\left\langle \psi |\sigma_x + \sigma_y | \psi \right\rangle", fontsize=12)
    axs[i, 1].legend(fontsize=12)
    E_glob = quant_system.GLOB_EIG_E[index]
    axs[i, 1].set_title(L"E_{glob} = "*string(round(E_glob, digits=3)), fontsize=18)
    
    axs[i, 2].plot(T, overlap, label="Overlap")
    axs[i, 2].set_xlabel(L"t", fontsize=12)
    axs[i, 2].set_ylabel(L"\left\langle \psi_{rel} | \psi_{semi} \right\rangle", fontsize=12)
    axs[i, 2].set_title(L"E_{c} = "*string(round(Ec, digits=3)), fontsize=18)
    axs[i, 2].legend(fontsize=12)
end

for ax in axs
    ax.grid(linestyle=":", color="lightgrey")
    ax.tick_params(axis="both", labelsize=12)
end
# Increase the space between subplots
subplots_adjust(hspace=0.5)
#suptitle(L"V = \hbar \Omega (\sigma_+ a+   \sigma_-a^\dagger) / |\alpha|",)
suptitle(L"V = \hbar \Omega (\sigma_x a+   \sigma_x a^\dagger) / |\alpha|", fontsize=18)
tight_layout()
savefig("data/SpinBoson/$hamiltonian/dynamics/semi_vs_quant.pdf")