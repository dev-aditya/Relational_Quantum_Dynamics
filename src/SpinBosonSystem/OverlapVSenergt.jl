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
mpl.style.use(["science","no-latex"])
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

α_values = [50, 300, 700]  # Add the alpha values you want to simulate
include("hamiltonian/$hamiltonian.jl")

p = spinup(spin_basis)
m = spindown(spin_basis)
pp = projector(p, dagger(p))
mm = projector(m, dagger(m))
pm = projector(p, dagger(m))
mp = projector(m, dagger(p))

quant_system = BosonQuantumSystem(Hs, Hc,  0*(tensor(sigmax(spin_basis), identity(Hc))))
global OVERLAP_avg = zeros(Float64, length(quant_system.GLOB_EIG_E), length(α_values))
global OVERLAP_var = zeros(Float64, length(quant_system.GLOB_EIG_E), length(α_values))
for (alpha_index, α) in enumerate(α_values)
    println("Running simulation for alpha = $α")
    φ = (1 + √5)/2 
    α = sqrt(α) * exp(im * φ)
    g = 2π*0.01/abs(α)
    #V = ħ*Ω*((sigmap(spin_basis)⊗a) + (sigmam(spin_basis)⊗ad)) / abs(α)
    V = ħ*g*(tensor(sigmax(spin_basis), x))/abs(α)
    quant_system = BosonQuantumSystem(Hs, Hc, V)
    function χ(t)
        return coherentstate(basis(quant_system.Hc), α * exp(-im * ω0 * t))
    end
    #U(t) = cos(t*Ω)*(exp(-im*ω0*t/2)*pp + exp(im*ω0*t/2)*mm) - im*sin(t*Ω)*(exp(-im*ω0*t/2)*pm + exp(im*ω0*t/2)*mp)
    U(t, p) = Hs + ħ*g*sigmax(spin_basis)*(α * exp(-im * ω0 * t) + conj(α * exp(-im * ω0 * t)))
    for index in eachindex(quant_system.GLOB_EIG_E)
        T = LinRange(0, 4*pi/Ω, 2000)
        println("index = $index")
        local Ψ = quant_system.GLOB_EIG_V[index]
        local overlap = Vector{Float64}(undef, length(T))
        local psi_0 = (tensor(one(Hs), dagger(χ(0.0))) * Ψ)
        if abs(norm(psi_0)) == 0.0
            continue
        else
            psi_0 = normalize!(psi_0)
        end
        local overlap = Vector{Float64}(undef, length(T))
        T_, psi_semi_t = timeevolution.schroedinger_dynamic(T, psi_0, U)
        @threads for i in eachindex(T)
            cond_proj_t = tensor(one(Hs), χ(T[i]))
            ϕ =  dagger(cond_proj_t) * Ψ
            #psi_semi = U(T[i]) * psi_0
            psi_semi = psi_semi_t[i]
            if abs(norm(ϕ)) == 0.0 || abs(norm(psi_semi)) == 0.0
                overlap[i] = abs(dagger(ϕ) * psi_semi)
            else
                overlap[i] = abs(dagger(normalize!(ϕ)) * normalize!(psi_semi))
            end
        end
        OVERLAP_avg[index, alpha_index] = mean(overlap)
        OVERLAP_var[index, alpha_index] = var(overlap)
    end
end
# Write OVERLAP_avg to a text file
writedlm("data/SpinBoson/$hamiltonian/OVERLAP_avg.txt", OVERLAP_avg)

writedlm("data/SpinBoson/$hamiltonian/OVERLAP_var.txt", OVERLAP_var)
println("-------------- \n Simulation completed \n ===================")
# Read the data from the text files
OVERLAP_avg = readdlm("data/SpinBoson/$hamiltonian/OVERLAP_avg.txt")
OVERLAP_var = readdlm("data/SpinBoson/$hamiltonian/OVERLAP_var.txt")

fig_size = set_size("thesis", 1, (length(α_values), 2))
fig, axs = subplots(length(α_values), 2, figsize=fig_size, sharex=true)

for (alpha_index, α) in enumerate(α_values)
    @show alpha_index
    axs[alpha_index, 1].plot(quant_system.GLOB_EIG_E, OVERLAP_avg[:, alpha_index], label=L"\alpha^2 =" * "$α", color="black", linestyle="-.")
    axs[alpha_index, 1].set_ylabel(L"\mathrm{avg}\left( \left| \left\langle \psi_{\text{rel}} | \psi_{\text{semi}} \right\rangle \right| \right)",fontsize=12)
    axs[alpha_index, 1].legend()
    if alpha_index == length(α_values)
        axs[alpha_index, 1].set_xlabel(L"E_{\text{global}}")
    end
    axs[alpha_index, 1].grid(color="gray", linestyle=":", linewidth=0.5)
    axs[alpha_index, 1].tick_params(axis="both", which="major", labelsize=12)  # Set tick font size
    
    axs[alpha_index, 2].plot(quant_system.GLOB_EIG_E, OVERLAP_var[:, alpha_index], label=L"\alpha^2 =" * "$α", color="black", linestyle="-.")
    if alpha_index == length(α_values)
        axs[alpha_index, 2].set_xlabel(L"E_{\text{global}}",fontsize=12)
    end
    axs[alpha_index, 2].set_ylabel(L"\mathrm{var}\left( \left| \left\langle \psi_{\text{rel}} |\psi_{\text{semi}} \right\rangle \right| \right)",fontsize=12)
    axs[alpha_index, 2].legend()
    axs[alpha_index, 2].grid(color="gray", linestyle=":", linewidth=0.5)
    axs[alpha_index, 2].tick_params(axis="both", which="major", labelsize=12)  # Set tick font size
    
    # Calculate expectation value and draw horizontal line
    energy = real(expect(Hc, coherentstate(basis(quant_system.Hc), sqrt(α))))
    axs[alpha_index, 1].axvline(x=energy, color="red")
    axs[alpha_index, 2].axvline(x=energy, color="red")
end

# Set title and caption font size
suptitle(L"V = \hbar \Omega (\sigma_x a+   \sigma_x a^\dagger) / |\alpha|", fontsize=18)
tight_layout()
savefig("data/SpinBoson/$hamiltonian/OverlapVStime.pdf", dpi=300,)
