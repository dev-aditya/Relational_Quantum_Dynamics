using QuantumOptics
using PyPlot
using PyCall
using LaTeXStrings
pyimport("scienceplots")
mpl = pyimport("matplotlib")
mpl.style.use(["science", "no-latex"])
using DelimitedFiles
hamiltonian = "JyaCumModel"
include("hamiltonian/$hamiltonian.jl")

if isdir("data/SpinBoson/$hamiltonian/entropy") == false
    mkdir("data/SpinBoson/$hamiltonian/entropy")
end

α_values = [50, 300, 700]
for i in eachindex(α_values)
    φ = (1 + √5)/2 
    α = sqrt(α_values[i]) * exp(im * φ)
    g = 2π*0.01
    V = ħ*g*((sigmap(spin_basis)⊗a) + (sigmam(spin_basis)⊗ad)) / abs(α)
    H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V
    GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
    function Entropy(Ψ::Ket)
        return real(entanglement_entropy(Ψ, [2]))/log(2)
    end

    ent = Entropy.(GLOB_EIG_V)
    data_mat = [GLOB_EIG_E ent]
    writedlm("data/SpinBoson/$hamiltonian/entropy/entropy_alpha_$(α_values[i]).dat", data_mat)
    println("α = $(α_values[i]) done")
end

fig, axs = subplots(length(α_values), 1, figsize=(8, 6*length(α_values)), sharex=true, sharey=true)

for i in eachindex(α_values)
    α = α_values[i]
    data = readdlm("data/SpinBoson/$hamiltonian/entropy/entropy_alpha_$(α).dat")
    energy = data[:, 1]
    entropy = data[:, 2]
    axs[i].hist2d(energy, entropy, bins=50, cmap="viridis")
    axs[i].set_title("Alpha = $(α)")
    axs[i].set_xlabel("Energy")
    axs[i].set_ylabel("Entropy")
end

tight_layout()
savefig("data/SpinBoson/$hamiltonian/entropy/entropy_profile.pdf")