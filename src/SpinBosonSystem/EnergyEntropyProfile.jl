using QuantumOptics
using PyPlot
using PyCall
using LaTeXStrings
pyimport("scienceplots")
mpl = pyimport("matplotlib")
mpl.style.use(["science"])

α = sqrt.([1, 6, 10, 100, 600, 1000])
include("hamiltonian/SpinBosonLinear.jl")
for i in eachindex(α)
    H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V/abs(α[i])
    GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
    function Entropy(Ψ::Ket)
        return real(entanglement_entropy(Ψ, [2]))/log(2)
    end

    ent = Entropy.(GLOB_EIG_V)

    # Create a new figure
    figure(figsize=(6, 8))

    # Plot the 2D histogram
    hist2D(GLOB_EIG_E, ent, bins=(300, 300), cmap="plasma", cmin=1)
    colorbar(orientation="horizontal")
    # Set labels and title
    xlabel("Energy")
    ylabel("Entropy")
    grid(true, linestyle=":")
    title("SpinBoson with linear coupling (x/" * L"\alpha" *") at \n CutOff N = $N_cutoff"  * L"\alpha ^2" *"= $(abs(α[i])^2)")
    PyPlot.savefig("data/SpinBoson/EnergyVsEntropyHist_alpha2=$(abs(α[i])^2).png", dpi=300)
    println("done")
    close("all")
end