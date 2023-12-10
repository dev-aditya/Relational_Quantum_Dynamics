using QuantumOptics
using PyPlot

include("hamiltonian/CoupledHarmonicOscillator.jl")
H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V
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
grid(true)
title("CoupledHarmonicOscillator")
PyPlot.savefig("data/EntropyEnergy-CoupledHarmonicOscillator_hist.png", dpi=300)
println("done")